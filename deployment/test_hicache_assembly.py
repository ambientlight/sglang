"""Exercise the real assembler on tiny CPU tensors, with allocation mocked.

Run in the pinned image with runc/NVIDIA_VISIBLE_DEVICES=void and the two
reviewed source overrides mounted. No GPU or large host buffers are allocated.
"""

import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembly
from sglang.srt.mem_cache.hybrid_cache.dsv41_host_budget import page_costs


class Logical:
    def __init__(self, size, page_size, layout):
        self.size = self.logical_size = size
        self.page_size, self.layout, self.device = page_size, layout, "cpu"
        self.kv_buffer = None
        self.free_slots = NS(nbytes=size * 8)
        self.can_use_write_back_jit = True


class Paged(Logical):
    def __init__(self, *, device_buffers, item_bytes, num_host_pages, slot_page_size, layout, **kwargs):
        super().__init__(num_host_pages * slot_page_size, slot_page_size, layout)
        self.kv_buffer = NS(nbytes=len(device_buffers) * item_bytes * num_host_pages)


class AssemblyTest(unittest.TestCase):
    def test_real_low_ratio_shapes_match_bounded_plan_and_allocated_total(self):
        self.exercise_layout(None)

    def test_payload_split_with_real_assembler_and_packed_draft(self):
        self.exercise_layout("0.236")

    def exercise_layout(self, fraction):
        swa = lambda n: NS(
            kv_buffer=[torch.zeros((30, 2048), dtype=torch.uint8) for _ in range(n)], bytes_per_page_padded=2048
        )
        indexes = {
            r: NS(
                page_size=64,
                index_k_with_scale_buffer=None,
                index_k_payload_buffer=[torch.zeros((80, 2, 64), dtype=torch.uint8) for _ in range(n)],
                index_k_scale_buffer=[torch.zeros((80, 2, 1), dtype=torch.float16) for _ in range(n)],
            )
            for r, n in ((1, 1), (2, 3))
        }
        kv = NS(
            start_layer=0,
            end_layer=40,
            size=5120,
            swa_size=640,
            swa_page_size=64,
            layer_mapping=[NS(compress_ratio=1) for _ in range(40)],
            sources_by_ratio={1: [20], 2: [2, 8, 14]},
            kv_pools={
                1: NS(kv_buffer=[torch.zeros((20, 1024), dtype=torch.uint8)]),
                2: NS(kv_buffer=[torch.zeros((20, 512), dtype=torch.uint8) for _ in range(3)]),
            },
            index_pools=indexes,
            swa_kv_pool=swa(40),
        )
        draft = [NS(swa_kv_pool=swa(3))]
        full_bytes, swa_bytes, _, _ = page_costs(kv, 256, draft)
        params = NS(
            page_size=256,
            mtp_draft_device_pools=draft,
            token_to_kv_pool_allocator=NS(size_full=5120, swa_attn_allocator=NS(alloc=None, free=None)),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        memory = NS(
            hicache_size=0,
            hicache_ratio=2,
            hicache_mem_layout="page_first",
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_host_memory_mode="cache",
        )
        # This synthetic layout has much more slot metadata per payload byte
        # than production. The explicit split still must fit one GPU equivalent.
        cap = 1_500_000_000 if fraction is not None else 1_100_000_000
        env = {"DSV41_HICACHE_HOST_BUDGET_BYTES": str(cap)}
        if fraction is not None:
            env["DSV41_HICACHE_SWA_FRACTION"] = fraction
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(assembly, "get_memory", return_value=memory),
            patch.object(assembly, "get_parallel", return_value=NS(nnodes=1, tp_size=4, pp_size=1, tp_rank=0)),
            patch("sglang.srt.mem_cache.pool_host.base.ranks_per_host", return_value=4),
            patch.object(assembly, "_get_allocator_type", return_value="default"),
            patch.object(assembly, "LogicalHostPool", Logical),
            patch.object(assembly, "DeepSeekV4PagedHostPool", Paged),
            patch.object(assembly, "HybridCacheController"),
        ):
            group, _ = assembly.build_deepseek_v4_hicache_stack(
                params=params, kvcache=kv, load_cache_event=None, storage_backend=None
            )
        self.assertEqual(len(group.entries), 8)  # logical, SWA, two × (KV/payload/scale)
        full_pages = group.entries[0].host_pool.size // 256
        swa_pages = group.entries[1].host_pool.size // 64
        actual_payload = sum(e.host_pool.kv_buffer.nbytes for e in group.entries if e.host_pool.kv_buffer)
        self.assertEqual(actual_payload, full_bytes * full_pages + swa_bytes * swa_pages)
        if fraction is not None:
            self.assertAlmostEqual(swa_bytes * swa_pages / actual_payload, float(fraction), delta=0.02)
        self.assertLessEqual(
            sum(
                (e.host_pool.kv_buffer.nbytes if e.host_pool.kv_buffer else 0) + e.host_pool.free_slots.nbytes
                for e in group.entries
            )
            * 4,
            cap,
        )


if __name__ == "__main__":
    unittest.main()
