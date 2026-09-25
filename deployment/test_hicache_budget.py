"""CPU-only budget arithmetic and integration checks; never allocate model KV."""

import ast
from fractions import Fraction
from math import ceil as math_ceil
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


DIRECTORY = Path(__file__).resolve().parents[1] / "python/sglang/srt/mem_cache/hybrid_cache"
spec = importlib.util.spec_from_file_location("budget", DIRECTORY / "dsv41_host_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class Buffer:
    def __init__(self, shape, itemsize=1):
        self.shape, self.itemsize = shape, itemsize

    def element_size(self):
        return self.itemsize


def fixture(fused=False):
    swa = lambda layers: NS(kv_buffer=[Buffer((100, 2048)) for _ in range(layers)], bytes_per_page_padded=2048)
    indexes = {}
    for ratio, layers in ((1, 1), (2, 3)):
        indexes[ratio] = NS(
            page_size=64,
            index_k_with_scale_buffer=[Buffer((80, 132)) for _ in range(layers)] if fused else None,
            index_k_payload_buffer=[Buffer((80, 2, 64)) for _ in range(layers)],
            index_k_scale_buffer=[Buffer((80, 2, 1), 2) for _ in range(layers)],
        )
    return NS(
        sources_by_ratio={1: [20], 2: [2, 8, 14]},
        kv_pools={1: NS(kv_buffer=[Buffer((20, 1024))]), 2: NS(kv_buffer=[Buffer((20, 512)) for _ in range(3)])},
        index_pools=indexes,
        swa_kv_pool=swa(40),
        swa_page_size=64,
        size=256000,
        swa_size=6400,
    ), [NS(swa_kv_pool=swa(3))]


class BudgetTest(unittest.TestCase):
    def live_values(self):
        return dict(budget_bytes=256_000_000_000, ranks=4, full_pages=12545, swa_pages=102,
                    full_page_bytes=417920, swa_page_bytes=6439680,
                    full_metadata_page_bytes=30720, swa_metadata_page_bytes=6144)

    def test_67_percent_payload_split_has_exact_projected_capacity(self):
        plan = budget.plan_pages(**self.live_values(), swa_fraction="0.67")
        self.assertEqual(plan["full_host_pages"] * 256, 12_570_368)
        self.assertEqual(plan["swa_host_pages"] * 256, 1_656_064)
        self.assertEqual(plan["planned_bytes_host_total"], 255_984_163_328)
        self.assertLessEqual(plan["planned_bytes_host_total"], 256_000_000_000)

    def test_explicit_payload_split_stays_within_budget_and_is_gpu_independent(self):
        values = self.live_values()
        old = budget.plan_pages(**values)
        plan = budget.plan_pages(**values, swa_fraction="0.236")
        self.assertEqual(plan["allocation_mode"], "payload_fraction")
        self.assertEqual(plan["requested_swa_payload_fraction"], 0.236)
        self.assertAlmostEqual(plan["actual_swa_payload_fraction"], 0.236, delta=0.0001)
        self.assertLessEqual(plan["actual_swa_payload_fraction"], 0.236)
        self.assertLess(plan["full_host_pages"], old["full_host_pages"])
        self.assertGreater(plan["swa_host_pages"], 2 * old["swa_host_pages"])
        self.assertLessEqual(plan["planned_bytes_host_total"], values["budget_bytes"])
        for df, ds in ((11729, 408), (13761, 102), (15000, 128)):
            changed = budget.plan_pages(**dict(values, full_pages=df, swa_pages=ds), swa_fraction="0.236")
            for key in ("full_host_pages", "swa_host_pages", "planned_bytes_host_total"):
                self.assertEqual(changed[key], plan[key])

    def test_explicit_split_rounding_is_maximal_and_keeps_device_equivalents(self):
        rng = random.Random(23)
        for _ in range(500):
            df, ds, fb, sb = [rng.randint(1, 1000) for _ in range(4)]
            fm, sm = rng.randint(0, 100), rng.randint(0, 100)
            fraction = Fraction(rng.randint(1, 999), 1000)
            needed_full = max(df, math_ceil(ds * sb * (1 - fraction) / (fb * fraction)))
            def usage(pages):
                swa = int(pages * fb * fraction / (sb * (1 - fraction)))
                return 4 * (pages * (fb + fm) + swa * (sb + sm)) + budget.METADATA_RESERVE_BYTES
            cap = usage(needed_full) + rng.randint(0, 1_000_000)
            plan = budget.plan_pages(budget_bytes=cap, ranks=4, full_pages=df, swa_pages=ds,
                                     full_page_bytes=fb, swa_page_bytes=sb,
                                     full_metadata_page_bytes=fm, swa_metadata_page_bytes=sm,
                                     swa_fraction=str(fraction))
            self.assertGreaterEqual(plan["full_host_pages"], df)
            self.assertGreaterEqual(plan["swa_host_pages"], ds)
            self.assertLessEqual(plan["planned_bytes_host_total"], cap)
            self.assertGreater(usage(plan["full_host_pages"] + 1), cap)
            self.assertLessEqual(Fraction(plan["swa_tensor_bytes_per_rank"], plan["tensor_bytes_per_rank"]), fraction)

    def test_fraction_invalid_and_split_too_small_fail_closed(self):
        for value in ("", "no", "nan", "inf", "-0.1", "0", "1", "1.1", True, "1/0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.plan_pages(**self.live_values(), swa_fraction=value)
        for value in ("0.00001", "0.99999"):
            with self.assertRaisesRegex(ValueError, "device-cache equivalent"):
                budget.plan_pages(**self.live_values(), swa_fraction=value)

    def test_float_fraction_matches_decimal_string(self):
        self.assertEqual(budget.plan_pages(**self.live_values(), swa_fraction=0.236),
                         budget.plan_pages(**self.live_values(), swa_fraction="0.236"))

    def test_live_192_and_projected_256_keep_exact_host_total_bounds(self):
        values = dict(
            ranks=4,
            full_pages=13068,
            swa_pages=102,
            full_page_bytes=417920,
            swa_page_bytes=6439680,
            full_metadata_page_bytes=30720,
            swa_metadata_page_bytes=6144,
        )
        old = budget.plan_pages(budget_bytes=192_000_000_000, **values)
        new = budget.plan_pages(budget_bytes=256_000_000_000, **values)
        self.assertEqual(old["full_host_pages"], 95673)
        self.assertEqual(new["full_host_pages"] * 256, 32699648)
        self.assertLessEqual(new["planned_bytes_host_total"], 256_000_000_000)
        self.assertGreater(new["full_host_pages"], old["full_host_pages"])
        actual = new["tensor_bytes_host_total"] + new["slot_metadata_bytes_per_rank"] // 3 * 4
        self.assertLess(actual, 256_000_000_000)

    def test_split_fp4_shared_sources_and_packed_draft_counted(self):
        kv, draft = fixture()
        self.assertEqual(budget.page_costs(kv, 256, draft), (3880, 43 * 2048, 7 * 256 * 24, 64 * 24))

    def test_fused_indexer_has_same_payload_but_fewer_metadata_pools(self):
        kv, draft = fixture(True)
        self.assertEqual(budget.page_costs(kv, 256, draft), (3880, 43 * 2048, 5 * 256 * 24, 64 * 24))

    def test_unsupported_pools_fail_closed(self):
        for change in ({"_unified_kv": True}, {"sources_by_ratio": {4: [2]}}, {"c128_kv_pool": object()}):
            kv, draft = fixture()
            kv.__dict__.update(change)
            with self.assertRaises(ValueError):
                budget.page_costs(kv, 256, draft)

    def test_no_draft_or_misaligned_index_pages_rejected(self):
        kv, draft = fixture()
        with self.assertRaises(ValueError):
            budget.page_costs(kv, 256, [])
        kv.index_pools[1].page_size = 100
        with self.assertRaises(ValueError):
            budget.page_costs(kv, 256, draft)

    def test_budget_is_host_total_and_accounts_for_metadata(self):
        plan = budget.plan_pages(
            budget_bytes=192_000_000_000,
            ranks=4,
            full_pages=11729,
            swa_pages=408,
            full_page_bytes=450560,
            swa_page_bytes=1761280,
            full_metadata_page_bytes=43008,
            swa_metadata_page_bytes=1536,
        )
        self.assertLessEqual(plan["planned_bytes_host_total"], 192_000_000_000)
        self.assertEqual(plan["tensor_bytes_host_total"], 4 * plan["tensor_bytes_per_rank"])
        self.assertGreater(plan["slot_metadata_bytes_per_rank"], 0)
        self.assertLess(plan["tensor_bytes_host_total"], plan["planned_bytes_host_total"])

    def test_page_rounding_never_exceeds_budget_and_is_maximal(self):
        rng = random.Random(7)
        for _ in range(500):
            df, ds, fb, sb = [rng.randint(1, 10000) for _ in range(4)]
            cap = 1024**3 + 4 * (df * fb + ds * sb) * rng.randint(1, 10)
            plan = budget.plan_pages(
                budget_bytes=cap, ranks=4, full_pages=df, swa_pages=ds, full_page_bytes=fb, swa_page_bytes=sb
            )
            self.assertLessEqual(plan["planned_bytes_host_total"], cap)
            extra = plan["full_host_pages"] + 1
            self.assertGreater(4 * (extra * fb + (extra * ds // df) * sb) + 1024**3, cap)

    def test_invalid_or_too_small_budgets_rejected(self):
        for override in ({"ranks": 1}, {"ranks": 8}, {"budget_bytes": -1}, {"budget_bytes": 100}, {"full_pages": 0}):
            values = dict(
                budget_bytes=192_000_000_000,
                ranks=4,
                full_pages=100,
                swa_pages=10,
                full_page_bytes=1024,
                swa_page_bytes=2048,
            )
            values.update(override)
            with self.assertRaises(ValueError):
                budget.plan_pages(**values)

    def sizing_function(self, memory):
        source = ast.parse((DIRECTORY / "hybrid_pool_assembler.py").read_text())
        node = next(
            n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_deepseek_v4_num_host_pages"
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        scope = dict(
            os=os,
            json=json,
            logger=logging.getLogger(__name__),
            get_memory=lambda: memory,
            get_parallel=lambda: NS(nnodes=1, tp_size=4, pp_size=1, tp_rank=0),
        )
        exec(compile(module, "<sizing-function>", "exec"), scope)
        return scope["_deepseek_v4_num_host_pages"]

    def test_original_ratio_path_unchanged_when_budget_absent(self):
        kv, draft = fixture()
        fn = self.sizing_function(NS(hicache_size=0, hicache_ratio=2.5))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                fn(
                    params=NS(token_to_kv_pool_allocator=NS(size_full=256000)),
                    kvcache=kv,
                    page_size=256,
                    swa_page_size=64,
                ),
                (2500, 250),
            )

    def test_fraction_without_host_budget_rejected(self):
        kv, _ = fixture()
        fn = self.sizing_function(NS(hicache_size=0, hicache_ratio=2.5))
        with patch.dict(os.environ, {"DSV41_HICACHE_SWA_FRACTION": "0.236"}, clear=True):
            with self.assertRaisesRegex(ValueError, "nonzero host budget"):
                fn(params=NS(token_to_kv_pool_allocator=NS(size_full=256000)), kvcache=kv,
                   page_size=256, swa_page_size=64)

    def test_fraction_reaches_real_assembler_sizing(self):
        kv, draft = fixture()
        fn = self.sizing_function(NS(hicache_size=0, hicache_ratio=2))
        modules = {"sglang.srt.mem_cache.hybrid_cache.dsv41_host_budget": budget,
                   "sglang.srt.mem_cache.pool_host.base": NS(ranks_per_host=lambda: 4)}
        fb, sb, fm, sm = budget.page_costs(kv, 256, draft)
        expected = budget.plan_pages(budget_bytes=256_000_000_000, ranks=4, full_pages=1000, swa_pages=100,
                                     full_page_bytes=fb, swa_page_bytes=sb, full_metadata_page_bytes=fm,
                                     swa_metadata_page_bytes=sm, swa_fraction="0.236")
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
            "DSV41_HICACHE_HOST_BUDGET_BYTES": "256000000000", "DSV41_HICACHE_SWA_FRACTION": "0.236"}, clear=True):
            actual = fn(params=NS(token_to_kv_pool_allocator=NS(size_full=256000), mtp_draft_device_pools=draft),
                        kvcache=kv, page_size=256, swa_page_size=64)
        self.assertEqual(actual, (expected["full_host_pages"], expected["swa_host_pages"]))

    def test_bounded_sizing_is_wired_and_generic_size_still_rejected(self):
        kv, draft = fixture()
        params = NS(token_to_kv_pool_allocator=NS(size_full=256000), mtp_draft_device_pools=draft)
        memory = NS(hicache_size=0, hicache_ratio=2)
        fn = self.sizing_function(memory)
        modules = {
            "sglang.srt.mem_cache.hybrid_cache.dsv41_host_budget": budget,
            "sglang.srt.mem_cache.pool_host.base": NS(ranks_per_host=lambda: 4),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"DSV41_HICACHE_HOST_BUDGET_BYTES": "192000000000"}),
        ):
            result = fn(params=params, kvcache=kv, page_size=256, swa_page_size=64)
            self.assertGreater(result[0], 1000)
            self.assertGreater(result[1], 100)
            memory.hicache_size = 192
            with self.assertRaisesRegex(ValueError, "does not support"):
                fn(params=params, kvcache=kv, page_size=256, swa_page_size=64)


if __name__ == "__main__":
    unittest.main()
