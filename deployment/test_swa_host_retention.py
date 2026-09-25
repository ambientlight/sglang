"""CPU regressions using the deployed Python TreeCore, not a replacement cache.

The pressure fixture has the captured 256k FULL / 24,064 SWA slots per prompt
and the exact 254,976-slot host SWA capacity. It models transfers with integer
payloads; no model, CUDA kernel, or production cache is touched.
"""

from collections import deque
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    SWARebuild,
)
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore


F, S = ComponentType.FULL, ComponentType.SWA
FLAG = "SGLANG_OPT_HICACHE_SWA_RETAIN_REQUEST_BOUNDARIES"


class CPUAllocator(SWATokenToKVPoolAllocator):
    def __init__(self):
        self.device = "cpu"

    def translate_loc_from_full_to_swa(self, indices):
        return indices.clone()


class HostSlots:
    """Bounded host allocator; detects double free and payload misalignment."""

    def __init__(self, size, page):
        self.size = self.logical_size = size
        self.page = page
        self.free_pages = deque(range(size // page))
        self.allocated = set()
        self.payload = torch.full((size,), -1, dtype=torch.int64)

    def available_size(self):
        return len(self.free_pages) * self.page

    def alloc(self, size):
        assert size % self.page == 0
        if size > self.available_size():
            return None
        pages = [self.free_pages.popleft() for _ in range(size // self.page)]
        assert not self.allocated.intersection(pages)
        self.allocated.update(pages)
        return (
            torch.tensor(pages)[:, None] * self.page + torch.arange(self.page)
        ).flatten()

    def free(self, indices):
        assert len(indices) % self.page == 0
        pages = (indices[:: self.page] // self.page).tolist()
        assert len(set(pages)) == len(pages)
        assert set(pages) <= self.allocated
        self.allocated.difference_update(pages)
        self.free_pages.extend(pages)
        self.payload[indices] = -1


class CacheFixture:
    def __init__(
        self,
        retain=True,
        capacity=254976,
        window=128,
        page=256,
        sessions=False,
        mode="cache",
    ):
        alloc = CPUAllocator()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=alloc,
            page_size=page,
            sliding_window_size=window,
            enable_session_radix_cache=sessions,
        )
        self.cache = NS(
            enable_session_radix_cache=sessions,
            token_to_kv_pool_allocator=alloc,
            host_memory_mode=mode,
            page_size=page,
        )
        with patch.dict(
            os.environ,
            {
                FLAG: str(int(retain)),
                "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
            },
        ):
            self.swa = SWAComponent(self.cache, params)
        self.full = FullComponent(self.cache, params)
        self.tree = UnifiedTreeCore(params, {F: self.full, S: self.swa})
        self.tree.enable_hicache = self.tree.has_swa_host_pool = True
        self.tree.write_through_threshold = 1
        self.cache.tree_core = self.tree
        self.cache.tree_components = (F, S)
        self.pool = HostSlots(capacity, page)
        self.swa._swa_kv_pool_host = self.pool

    def node(self, parent, key, values, swa=True):
        node = self.tree._add_new_node(parent, RadixKey(key, None), values)
        if swa:
            self.tree.set_component_device_value(node.id, S, values.clone())
        return node

    def mark(self, node, chunked=False):
        self.swa.commit_insert_component_data(
            node, False, InsertParams(chunked=chunked), NS(), []
        )

    def reclaim(self, count):
        result = self.tree.drive_host_eviction(S, count)
        self.drain(result)
        return result

    def drain(self, result):
        for indices in result.host_frees.pop(S, []):
            self.pool.free(indices)
        # Full and device indices are synthetic addresses, not allocated pools.
        result.host_frees.clear()
        result.device_frees.clear()

    def backup(self, node):
        full, transfers = self.tree.build_backup_spec(node.id)
        for xfer in transfers.get(S, []):
            slots = self.pool.alloc(len(xfer.device_indices))
            if slots is None:
                self.reclaim(len(xfer.device_indices))
                slots = self.pool.alloc(len(xfer.device_indices))
            assert slots is not None, "bounded host allocation could not reclaim"
            self.pool.payload[slots] = xfer.device_indices
            xfer.host_indices = slots
        self.tree.commit_backup(node.id, full.clone(), transfers)
        return full, transfers

    def insert(self, tokens, chunked=False, swa_evicted_seqlen=0):
        """Drive real insert barriers; only the payload transfers are simulated."""
        step = self.tree.begin_insert(
            InsertParams(
                key=RadixKey(tokens, None),
                value=torch.arange(len(tokens)),
                chunked=chunked,
                swa_evicted_seqlen=swa_evicted_seqlen,
            )
        )
        while True:
            for action in step.actions:
                if isinstance(action, SWARebuild):
                    self.swa.apply_component_action(action)
                elif isinstance(action, BackupKV):
                    for node_id in action.node_ids:
                        self.backup(self.tree.node_by_id(node_id))
                else:
                    # Request-side indices are virtual CPU addresses. No real
                    # allocator owns them, unlike the bounded HostSlots above.
                    assert isinstance(action, (FreeDeviceKV, FreeDeviceKVFullOnly))
            if step.result is not None:
                return step.result
            step = self.tree.resume_insert()

    def prompt(self, case, length=256000, swa_slots=24064):
        page = self.cache.page_size
        base = (case + 1) * length
        root = self.tree.root_node
        parent = root
        nodes = []
        if length == 256000 and swa_slots == 24064 and page == 256:
            # Two page-aligned windows at each 2048-token chunk end, as in
            # cache_unfinished_req's (key_len - 1) retention boundary. Older
            # device checkpoints were already evicted before final backup.
            chunks = swa_slots // (2 * page)
            segments = [(length - chunks * 2048, False)]
            segments += [(1536, False), (page, True), (page, True)] * chunks
        else:
            segments = [(length - swa_slots, False)] + [(page, True)] * (
                swa_slots // page
            )
        start = 0
        for size, has_swa in segments:
            if not size:
                continue
            parent = self.node(
                parent,
                [case + 1] * size,
                torch.arange(base + start, base + start + size),
                swa=has_swa,
            )
            nodes.append(parent)
            start += size
        self.mark(parent)
        for node in nodes:
            self.backup(node)
        # Force Full KV off device too. A successful match must be host-backed.
        for node in reversed(nodes):
            self.drain(self.tree.demote(node.id))
        return nodes

    def match(self, case, length=256000):
        key = RadixKey([case + 1] * (length - self.cache.page_size), None)
        return self.tree.match_prefix(MatchPrefixParams(key=key))


class RetentionTests(unittest.TestCase):
    def test_64_long_prefixes_remain_reusable_after_full_device_eviction(self):
        f = CacheFixture()
        for case in range(64):
            f.prompt(case)
            self.assertLessEqual(len(f.pool.allocated) * 256, 254976)
        for case in range(64):
            result = f.match(case)
            self.assertEqual(result.host_hit_length, 255744)
            self.assertEqual(len(result.device_indices), 0)
            full, aux = f.tree.build_load_back_spec(result.best_match_node)
            expected = torch.arange((case + 1) * 256000, (case + 1) * 256000 + 255744)
            torch.testing.assert_close(full.host_indices, expected)
            self.assertIn(S, aux)
            swa_indices = aux[S][0].host_indices
            torch.testing.assert_close(
                f.pool.payload[swa_indices], expected[-len(swa_indices) :]
            )
        self.assertGreater(f.swa.host_evicted_tokens, 0)
        self.assertEqual(f.swa.host_boundary_evicted_tokens, 0)

    def test_legacy_eviction_reproduces_older_replay_miss(self):
        f = CacheFixture(retain=False)
        for case in range(16):
            f.prompt(case)
        result = f.match(0)
        self.assertEqual(len(result.device_indices) + result.host_hit_length, 0)
        self.assertGreater(result.full_kv_hit_length, 250000)
        self.assertEqual(f.match(15).host_hit_length, 255744)

    def test_boundary_mark_includes_replay_lookbehind_but_not_chunks(self):
        f = CacheFixture(capacity=4096)
        nodes = f.prompt(0, length=2048, swa_slots=2048)
        marks = [
            bool(n.component_data[S].metadata.get("host_request_boundary"))
            for n in nodes
        ]
        self.assertEqual(marks, [False] * 6 + [True, True])
        for n in nodes:
            n.component_data[S].metadata.pop("host_request_boundary", None)
        f.mark(nodes[-1], chunked=True)
        self.assertFalse(
            any(
                n.component_data[S].metadata.get("host_request_boundary") for n in nodes
            )
        )

    def test_last_resort_eviction_can_drain_retained_windows(self):
        f = CacheFixture(capacity=1024)
        for case in range(6):
            f.prompt(case, length=512, swa_slots=512)
        self.assertGreater(f.swa.host_boundary_evicted_tokens, 0)
        result = f.reclaim(1024)
        self.assertEqual(result.tracker[S], 1024)
        self.assertEqual(f.pool.available_size(), 1024)

    def test_session_priority_wins_over_boundary_preference(self):
        f = CacheFixture(capacity=4096, sessions=True)
        unreferenced = f.prompt(0, length=256, swa_slots=256)[0]
        referenced = f.prompt(1, length=256, swa_slots=256)[0]
        referenced.component_data[S].metadata.pop("host_request_boundary", None)
        referenced.component_data[S].session_ref = 1
        f.swa._refresh_session_partition(referenced)
        f.reclaim(256)
        self.assertIsNone(unreferenced.component_data[S].host_value)
        self.assertIsNotNone(referenced.component_data[S].host_value)

    def test_host_io_lock_survives_split_and_release(self):
        f = CacheFixture(capacity=4096)
        n = f.node(f.tree.root_node, [3] * 512, torch.arange(512))
        f.mark(n)
        f.backup(n)
        f.drain(f.tree.demote(n.id))
        receipt = f.tree.inc_host_lock_ref(n.id).to_dec_params()
        parent, action = f.tree._split_node(n.key, n, 256)
        self.assertIsNone(action)
        for part in (parent, n):
            self.assertEqual(part.component_data[S].host_lock_ref, 1)
            self.assertTrue(
                part.component_data[S].metadata.get("host_request_boundary")
            )
            self.assertFalse(f.tree.host_lru_lists[S].in_list(part))
        self.assertEqual(f.reclaim(512).tracker.get(S, 0), 0)
        f.tree.dec_host_lock_ref(n.id, receipt)
        for part in (parent, n):
            self.assertEqual(part.component_data[S].host_lock_ref, 0)
        self.assertEqual(f.reclaim(512).tracker[S], 512)

    def test_incremental_backup_repairs_missing_window_page_without_full_copy(self):
        f = CacheFixture(capacity=4096, window=512)
        nodes = f.prompt(0, length=768, swa_slots=768)
        middle, tail = nodes[-2:]
        # Bring device values back and remove only the middle SWA host copy.
        for n in (middle, tail):
            f.tree._unevict_node_on_insert(n, n.component_data[F].host_value.clone())
            f.tree.set_component_device_value(
                n.id, S, n.component_data[F].value.clone()
            )
        f.pool.free(middle.component_data[S].host_value)
        middle.component_data[S].host_value = None
        self.assertTrue(f.swa.needs_incremental_backup(tail))
        full, transfers = f.backup(tail)
        self.assertEqual(full.numel(), 0)
        self.assertEqual(transfers[S][0].nodes_to_load, [middle.id])
        self.assertIsNotNone(middle.component_data[S].host_value)
        self.assertFalse(f.swa.needs_incremental_backup(tail))
        torch.testing.assert_close(
            f.pool.payload[middle.component_data[S].host_value],
            middle.component_data[S].value,
        )

    def test_backup_does_not_copy_beyond_its_device_lock_window(self):
        f = CacheFixture(capacity=4096)
        a = f.node(f.tree.root_node, [4] * 256, torch.arange(256))
        b = f.node(a, [4] * 256, torch.arange(256, 512))
        f.mark(b)
        self.assertTrue(a.component_data[S].metadata["host_request_boundary"])
        self.assertEqual(f.swa._collect_unbacked_swa_nodes(b), [b])
        receipt = f.tree.inc_lock_ref(b.id).to_dec_params()
        self.assertEqual(a.component_data[S].lock_ref, 0)
        self.assertEqual(b.component_data[S].lock_ref, 1)
        f.tree.dec_lock_ref(b.id, receipt)

    def test_real_insert_marks_finish_not_chunk_and_repairs_backed_full(self):
        f = CacheFixture(capacity=4096)
        tokens = [17] * 2048
        result = f.insert(tokens, chunked=True, swa_evicted_seqlen=1536)
        tail = f.tree.node_by_id(result.last_device_node)
        self.assertFalse(tail.component_data[S].metadata.get("host_request_boundary"))
        f.insert(tokens, swa_evicted_seqlen=1536)
        self.assertTrue(tail.component_data[S].metadata["host_request_boundary"])
        self.assertTrue(tail.parent.component_data[S].metadata["host_request_boundary"])
        self.assertIsNotNone(tail.component_data[S].host_value)
        full_before = tail.component_data[F].host_value.clone()
        f.pool.free(tail.component_data[S].host_value)
        tail.component_data[S].host_value = None
        f.insert(tokens, swa_evicted_seqlen=1536)
        self.assertIsNotNone(tail.component_data[S].host_value)
        torch.testing.assert_close(tail.component_data[F].host_value, full_before)

    def test_real_insert_diverging_branch_does_not_claim_missing_swa(self):
        f = CacheFixture(capacity=8192)
        prefix = [20] * 1536
        f.insert(prefix + [21] * 512, swa_evicted_seqlen=1536)
        f.insert(prefix + [22] * 512, swa_evicted_seqlen=1536)
        for suffix in (21, 22):
            hit = f.tree.match_prefix(
                MatchPrefixParams(key=RadixKey(prefix + [suffix] * 256, None))
            )
            self.assertEqual(len(hit.device_indices), 1792)
        missing = f.tree.match_prefix(
            MatchPrefixParams(key=RadixKey(prefix + [23] * 256, None))
        )
        self.assertEqual(missing.full_kv_hit_length, 1536)
        self.assertEqual(len(missing.device_indices) + missing.host_hit_length, 0)

    def test_backup_stops_at_holes_and_pending_transfers(self):
        f = CacheFixture(capacity=4096, window=512)
        a = f.node(f.tree.root_node, [4] * 256, torch.arange(256))
        b = f.node(a, [4] * 256, torch.arange(256, 512))
        a.write_through_pending_id = a.id
        self.assertEqual(f.swa._collect_unbacked_swa_nodes(b), [b])
        b.write_through_pending_id = b.id
        self.assertEqual(f.swa._collect_unbacked_swa_nodes(b), [])
        a.write_through_pending_id = b.write_through_pending_id = None
        b.component_data[S].value = None
        self.assertEqual(f.swa._collect_unbacked_swa_nodes(b), [])

    def test_buffer_mode_stays_node_local_and_does_not_mark_boundaries(self):
        f = CacheFixture(capacity=4096, window=512, mode="buffer_only")
        a = f.node(f.tree.root_node, [5] * 256, torch.arange(256))
        b = f.node(a, [5] * 256, torch.arange(256, 512))
        f.mark(b)
        self.assertFalse(b.component_data[S].metadata.get("host_request_boundary"))
        self.assertEqual(f.swa._collect_unbacked_swa_nodes(b), [b])

    def test_multi_page_window_retains_window_plus_one_page(self):
        f = CacheFixture(capacity=4096, window=512)
        nodes = f.prompt(0, length=2048, swa_slots=2048)
        self.assertEqual(
            sum(
                bool(n.component_data[S].metadata.get("host_request_boundary"))
                for n in nodes
            ),
            3,
        )

    def test_boundary_feature_rejects_unsupported_tree_backend(self):
        f = CacheFixture(retain=False)
        params = CacheInitParams(
            False, None, CPUAllocator(), 256, sliding_window_size=128
        )
        with patch.dict(
            os.environ, {FLAG: "1", "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "rust"}
        ):
            with self.assertRaisesRegex(ValueError, "Python TreeCore"):
                SWAComponent(f.cache, params)


class RetentionMetricsTests(unittest.TestCase):
    def test_reporter_and_prometheus_export_track_swa_separately(self):
        from functools import partial

        import prometheus_client as prom

        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )
        from sglang.srt.observability.metrics_collector import (
            SchedulerMetricsCollector,
            SchedulerStats,
        )

        f = CacheFixture(capacity=4096)
        f.prompt(0, length=2048, swa_slots=2048)
        f.reclaim(256)
        stats = SchedulerStats()
        reporter = NS(
            scheduler=NS(
                enable_hierarchical_cache=True,
                tree_cache=NS(
                    full_kv_pool_host=NS(
                        logical_size=16384, available_size=lambda: 8192
                    ),
                    components={S: f.swa},
                    tree_core=f.tree,
                ),
            ),
            stats=stats,
        )
        SchedulerMetricsReporter._log_hicache_stats(reporter)
        expected = {
            "hicache_host_used_tokens": 8192,
            "hicache_host_total_tokens": 16384,
            "hicache_swa_host_used_tokens": 1792,
            "hicache_swa_host_total_tokens": 4096,
            "hicache_swa_host_evicted_tokens": 256,
            "hicache_swa_host_boundary_evicted_tokens": 0,
            "hicache_swa_host_retention_enabled": 1,
        }
        registry = prom.CollectorRegistry()
        labels = {"tp_rank": "0", "moe_ep_rank": "0"}
        # Inject an isolated registry into every real Prometheus metric type.
        with (
            patch(
                "sglang.srt.observability.metrics_collector.get_schedule",
                return_value=NS(
                    prefill_delayer_max_delay_passes=100,
                    prefill_delayer_forward_passes_buckets=None,
                    prefill_delayer_wait_seconds_buckets=None,
                ),
            ),
            patch.object(
                SchedulerMetricsCollector,
                "_gauge_cls",
                partial(prom.Gauge, registry=registry),
            ),
            patch.object(
                SchedulerMetricsCollector,
                "_counter_cls",
                partial(prom.Counter, registry=registry),
            ),
            patch.object(
                SchedulerMetricsCollector,
                "_histogram_cls",
                partial(prom.Histogram, registry=registry),
            ),
            patch.object(
                SchedulerMetricsCollector,
                "_summary_cls",
                partial(prom.Summary, registry=registry),
            ),
        ):
            collector = SchedulerMetricsCollector(
                labels, None, enable_hierarchical_cache=True
            )
        collector.log_stats(stats)
        for name, value in expected.items():
            self.assertEqual(getattr(stats, name), value)
            self.assertEqual(
                registry.get_sample_value("sglang:" + name, labels), value
            )
        self.assertEqual(registry.get_sample_value("sglang:hicache_swa_diagnostics_enabled", labels), 1)
        self.assertEqual(registry.get_sample_value("sglang:hicache_full_host_tree_tokens", labels), 2048)
        self.assertEqual(registry.get_sample_value("sglang:hicache_full_host_swa_uncovered_tokens", labels), 0)
        cause_labels = {**labels, "trigger": "swa_pressure", "disposition": "swa_only"}
        self.assertEqual(registry.get_sample_value("sglang:hicache_swa_host_evicted_tokens_by_cause", cause_labels), 256)
        self.assertEqual(registry.get_sample_value("sglang:hicache_swa_host_boundary_evicted_tokens_by_cause", cause_labels), 0)

    def test_reporter_without_swa_has_zero_auxiliary_metrics(self):
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )
        from sglang.srt.observability.metrics_collector import SchedulerStats

        reporter = NS(
            scheduler=NS(
                enable_hierarchical_cache=True,
                tree_cache=NS(
                    token_to_kv_pool_host=NS(
                        logical_size=4096, available_size=lambda: 1024
                    )
                ),
            ),
            stats=SchedulerStats(),
        )
        SchedulerMetricsReporter._log_hicache_stats(reporter)
        self.assertEqual(reporter.stats.hicache_host_used_tokens, 3072)
        self.assertEqual(reporter.stats.hicache_swa_host_retention_enabled, 0)
        self.assertEqual(reporter.stats.hicache_swa_host_total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
