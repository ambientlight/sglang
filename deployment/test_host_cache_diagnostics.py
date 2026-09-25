"""CPU-only attribution and reusable-residency tests against the real TreeCore."""

import unittest
from collections import defaultdict
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.host_cache_diagnostics import (
    HostCacheDiagnostics,
)
from test_swa_host_retention import CacheFixture, F, S


class AttributionTests(unittest.TestCase):
    def counts(self, f, trigger, disposition):
        return f.tree.host_cache_diagnostics.evictions[trigger, disposition]

    def test_swa_pressure_internal_node_keeps_full(self):
        f = CacheFixture(capacity=4096)
        nodes = f.prompt(0, length=1024, swa_slots=1024)
        for node in nodes:
            node.component_data[S].metadata["host_request_boundary"] = True
        # Choose a marked internal node as the LRU victim, not the atomic leaf.
        for node in nodes[1:]:
            f.tree.host_lru_lists[S].reset_node_mru(node)
        f.reclaim(256)
        self.assertIsNotNone(nodes[0].component_data[F].host_value)
        self.assertEqual(self.counts(f, "swa_pressure", "swa_only"), [256, 256])
        self.assertEqual(self.counts(f, "swa_pressure", "with_full"), [0, 0])

    def test_swa_pressure_atomic_leaf_also_removes_full(self):
        f = CacheFixture(capacity=4096)
        n = f.prompt(0, length=256, swa_slots=256)[0]
        f.reclaim(256)
        self.assertIsNone(n.component_data[F].host_value)
        self.assertEqual(self.counts(f, "swa_pressure", "with_full"), [256, 256])

    def test_full_pressure_atomic_leaf_is_not_swa_pressure(self):
        f = CacheFixture(capacity=4096)
        f.prompt(0, length=256, swa_slots=256)
        result = f.tree.drive_host_eviction(F, 256)
        f.drain(result)
        self.assertEqual(self.counts(f, "full_pressure", "with_full"), [256, 256])
        self.assertEqual(self.counts(f, "swa_pressure", "with_full"), [0, 0])

    def test_full_duplicate_reclaim_does_not_count_swa_eviction(self):
        f = CacheFixture(capacity=4096)
        f.tree.is_write_back = True
        n = f.node(f.tree.root_node, [7] * 512, torch.arange(512))
        f.backup(n)
        # The CPU fixture copies synchronously; model the completed backup ack.
        f.tree._update_duplicate_tracking(n)
        before = f.swa.host_evicted_tokens
        result = f.tree.drive_host_eviction(F, 256)
        self.assertGreater(result.tracker[F], 0)
        f.drain(result)
        self.assertEqual(f.swa.host_evicted_tokens, before)
        self.assertEqual(
            sum(v[0] for v in f.tree.host_cache_diagnostics.evictions.values()), before
        )

    def test_host_lock_prevents_counting_an_attempt_as_eviction(self):
        f = CacheFixture(capacity=4096)
        n = f.prompt(0, length=256, swa_slots=256)[0]
        receipt = f.tree.inc_host_lock_ref(n.id).to_dec_params()
        f.reclaim(256)
        self.assertEqual(
            sum(v[0] for v in f.tree.host_cache_diagnostics.evictions.values()), 0
        )
        f.tree.dec_host_lock_ref(n.id, receipt)
        f.reclaim(256)
        self.assertEqual(self.counts(f, "swa_pressure", "with_full"), [256, 256])

    def test_accounting_is_order_independent_nested_and_exception_safe(self):
        d = HostCacheDiagnostics()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with d.eviction_scope("swa_pressure"):
                d.record_freed(1, S, 256, True)
                with d.eviction_scope():
                    d.record_freed(1, F, 512)
                    d.record_freed(2, S, 128)
                raise RuntimeError("fixture")
        self.assertEqual(d.evictions["swa_pressure", "with_full"], [256, 256])
        self.assertEqual(d.evictions["swa_pressure", "swa_only"], [128, 0])
        d.record_freed(3, S, 256, True)
        self.assertEqual(d.evictions["other", "swa_only"], [256, 256])
        self.assertIsNone(d._operation)

    def test_non_pressure_atomic_removal_is_other(self):
        f = CacheFixture(capacity=4096)
        n = f.prompt(0, length=256, swa_slots=256)[0]
        f.tree._evict_host_leaf(
            n, defaultdict(int), defaultdict(list), defaultdict(list)
        )
        self.assertEqual(self.counts(f, "other", "with_full"), [256, 256])

    def test_breakdown_sums_to_legacy_counters_and_reset_preserves_both(self):
        f = CacheFixture(capacity=1024)
        for i in range(12):
            f.prompt(i, length=512, swa_slots=512)
        d = f.tree.host_cache_diagnostics
        self.assertEqual(
            sum(v[0] for v in d.evictions.values()), f.swa.host_evicted_tokens
        )
        self.assertEqual(
            sum(v[1] for v in d.evictions.values()), f.swa.host_boundary_evicted_tokens
        )
        before = {k: v[:] for k, v in d.evictions.items()}
        f.tree.reset()
        self.assertEqual(d.evictions, before)
        self.assertEqual(d.snapshot(f.tree)[0]["hicache_full_host_tree_tokens"], 0)


class CensusTests(unittest.TestCase):
    def census(self, f):
        d = f.tree.host_cache_diagnostics
        # Deterministic unit correctness independent of a busy test runner.
        d.CENSUS_MAX_SECONDS = 10
        d._last_attempt = float("-inf")
        return d.snapshot(f.tree)[0]

    def node(self, f, parent, token, size=256, swa=True):
        n = f.node(parent, [token] * size, torch.arange(size), swa=swa)
        f.backup(n)
        return n

    def test_missing_interior_swa_not_stranded_when_descendant_is_usable(self):
        f = CacheFixture(capacity=4096)
        f.prompt(0, length=2048, swa_slots=512)
        g = self.census(f)
        self.assertEqual(g["hicache_full_host_tree_tokens"], 2048)
        self.assertEqual(g["hicache_full_host_swa_covered_tokens"], 2048)
        self.assertEqual(g["hicache_full_host_swa_uncovered_tokens"], 0)

    def test_shared_ancestor_is_counted_once_and_bad_branch_only_is_uncovered(self):
        f = CacheFixture(capacity=4096)
        a = self.node(f, f.tree.root_node, 1, 1024, False)
        self.node(f, a, 2, 256, True)
        self.node(f, a, 3, 512, False)
        g = self.census(f)
        self.assertEqual(g["hicache_full_host_tree_tokens"], 1792)
        self.assertEqual(g["hicache_full_host_swa_covered_tokens"], 1280)
        self.assertEqual(g["hicache_full_host_swa_uncovered_tokens"], 512)

    def test_earlier_usable_boundary_does_not_cover_missing_tail(self):
        f = CacheFixture(capacity=4096)
        a = self.node(f, f.tree.root_node, 1, 512, True)
        b = self.node(f, a, 2, 512, False)
        f.mark(b)
        g = self.census(f)
        self.assertEqual(g["hicache_full_host_swa_covered_tokens"], 512)
        self.assertEqual(g["hicache_full_host_swa_uncovered_tokens"], 512)
        self.assertEqual(g["hicache_swa_marked_nodes_missing_window"], 1)

    def test_device_swa_can_make_host_full_reusable(self):
        f = CacheFixture(capacity=4096)
        n = self.node(f, f.tree.root_node, 1, 512, True)
        f.pool.free(n.component_data[S].host_value)
        n.component_data[S].host_value = None
        g = self.census(f)
        self.assertEqual(g["hicache_full_host_swa_uncovered_tokens"], 0)
        n.component_data[S].value = None
        self.assertEqual(self.census(f)["hicache_full_host_swa_uncovered_tokens"], 512)

    def test_window_must_be_contiguous_after_a_hole(self):
        f = CacheFixture(capacity=4096, window=512)
        a = self.node(f, f.tree.root_node, 1, 512, False)
        b = self.node(f, a, 2, 256, True)
        g = self.census(f)
        self.assertEqual(g["hicache_full_host_swa_uncovered_tokens"], 768)
        self.node(f, b, 3, 256, True)
        self.assertEqual(self.census(f)["hicache_full_host_swa_covered_tokens"], 1024)

    def test_full_chain_hole_is_not_attributed_to_swa(self):
        f = CacheFixture(capacity=4096)
        a = self.node(f, f.tree.root_node, 1, 256, True)
        self.node(f, a, 2, 256, True)
        a.component_data[F].value = a.component_data[F].host_value = None
        g = self.census(f)
        self.assertEqual(g["hicache_full_host_chain_gap_tokens"], 256)
        self.assertEqual(g["hicache_full_host_swa_uncovered_tokens"], 0)

    def test_marked_unmarked_slots_and_validity_are_not_checkpoint_counts(self):
        f = CacheFixture(capacity=4096)
        f.prompt(0, length=2048, swa_slots=2048)
        g = self.census(f)
        self.assertEqual(g["hicache_swa_host_marked_slots"], 512)
        self.assertEqual(g["hicache_swa_host_unmarked_slots"], 1536)
        self.assertEqual(g["hicache_swa_marked_nodes"], 2)
        self.assertEqual(g["hicache_swa_marked_nodes_missing_window"], 0)

    def test_census_does_not_touch_lru_split_or_read_tensor_values(self):
        f = CacheFixture(capacity=4096)
        f.prompt(0, length=2048, swa_slots=512)
        before = [
            (n.id, n.last_access_time, tuple(n.children))
            for n in f.tree._node_arena.values()
        ]
        with (
            patch.object(f.tree, "_split_node", side_effect=AssertionError("split")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("tensor read")
            ),
        ):
            self.census(f)
        self.assertEqual(
            before,
            [
                (n.id, n.last_access_time, tuple(n.children))
                for n in f.tree._node_arena.values()
            ],
        )

    def test_bounds_publish_no_partial_totals_and_keep_last_timestamp(self):
        f = CacheFixture(capacity=4096)
        f.prompt(0, length=2048, swa_slots=512)
        d = f.tree.host_cache_diagnostics
        first = self.census(f)
        d.CENSUS_MAX_NODES = 1
        failed = self.census(f)
        self.assertEqual(failed["hicache_swa_census_complete"], 0)
        self.assertEqual(
            failed["hicache_full_host_tree_tokens"],
            first["hicache_full_host_tree_tokens"],
        )
        self.assertEqual(
            failed["hicache_swa_census_timestamp_seconds"],
            first["hicache_swa_census_timestamp_seconds"],
        )
        d.CENSUS_MAX_SECONDS = 0
        d._last_attempt = float("-inf")
        self.assertEqual(d.snapshot(f.tree)[0]["hicache_swa_census_complete"], 0)

    def test_census_throttled_and_other_tp_ranks_skip_it(self):
        f = CacheFixture(capacity=4096)
        d = f.tree.host_cache_diagnostics
        with patch.object(d, "_census", wraps=d._census) as scan:
            d.snapshot(f.tree, census=False)
            self.assertEqual(scan.call_count, 0)
            d.snapshot(f.tree)
            d.snapshot(f.tree)
            self.assertEqual(scan.call_count, 1)

    def test_unsupported_layouts_do_not_publish_false_zero_measurements(self):
        f = CacheFixture(capacity=4096)
        for key in ("enable_hicache", "has_swa_host_pool"):
            with patch.object(f.tree, key, False):
                self.assertEqual(
                    f.tree.host_cache_diagnostics.snapshot(f.tree), ({}, {}, {})
                )


class MatchAttributionTests(unittest.TestCase):
    def match(self, f, tokens):
        return f.tree.match_prefix(MatchPrefixParams(key=RadixKey(tokens, None)))

    def test_fallback_counts_only_the_denied_tail_and_repeated_lookups(self):
        f = CacheFixture(capacity=4096)
        a = f.node(f.tree.root_node, [1] * 512, torch.arange(512))
        b = f.node(a, [2] * 512, torch.arange(512), swa=False)
        f.backup(a)
        f.backup(b)
        d = f.tree.host_cache_diagnostics
        for _ in range(2):
            result = self.match(f, [1] * 512 + [2] * 512)
            self.assertEqual(result.full_kv_hit_length, 1024)
            self.assertEqual(len(result.device_indices) + result.host_hit_length, 512)
        self.assertEqual(
            (
                d.lookups,
                d.limited_lookups,
                d.limited_full_tokens,
                d.limited_host_tokens,
            ),
            (2, 2, 1024, 1024),
        )

    def test_later_valid_window_covers_interior_hole(self):
        f = CacheFixture(capacity=4096)
        f.prompt(0, length=2048, swa_slots=512)
        self.match(f, [1] * 2048)
        self.assertEqual(f.tree.host_cache_diagnostics.limited_full_tokens, 0)

    def test_partial_match_split_uses_only_matched_length(self):
        f = CacheFixture(capacity=4096)
        n = f.node(f.tree.root_node, [1] * 1024, torch.arange(1024), swa=False)
        f.backup(n)
        self.match(f, [1] * 512 + [9] * 512)
        d = f.tree.host_cache_diagnostics
        self.assertEqual((d.limited_full_tokens, d.limited_host_tokens), (512, 512))

    def test_device_only_full_denial_is_not_reported_as_host_residency(self):
        f = CacheFixture(capacity=4096)
        f.node(f.tree.root_node, [1] * 512, torch.arange(512), swa=False)
        self.match(f, [1] * 512)
        d = f.tree.host_cache_diagnostics
        self.assertEqual((d.limited_full_tokens, d.limited_host_tokens), (512, 0))

    def test_empty_cold_and_unsupported_lookups(self):
        f = CacheFixture(capacity=4096)
        self.match(f, [])
        self.assertEqual(f.tree.host_cache_diagnostics.lookups, 0)
        self.match(f, [1] * 512)
        self.assertEqual(f.tree.host_cache_diagnostics.lookups, 1)
        self.assertEqual(f.tree.host_cache_diagnostics.limited_lookups, 0)
        f.tree.has_swa_host_pool = False
        self.match(f, [1] * 512)
        self.assertEqual(f.tree.host_cache_diagnostics.lookups, 1)


if __name__ == "__main__":
    unittest.main()
