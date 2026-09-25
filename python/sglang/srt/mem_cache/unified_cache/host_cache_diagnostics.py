"""Read-only FULL/SWA cache diagnostics for Python TreeCore.

Eviction cause and disposition are orthogonal: SWA pressure can atomically
remove FULL too. Residency is the union of prefixes ending at a usable SWA
window, not the sum of nodes without a local SWA copy. No tensor reads, CUDA
operations, tree splits, LRU touches, or eviction-policy changes occur here.
"""

import time
from contextlib import contextmanager
from functools import wraps

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

TRIGGERS = ("swa_pressure", "full_pressure", "other")
DISPOSITIONS = ("swa_only", "with_full")
GAUGES = {
    "hicache_swa_diagnostics_enabled": "FULL/SWA diagnostics supported by this Python TreeCore (0 or 1).",
    "hicache_swa_match_lookups": "Cumulative nonempty prefix lookups; not unique requests.",
    "hicache_swa_limited_match_lookups": "Cumulative prefix lookups shortened solely by missing SWA availability.",
    "hicache_swa_limited_match_full_tokens": "Cumulative available FULL prefix tokens denied by SWA per lookup, including L1 and L2; repeated lookups count again.",
    "hicache_swa_limited_match_host_tokens": "Cumulative host-resident FULL prefix tokens denied by SWA per lookup; can include L1 duplicates.",
    "hicache_swa_census_complete": "Whether the last bounded residency census completed; zero means unavailable/stale, not empty.",
    "hicache_swa_census_timestamp_seconds": "Wall-clock timestamp of the last complete residency census; zero until available.",
    "hicache_swa_census_duration_seconds": "Time spent on the latest bounded census attempt.",
    "hicache_swa_census_nodes": "Tree nodes visited by the latest census attempt.",
    "hicache_full_host_tree_tokens": "Unique FULL host token slots attached to tree nodes in the last complete census; excludes allocator-only/in-flight slots.",
    "hicache_full_host_swa_covered_tokens": "Unique host FULL tokens covered by at least one currently resumable FULL+SWA prefix (SWA may be in L1 or L2).",
    "hicache_full_host_swa_uncovered_tokens": "Unique host FULL tokens on continuous FULL paths not covered by any resumable descendant prefix because SWA is missing.",
    "hicache_full_host_chain_gap_tokens": "Host FULL tokens unreachable because their FULL ancestor path has a hole; not attributed to SWA.",
    "hicache_swa_host_marked_slots": "Host SWA slots on marked tree nodes in the last complete census; not request or checkpoint counts.",
    "hicache_swa_host_unmarked_slots": "Host SWA slots on unmarked tree nodes in the last complete census.",
    "hicache_swa_marked_nodes": "Marked tree nodes in the last complete census; one request may mark several nodes.",
    "hicache_swa_marked_nodes_missing_window": "Marked nodes with a continuous FULL path but no valid SWA resumption window in either L1 or L2.",
}


def host_eviction_operation(method):
    """Group atomic removals, preserving an enclosing pressure trigger."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.host_cache_diagnostics.eviction_scope():
            return method(self, *args, **kwargs)

    return wrapped


class HostCacheDiagnostics:
    CENSUS_INTERVAL_SECONDS = 60.0
    CENSUS_MAX_SECONDS = 0.250
    CENSUS_MAX_NODES = 200_000

    def __init__(self):
        self.evictions = {(t, d): [0, 0] for t in TRIGGERS for d in DISPOSITIONS}
        self._operation = None
        self.lookups = self.limited_lookups = 0
        self.limited_full_tokens = self.limited_host_tokens = 0
        self.invalidate_census()

    def invalidate_census(self):
        # Tree reset does not reset eviction counters on the existing component.
        self._last_attempt = float("-inf")
        self.census = {
            name: 0
            for name in GAUGES
            if "census" in name
            or "host_tree" in name
            or "host_swa_" in name
            or "chain_gap" in name
            or "marked_" in name
        }

    @staticmethod
    def supported(tree):
        return (
            tree.enable_hicache
            and tree.has_swa_host_pool
            and set(tree.component_types) == {ComponentType.FULL, ComponentType.SWA}
        )

    @contextmanager
    def eviction_scope(self, trigger="other"):
        if self._operation is not None:
            yield
            return
        operation = {"trigger": trigger, "full": set(), "swa": []}
        self._operation = operation
        try:
            yield
        finally:
            # Classify after actual frees: no dependency on component order,
            # and an attempted/locked FULL eviction is not a FULL removal.
            for node_id, slots, boundary in operation["swa"]:
                disposition = (
                    "with_full" if node_id in operation["full"] else "swa_only"
                )
                counters = self.evictions[(trigger, disposition)]
                counters[0] += slots
                counters[1] += slots if boundary else 0
            self._operation = None

    def record_freed(self, node_id, component, slots, boundary=False):
        if not slots or component not in (ComponentType.FULL, ComponentType.SWA):
            return
        if self._operation is None:
            with self.eviction_scope():
                self.record_freed(node_id, component, slots, boundary)
        elif component == ComponentType.FULL:
            self._operation["full"].add(node_id)
        else:
            self._operation["swa"].append((node_id, slots, boundary))

    def observe_match(self, full_length, best_length, host_length, best_host_length):
        self.lookups += 1
        lost = full_length - best_length
        if lost > 0:
            self.limited_lookups += 1
            self.limited_full_tokens += lost
            self.limited_host_tokens += host_length - best_host_length

    def _census(self, tree, *, now):
        self._last_attempt = now
        start = time.monotonic()
        root = tree.root_node
        window = tree.components_by_type[ComponentType.SWA].sliding_window_size
        # Starting at infinity mirrors SWA.create_match_validator's root state.
        stack = [(child, True, float("inf")) for child in root.children.values()]
        visited = []
        covered = set()
        marked_slots = unmarked_slots = marked_nodes = missing_marked = 0
        complete = True
        while stack:
            if len(visited) >= self.CENSUS_MAX_NODES or (
                len(visited) % 128 == 0
                and time.monotonic() - start >= self.CENSUS_MAX_SECONDS
            ):
                complete = False
                break
            node, full_path, swa_run = stack.pop()
            full = node.component_data[ComponentType.FULL]
            swa = node.component_data[ComponentType.SWA]
            full_path = full_path and (
                full.value is not None or full.host_value is not None
            )
            swa_run = (
                swa_run + len(node.key)
                if swa.value is not None or swa.host_value is not None
                else 0
            )
            usable = full_path and swa_run >= window
            if usable:
                covered.add(node.id)
            host_len = len(full.host_value) if full.host_value is not None else 0
            visited.append((node, full_path, host_len))
            marked = bool(swa.metadata.get("host_request_boundary"))
            slots = len(swa.host_value) if swa.host_value is not None else 0
            if marked:
                marked_slots += slots
                marked_nodes += 1
                missing_marked += int(full_path and not usable)
            else:
                unmarked_slots += slots
            stack.extend(
                (child, full_path, swa_run) for child in node.children.values()
            )
        total = reusable = uncovered = chain_gap = 0
        if complete:
            # A valid descendant covers its entire FULL history, including
            # interior nodes with no SWA of their own. Shared ancestors count once.
            for index, (node, full_path, host_len) in enumerate(reversed(visited)):
                if (
                    index % 128 == 0
                    and time.monotonic() - start >= self.CENSUS_MAX_SECONDS
                ):
                    complete = False
                    break
                total += host_len
                if node.id in covered:
                    reusable += host_len
                    covered.add(node.parent.id)
                elif not full_path:
                    chain_gap += host_len
                else:
                    uncovered += host_len
        self.census.update(
            {
                "hicache_swa_census_complete": int(complete),
                "hicache_swa_census_duration_seconds": time.monotonic() - start,
                "hicache_swa_census_nodes": len(visited),
            }
        )
        if complete:
            self.census.update(
                {
                    "hicache_swa_census_timestamp_seconds": time.time(),
                    "hicache_full_host_tree_tokens": total,
                    "hicache_full_host_swa_covered_tokens": reusable,
                    "hicache_full_host_swa_uncovered_tokens": uncovered,
                    "hicache_full_host_chain_gap_tokens": chain_gap,
                    "hicache_swa_host_marked_slots": marked_slots,
                    "hicache_swa_host_unmarked_slots": unmarked_slots,
                    "hicache_swa_marked_nodes": marked_nodes,
                    "hicache_swa_marked_nodes_missing_window": missing_marked,
                }
            )

    def snapshot(self, tree, *, census=True):
        if not self.supported(tree):
            return {}, {}, {}
        now = time.monotonic()
        if census and now - self._last_attempt >= self.CENSUS_INTERVAL_SECONDS:
            self._census(tree, now=now)
        gauges = {
            **self.census,
            "hicache_swa_diagnostics_enabled": 1,
            "hicache_swa_match_lookups": self.lookups,
            "hicache_swa_limited_match_lookups": self.limited_lookups,
            "hicache_swa_limited_match_full_tokens": self.limited_full_tokens,
            "hicache_swa_limited_match_host_tokens": self.limited_host_tokens,
        }
        totals = {
            t: {d: self.evictions[t, d][0] for d in DISPOSITIONS} for t in TRIGGERS
        }
        boundaries = {
            t: {d: self.evictions[t, d][1] for d in DISPOSITIONS} for t in TRIGGERS
        }
        return gauges, totals, boundaries
