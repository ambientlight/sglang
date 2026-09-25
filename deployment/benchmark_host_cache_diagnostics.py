"""CPU metadata-only census cost check; never loads a model or touches CUDA."""

import json
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.host_cache_diagnostics import (
    HostCacheDiagnostics,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


def main():
    full, swa = ComponentType.FULL, ComponentType.SWA
    # A shared CPU tensor is only a stand-in for opaque slot-index metadata.
    # The census inspects lengths, never values or unique physical addresses.
    payload = torch.arange(256)
    results = []
    for count in (1_000, 10_000, 50_000, 80_000):
        root = UnifiedTreeNode((full, swa))
        nodes = [root]
        for i in range(count):
            node = UnifiedTreeNode((full, swa))
            node.parent = nodes[i // 10]
            node.key = range(256)
            node.component_data[full].host_value = payload
            node.component_data[swa].host_value = payload if i % 5 == 0 else None
            node.parent.children[i] = node
            nodes.append(node)
        tree = SimpleNamespace(
            root_node=root,
            enable_hicache=True,
            has_swa_host_pool=True,
            component_types=(full, swa),
            components_by_type={swa: SimpleNamespace(sliding_window_size=128)},
        )
        diagnostics = HostCacheDiagnostics()
        gauges, _, _ = diagnostics.snapshot(tree)
        row = {
            "tree_nodes": count,
            "complete": gauges["hicache_swa_census_complete"],
            "visited_nodes": gauges["hicache_swa_census_nodes"],
            "duration_ms": gauges["hicache_swa_census_duration_seconds"] * 1000,
            "work_budget_ms": diagnostics.CENSUS_MAX_SECONDS * 1000,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        # Test-scoped tree cycles must not accumulate between measurements.
        for node in nodes:
            node.children.clear()
            node.parent = None
    return results


if __name__ == "__main__":
    main()
