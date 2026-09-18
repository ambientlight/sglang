"""Byte-bounded RAM HiCache sizing for the pinned single-node V4.1 layout.

No CUDA operations: tensor shapes describe exactly the same page rows used by
hybrid_pool_assembler. The budget is host-wide, NOT a per-TP-rank setting.
Other model/layout combinations fail closed instead of silently undercounting.
"""

import math


METADATA_RESERVE_BYTES = 1024**3  # host-wide allowance; not a process RSS limit


def page_costs(kvcache, page_size, draft_pools):
    sources = getattr(kvcache, "sources_by_ratio", {})
    if (
        getattr(kvcache, "_unified_kv", False)
        or not sources
        or any(ratio not in (1, 2) for ratio, layers in sources.items() if layers)
        or getattr(kvcache, "c4_kv_pool", None) is not None
        or getattr(kvcache, "c128_kv_pool", None) is not None
    ):
        raise ValueError("Bounded HiCache requires non-unified V4.1 C1/C2 pools")

    def row_bytes(buffers):
        if not buffers:
            raise ValueError("Missing HiCache device buffers")
        sizes = [math.prod(b.shape[1:]) * b.element_size() for b in buffers]
        if min(sizes) <= 0 or len(set(sizes)) != 1:
            raise ValueError("Inconsistent HiCache page-row sizes")
        return sum(sizes)

    full_bytes = 0
    full_pools = 1  # logical anchor also has an int64 free-slot array
    for ratio, layers in sources.items():
        if not layers:
            continue
        kv = kvcache.kv_pools[ratio]
        index = kvcache.index_pools[ratio]
        if page_size % (ratio * index.page_size):
            raise ValueError("HiCache FULL pages must align with indexer pages")
        if len(kv.kv_buffer) != len(layers) or any(len(b.shape) != 2 for b in kv.kv_buffer):
            raise ValueError("Unexpected shared-source KV layout")
        full_bytes += row_bytes(kv.kv_buffer)
        full_pools += 1
        fused = index.index_k_with_scale_buffer
        regions = [fused] if fused is not None else [index.index_k_payload_buffer, index.index_k_scale_buffer]
        for buffers in regions:
            if len(buffers) != len(layers):
                raise ValueError("Unexpected shared-source indexer layer count")
            full_bytes += row_bytes(buffers) * (page_size // ratio // index.page_size)
            full_pools += 1

    swa = kvcache.swa_kv_pool
    swa_bytes = row_bytes(swa.kv_buffer)
    if swa_bytes != len(swa.kv_buffer) * swa.bytes_per_page_padded:
        raise ValueError("Unexpected SWA page layout")
    # DSpark packs draft SWA alongside target SWA. No separately sized sidecar.
    if not draft_pools:
        raise ValueError("Bounded V4.1 HiCache requires packed DSpark draft pools")
    for pool in draft_pools:
        draft = pool.swa_kv_pool
        if draft.bytes_per_page_padded != swa.bytes_per_page_padded:
            raise ValueError("Target and draft SWA page sizes must match")
        swa_bytes += row_bytes(draft.kv_buffer)
    # Allow three copies of each slot list for live indices and merging frees.
    return full_bytes, swa_bytes, full_pools * page_size * 8 * 3, kvcache.swa_page_size * 8 * 3


def plan_pages(
    *,
    budget_bytes,
    ranks,
    full_pages,
    swa_pages,
    full_page_bytes,
    swa_page_bytes,
    full_metadata_page_bytes=0,
    swa_metadata_page_bytes=0,
):
    values = (budget_bytes, ranks, full_pages, swa_pages, full_page_bytes, swa_page_bytes)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("HiCache sizing requires positive integer inputs")
    if ranks != 4:
        raise ValueError("Bounded V4.1 HiCache is qualified only for four local ranks")
    if any(type(v) is not int or v < 0 for v in (full_metadata_page_bytes, swa_metadata_page_bytes)):
        raise ValueError("Invalid metadata allowance")
    per_rank = (budget_bytes - METADATA_RESERVE_BYTES) // ranks

    def usage(pages):
        swa = pages * swa_pages // full_pages
        return (
            pages * (full_page_bytes + full_metadata_page_bytes) + swa * (swa_page_bytes + swa_metadata_page_bytes)
        ), swa

    if usage(full_pages)[0] > per_rank:
        raise ValueError("HiCache budget cannot hold one device-cache equivalent")
    lo, hi = full_pages, per_rank // full_page_bytes + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if usage(mid)[0] <= per_rank:
            lo = mid
        else:
            hi = mid
    planned_bytes, swa = usage(lo)
    tensor_bytes = lo * full_page_bytes + swa * swa_page_bytes
    return {
        "host_budget_bytes": budget_bytes,
        "local_ranks": ranks,
        "metadata_reserve_bytes": METADATA_RESERVE_BYTES,
        "full_host_pages": lo,
        "swa_host_pages": swa,
        "device_full_pages": full_pages,
        "device_swa_pages": swa_pages,
        "full_page_bytes": full_page_bytes,
        "swa_page_bytes": swa_page_bytes,
        "tensor_bytes_per_rank": tensor_bytes,
        "tensor_bytes_host_total": tensor_bytes * ranks,
        "planned_bytes_host_total": planned_bytes * ranks + METADATA_RESERVE_BYTES,
        "slot_metadata_bytes_per_rank": planned_bytes - tensor_bytes,
    }
