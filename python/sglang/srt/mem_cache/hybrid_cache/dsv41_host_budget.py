"""Byte-bounded RAM HiCache sizing for the pinned single-node V4.1 layout.

No CUDA operations: tensor shapes describe exactly the same page rows used by
hybrid_pool_assembler. The budget is host-wide, NOT a per-TP-rank setting.
Other model/layout combinations fail closed instead of silently undercounting.
"""

import math
from fractions import Fraction


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
    swa_fraction=None,
):
    """Size both pools, optionally fixing SWA's fraction of tensor payload.

    Metadata stays inside budget_bytes, but outside the payload split. Unset
    preserves device-proportional sizing. Rational arithmetic keeps whole-page
    rounding deterministic across ranks and avoids float budget overshoots.
    """
    values = (budget_bytes, ranks, full_pages, swa_pages, full_page_bytes, swa_page_bytes)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("HiCache sizing requires positive integer inputs")
    if ranks != 4:
        raise ValueError("Bounded V4.1 HiCache is qualified only for four local ranks")
    if any(type(v) is not int or v < 0 for v in (full_metadata_page_bytes, swa_metadata_page_bytes)):
        raise ValueError("Invalid metadata allowance")
    fraction = None
    if swa_fraction is not None:
        try:
            fraction = Fraction(str(swa_fraction))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("SWA payload fraction must be strictly between 0 and 1") from exc
        if not 0 < fraction < 1:
            raise ValueError("SWA payload fraction must be strictly between 0 and 1")
    per_rank = (budget_bytes - METADATA_RESERVE_BYTES) // ranks

    def usage(pages):
        if fraction is None:
            swa = pages * swa_pages // full_pages
        else:
            swa = (
                pages * full_page_bytes * fraction.numerator
                // (swa_page_bytes * (fraction.denominator - fraction.numerator))
            )
        return (
            pages * (full_page_bytes + full_metadata_page_bytes) + swa * (swa_page_bytes + swa_metadata_page_bytes)
        ), swa

    minimum_full_pages = full_pages
    if fraction is not None:
        numerator = swa_pages * swa_page_bytes * (fraction.denominator - fraction.numerator)
        denominator = full_page_bytes * fraction.numerator
        minimum_full_pages = max(full_pages, (numerator + denominator - 1) // denominator)
    if usage(minimum_full_pages)[0] > per_rank:
        raise ValueError("HiCache budget cannot hold one device-cache equivalent")
    lo, hi = minimum_full_pages, per_rank // full_page_bytes + 1
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
        "allocation_mode": "device_ratio" if fraction is None else "payload_fraction",
        "requested_swa_payload_fraction": None if fraction is None else float(fraction),
        "actual_swa_payload_fraction": swa * swa_page_bytes / tensor_bytes,
        "full_tensor_bytes_per_rank": lo * full_page_bytes,
        "swa_tensor_bytes_per_rank": swa * swa_page_bytes,
    }
