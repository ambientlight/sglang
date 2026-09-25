# FULL/SWA eviction attribution and reusable-residency telemetry

This is an **observability-only** extension for the deployed Python TreeCore
with exactly FULL + SWA components and an SWA host pool. It does not change
matching, retention, page allocation, transfer locks, eviction ordering or
model/API behavior. Unsupported layouts explicitly report diagnostics disabled.

## 1. Trigger and disposition are independent

Existing cumulative gauges remain unchanged:

- `sglang:hicache_swa_host_evicted_tokens`
- `sglang:hicache_swa_host_boundary_evicted_tokens` (subset of the first)

New gauges with bounded labels are:

- `sglang:hicache_swa_host_evicted_tokens_by_cause`
- `sglang:hicache_swa_host_boundary_evicted_tokens_by_cause`

Each has `trigger=swa_pressure|full_pressure|other` and
`disposition=swa_only|with_full`, plus the usual rank/model labels.

| Label | Exact interpretation |
|---|---|
| `swa_pressure` | The host reclaimer was asked to free SWA slots. |
| `full_pressure` | The host reclaimer was asked to free FULL slots. |
| `other` | Removal outside those pressure-driven operations, including cleanup. |
| `swa_only` | This node lost SWA host slots without also losing FULL **host** slots in the operation. This alone does not prove FULL remains resident or useful. |
| `with_full` | Actual FULL host slots from this same node were also freed in the operation. SWA pressure can cause this through atomic host-leaf eviction. |

Classification uses actual returned freed counts, not intended victims or call
order. Nested atomic removals preserve the initiating trigger. A locked node or
a device-only eviction adds nothing. FULL host-duplicate reclamation with no
SWA removal adds nothing. No token text, request IDs or node IDs are exported.
For each rank the six total cells sum to the legacy total; the six boundary
cells sum to the legacy boundary total. Counters survive a TreeCore reset just
as the legacy component counters do, and restart with a new cache instance.

## 2. FULL residency that lacks a reusable SWA boundary

A missing SWA copy on an interior node does **not** make its FULL history
unusable if a valid later checkpoint covers it. Nor is missing host SWA a loss
when the needed SWA is on GPU. The census respects both cases.

Every 60 seconds at most, TP rank 0 performs a read-only traversal using the
same continuous-SWA-window rule as the matcher. A reverse pass forms the union
of FULL prefixes covered by at least one usable endpoint. Shared ancestors
count once, even when several branches can reuse them.

| Gauge (`sglang:` prefix omitted) | Unit / meaning |
|---|---|
| `hicache_full_host_tree_tokens` | Unique FULL host token slots attached to tree nodes. |
| `hicache_full_host_swa_covered_tokens` | Those slots lying in at least one currently resumable FULL+SWA prefix, with SWA present in L1 and/or L2. |
| `hicache_full_host_swa_uncovered_tokens` | Those slots on continuous FULL paths, but outside every resumable descendant prefix because SWA is missing. |
| `hicache_full_host_chain_gap_tokens` | Host FULL beyond a missing FULL ancestor; separately classified, not blamed on SWA. |
| `hicache_swa_host_marked_slots` / `hicache_swa_host_unmarked_slots` | Attached SWA host slots, separated by the soft boundary marker. |
| `hicache_swa_marked_nodes` / `hicache_swa_marked_nodes_missing_window` | Marked tree nodes, and marked nodes with continuous FULL but no valid L1/L2 SWA window. Not requests or distinct checkpoint counts. |

`tree_tokens = covered_tokens + uncovered_tokens + chain_gap_tokens`.
This is an instantaneous structural measurement, not a prediction of future
hits. A token counted as covered by a later endpoint may still fail an exact
shorter replay. The lookup counters below measure that separate effect.

Allocator occupancy can differ from `tree_tokens`: in-flight transfers,
pending releases and other allocator-only slots are not attached residency.
Do not subtract this census from a differently timed allocator gauge and call
the difference SWA loss. Report the uncovered fraction against `tree_tokens`
from the **same** completed census.

The scan has a 250 ms work budget, checked every 128 nodes, and a 200,000-node
limit (normal Python cleanup adds some overhead). It never reads tensor values,
synchronizes CUDA, touches LRUs, splits nodes or runs inference. It executes
only on rank 0, not independently on all TP ranks. Bounded scans can fail to
complete on a large/busy tree; they **never publish partial residency totals**.

Check `hicache_swa_diagnostics_enabled`, `hicache_swa_census_complete`, and
`hicache_swa_census_timestamp_seconds`. A failed scan sets `complete=0` and
preserves the last complete totals and their timestamp. Duration and visited
node count describe the latest attempt. Before the first successful scan,
zero-valued totals are unavailable, not evidence of an empty/unimpaired cache.
Tree reset invalidates the census immediately. The passive monitor regards
incomplete snapshots or snapshots older than 150 seconds as unavailable.
This bounded synchronous rank-0 work can briefly delay a scheduler iteration;
250 ms per minute is at most about 0.42% scan duty cycle, excluding Python
cleanup. CPU metadata benchmarks and the explicit completion signal are used
instead of assuming that every possible large tree fits the work budget.

## 3. Actual prefix-lookup opportunity lost to missing SWA

The existing matcher records cheap integer observations without an extra
traversal or a changed match result:

- `hicache_swa_match_lookups`: nonempty aligned lookups.
- `hicache_swa_limited_match_lookups`: lookups whose available FULL prefix
  exceeds the best jointly usable FULL+SWA prefix.
- `hicache_swa_limited_match_full_tokens`: sum of that denied FULL suffix.
- `hicache_swa_limited_match_host_tokens`: host-resident part of that suffix,
  including host copies of device-resident FULL where both exist.

These are **lookup counters, not unique requests or completed-input counters**.
Scheduler retries and repeated lookups count again. Do not divide them by
served input tokens to claim a request miss rate. A fallback to an earlier
valid boundary counts only the denied tail, not the already reusable prefix.
Cold mismatches and unavailable FULL ancestors are not SWA-attributed loss.

## 4. Deployment and tests

The benchmark deployment stages `cache-attribution-v1` on the immutable
`diagnostics-v1` image, with a frozen source receipt and four runtime files.
The source image retains existing API, SWA retention, CUDA/NCCL diagnostics,
Engram and kernel code. Tests use real Python TreeCore and bounded CPU payloads
in a container without GPU access; no production cache is flushed or replayed.

The accompanying **67% SWA / 33% FULL payload** setting remains within the
existing 256,000,000,000-byte host-total plan. At the measured page costs it
plans 12,570,368 FULL tokens and 1,656,064 SWA slots per rank. This is a capacity
projection, not live allocation qualification. A Compose recreation is needed;
plain `docker restart` retains an existing container's old image/environment.

Read-only telemetry must be verified against real captures after the next
authorized recreation. Staging is not proof of live behavior or improved hits.
