# V4.1 RAM HiCache and SWA retention: deployment reference

This guide covers the single-node, four-RTX-PRO-6000 deployment on
`fix/dsv41-sero-v12-api-overlay`. Configuration and qualification snapshots below
are from **September 18, 2026 UTC** (September 17 local). They are recorded
measurements, not promises that a later restart has identical capacity.

The launcher/host-budget commit `01ab30e` precedes the SWA retention change in
this branch. The branch started from an image-extracted Python baseline, not a
known upstream SGLang commit; see [overlay provenance](../DSV41_OVERLAY.md).

## 1. What is cached, and why the old replays missed

**SWA is cached in RAM.** HiCache already supported this before the retention
patch. The defect addressed here is loss of useful SWA checkpoints while their
corresponding FULL KV remains cached.

| State | GPU role | RAM role |
|---|---|---|
| FULL KV and associated indexer state | Prefix state used by inference | Larger cache of prefixes evicted from GPU |
| Sliding-window attention (SWA) state | Recent window required to resume at a prefix boundary | Saved windows/checkpoints that can be copied back to GPU |
| Packed DSPARK draft SWA | Speculative decoding state | Included in the host SWA page accounting |
| Engram weights | Requested rows used in model computation | Separate model-weight offload, not prompt KV and not part of the HiCache budget |

Attention still executes on GPU. A host hit schedules RAM-to-GPU copies; it does
not run attention directly against system RAM. This deployment has no KV
storage backend on NVMe and no cache persistence across process restarts.

A reusable prefix must have both the required FULL state and a valid SWA
resumption window. Missing SWA must cause a shorter match or a miss, even if
FULL state remains. The patch does not weaken that correctness check.

In the older 64 × 256k experiment:

- FULL RAM occupancy was only 16,381,952 / 32,699,648 logical tokens (50.10%).
- Each cold request backed up 24,064 SWA slots/rank in intermediate checkpoints.
- The host SWA pool held 254,976 slots/rank: about 10.6 such footprints.
- Seven older exact replays missed; only the newest replay reused 255,744 tokens.
- Captured input IDs and cache namespaces matched. TTL expiry was not the cause.

Historical per-node eviction traces were not recorded, so the exact victim
sequence is an inference. Source inspection and the real-tree legacy-policy
regression reproduce this FULL/SWA retention mismatch. The new live test
demonstrates successful restoration after comparable pressure.

## 2. Qualified configuration

The authoritative operator profile is in the sibling benchmark repository:
`docker/deepseek-v41/compose.api.yaml`. Its service is `deepseek`; the container
name is `dsv41`. Do not confuse those names in Compose commands.

| Setting | Qualified value / meaning |
|---|---|
| Image | `ambientlight/dsv41-sm120:swa-retention-v1` |
| Deployment label | `dsv41-v2-swa-retention-v1` |
| `SERVER_PORT` / `SERVED_MODEL_NAME` | `8000` / `deepseek-v4-flash` |
| `REQUIRE_API_KEY` | `0`, preserving the existing no-backend-key endpoint |
| Parallelism / maximum running requests | TP4 / EP4 / 8 |
| `CONTEXT_LENGTH` | `524288` |
| `MEMORY_FRACTION` | `0.87` |
| `MAX_TOTAL_TOKENS` | Empty: automatic GPU KV sizing |
| `CHUNKED_PREFILL_SIZE` | `2048` |
| `PREFILL_DECODE_INTERVAL` | `4`, forwarded to `--prefill-decode-interval` |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:False` |
| `OFFLOAD_MODE` / `DSV41_CACHE_GIB` | `ram` / `0`; no separate NVMe Engram row cache |
| `DSV41_HICACHE_HOST_BUDGET_BYTES` | `256000000000`, **host total across all four ranks** |
| `SGLANG_OPT_HICACHE_SWA_RETAIN_REQUEST_BOUNDARIES` | `1` |
| `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND` | `python` |
| HiCache host mode / write policy | `cache` / `write_through` |
| HiCache IO backend / layout | `kernel` / `page_first` |
| Cache TTL / KV disk backend | Disabled (`null`) / none |
| Restart policy | `unless-stopped` |

The retention flag defaults to **off** in source. With it enabled, a non-Python
TreeCore backend fails explicitly. The policy is active only with HiCache, an
SWA host pool, and persistent `cache` mode. It is inactive in `buffer_only` mode.

The launcher accepts a nonnegative integer prefill/decode interval and forwards
it unchanged. This does not enable mixed-chunk batching or two-batch overlap,
and should not be described as a guarantee of simultaneous prefill and decode.

The launcher enables RAM HiCache for a nonzero host budget with these arguments:

```text
--enable-hierarchical-cache
--hicache-host-memory-mode cache
--hicache-write-policy write_through
--hicache-io-backend kernel
--hicache-mem-layout page_first
```

`DSV41_HICACHE_HOST_BUDGET_BYTES` is a local deployment extension, not a generic
upstream `--hicache-size` setting. The pinned V4 pool assembler rejects
`--hicache-size`; do not substitute that option. With the budget unset/zero,
the launcher does not enable HiCache and the assembler's original ratio-sizing
path remains available to other callers that explicitly enable HiCache.

## 3. Host memory accounting

The budget is **256 decimal GB**, approximately 238.42 GiB, not 256 GiB and not
256 GB per GPU. It covers planned HiCache allocations and allowances, not total
process RSS, all host memory, or the separate Engram backing weights.

The [budget planner](../python/sglang/srt/mem_cache/hybrid_cache/dsv41_host_budget.py)
derives page costs from actual tensor shapes, including:

- Shared C1/C2 source KV pages and their indexer payload/scale buffers.
- Target SWA plus packed DSPARK draft SWA pages.
- Logical and physical pool free-slot arrays, with allowances for three copies
  of slot indices while merging frees.
- A further host-wide 1 GiB metadata allowance.

For candidate FULL host page count `F`, the SWA count is proportional to the
GPU pool split:

```text
S = floor(F * device_swa_pages / device_full_pages)
planned_host_bytes = 4 * (
    F * (full_page_bytes + full_metadata_page_bytes)
  + S * (swa_page_bytes + swa_metadata_page_bytes)
) + 1 GiB
```

The planner selects the largest whole-page allocation within the budget. It
requires enough space for at least one GPU-cache equivalent. After assembling
the pools, it also checks actual initial KV buffers and free-slot arrays
against the host-wide budget. This is a bounded allocation plan, not an OS RSS
limit; Python objects, temporary allocations, and other processes still need
headroom.

The qualified layout is single-node TP4/PP1, non-unified V4.1 C1/C2 pools with
packed DSPARK draft pools. Unsupported layouts, missing draft state, misaligned
index pages, and insufficient budgets fail closed. The launcher accepts budgets
from zero through 256,000,000,000 bytes and requires RAM Engram mode for this
bounded rollout. Its preflight reserves the Engram backing shard sizes, the
requested HiCache budget, and an additional 24 GiB of available host RAM.

### Measured capacity at the qualified startup

| Quantity | Measurement |
|---|---:|
| FULL GPU capacity | 3,367,936 logical tokens |
| SWA GPU capacity | 26,112 slots/rank |
| FULL RAM capacity | 32,721,664 logical tokens |
| SWA RAM capacity | 253,440 slots/rank, 990 pages |
| SWA attention window / deployed page size | 128 / 256 tokens |
| Actual initial host buffers, all ranks | 244,417,175,040 bytes |
| Host KV payload, all ranks | 239,173,598,720 bytes |
| Planned allocation including allowances | 255,978,069,504 bytes |
| Minimum available RAM sampled in live pressure test | 39.232 GiB |

Logical capacities are not multiplied by TP4 to obtain a larger context limit.
The four ranks hold the corresponding distributed state. Automatic GPU sizing
can shift pool proportions and therefore host capacities after a restart.

## 4. Retention and transfer lifecycle

### Request-boundary preference

On a **non-chunked insertion**, the SWA component marks existing tree nodes
covering one sliding window plus one page at the request boundary. Intermediate
prefill chunks are not marked as completed-request boundaries.

For window 128 and page 256, this normally marks two pages. The extra page
handles exact replay's held-back last token, which can move the reusable end
back by a page. Coarse existing nodes may retain more state than this minimum;
the policy does not introduce new splits or allocate extra GPU memory.

Under SWA host pressure, eviction prefers ordinary intermediate checkpoints
before marked boundary windows. With session-aware caching, the order is:

1. Unreferenced ordinary checkpoints.
2. Any remaining unreferenced checkpoints, including marked boundaries.
3. Referenced ordinary checkpoints.
4. Any remaining eligible referenced checkpoints, including marked boundaries.

Without session-aware caching, only the ordinary-then-boundary distinction
applies. Existing IO locks, atomic host-leaf eviction, and cascading cleanup
continue to apply. Boundary marking is **not** a compute/IO pin. The fallback
pass can drain retained boundaries, so a full pool cannot grow without bound.

### Incremental backup

A request can repair missing SWA while its FULL KV is already backed up. The
component now collects missing device-to-host SWA copies within the required
window, including eligible ancestors, instead of unconditionally declining an
incremental backup. The core no longer skips a component merely because the
boundary node already has host state.

Collection stops at missing state or an in-flight write-through copy. Transfers
preserve node order and assign returned host indices to the corresponding
nodes. `buffer_only` keeps its existing node-local transfer contract.

**Copy scope differs from retention scope:** asynchronous backup collects only
the nominal SWA window. It must not copy the extra replay-lookbehind page
unless the device transfer lock is also extended. The existing lock covers one
window; a regression test protects this distinction.

### Splits and asynchronous host locks

Existing splits copy SWA host lock references to the new parent, keep locked
fragments off the host eviction LRU, and migrate the host UUID boundary once
to the older fragment. Boundary markers are inherited conservatively. These
changes preserve the lifetime of host data used by asynchronous transfers.

## 5. Source map and upstream lineage

| File | Responsibility |
|---|---|
| [boot.py](boot.py) | Budget validation, RAM preflight, HiCache flags, prefill/decode interval |
| [dsv41_host_budget.py](../python/sglang/srt/mem_cache/hybrid_cache/dsv41_host_budget.py) | Pure shape/page arithmetic and host-wide allocation plan |
| [hybrid_pool_assembler.py](../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py) | Planner integration, real allocation check and startup logs |
| [environ.py](../python/sglang/srt/environ.py) | Opt-in retention environment flag |
| [swa_component.py](../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py) | Boundary preference, incremental SWA backup, host-lock split handling, eviction totals |
| [unified_tree_core.py](../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py) | Permit required incremental auxiliary backup |
| [metrics_reporter.py](../python/sglang/srt/managers/scheduler_components/metrics_reporter.py) | Read independent SWA pool statistics without a tree scan or GPU work |
| [metrics_collector.py](../python/sglang/srt/observability/metrics_collector.py) | Scheduler fields and Prometheus exports |

These are selective adaptations, not a wholesale upstream rebase or a claim
that complete PRs were cherry-picked. Status is **as reviewed on September 18,
2026 UTC**, not a continuously updated tracker.

| Upstream work | Reviewed status | Treatment here |
|---|---|---|
| [#26907](https://github.com/sgl-project/sglang/pull/26907), [#27009](https://github.com/sgl-project/sglang/pull/27009): keep SWA suffix during eviction | Closed, unmerged | Not transplanted; the local policy does not add trim-on-evict splits |
| [#27210](https://github.com/sgl-project/sglang/pull/27210): limit compute-lock scope | Open, unmerged | Not backported; compute-lock span is unchanged |
| [#27444](https://github.com/sgl-project/sglang/pull/27444): pin host buffers through asynchronous restore | Merged; essential behavior already present | Preserved |
| [#34565](https://github.com/sgl-project/sglang/pull/34565): SWA branching-point caching | Merged | Selected window-scoped incremental backup plumbing, not the complete branch scheduler feature |
| [#37584](https://github.com/sgl-project/sglang/pull/37584): branch-caching follow-up | Merged | Followed current Python collector/buffer-mode behavior; no Rust enablement |
| [#38138](https://github.com/sgl-project/sglang/pull/38138), [#38835](https://github.com/sgl-project/sglang/pull/38835): split/host-lock bookkeeping | Merged | Host lock inheritance and final single-UUID-migration behavior adapted |
| [#38482](https://github.com/sgl-project/sglang/pull/38482): split LRU recency | Merged | Not included; no new splits introduced by retention |
| [#38926](https://github.com/sgl-project/sglang/pull/38926): V4.1 L2/L3 tests | Open, unmerged | Qualification reference only; GB300 results do not qualify our SM120 deployment |

The current upstream file was renamed to
[`components/swa.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/swa.py).
This pinned tree still uses `swa_component.py`. The local completed-request
eviction preference is a deployment-specific addition, not a claim of upstream
support for that exact policy.

## 6. Telemetry and read-only diagnosis

All names below have the `sglang:` prefix. Select `tp_rank="0"` for the logical
capacities and snapshot gauges; do not sum identical rank capacities as if
they represented independent prompts.

| Metric suffix | Interpretation |
|---|---|
| `hicache_host_used_tokens`, `hicache_host_total_tokens` | FULL host occupancy/capacity; insufficient alone to establish hybrid reuse |
| `hicache_swa_host_used_tokens`, `hicache_swa_host_total_tokens` | Independent SWA host occupancy/capacity |
| `hicache_swa_host_evicted_tokens` | Cumulative SWA slots reclaimed since cache construction |
| `hicache_swa_host_boundary_evicted_tokens` | Cumulative marked boundary slots reclaimed, including FULL-driven eviction |
| `hicache_swa_host_retention_enabled` | Effective policy enablement, 0 or 1 |
| `load_back_tokens_total{pool="kv"}` | FULL host restoration counter |
| `load_back_tokens_total{pool="swa"}` | SWA host restoration counter |

The two eviction totals are **gauges containing cumulative snapshots**, not
Prometheus Counters. They reset with cache reconstruction; compare within one
deployment/cache lifetime. The load-back counters aggregate ranks in the
current export: a measured 1,024-slot SWA restore across TP4 is 256 slots/rank.

Useful read-only checks:

```bash
curl -fsS --max-time 10 http://127.0.0.1:8000/v1/loads
curl -fsS --max-time 10 http://127.0.0.1:8000/metrics \
  | rg 'hicache_(swa_)?host_|load_back_tokens_total'
docker inspect dsv41 \
  --format '{{.State.Status}} {{.State.StartedAt}} {{.RestartCount}}'
docker logs --since 30m dsv41 2>&1 \
  | rg 'DSV41_HICACHE_BUDGET|DSV41_HICACHE_ALLOCATED|Scheduler hit an exception|ncclUnhandledCudaError'
readlink -f /mnt/hot/dsv41_dumps/current
```

Use `/v1/loads`, not deprecated `/get_load`. Startup sizing records may be older
than the example's 30-minute log window. Capture lives under
`/mnt/hot/dsv41_dumps/`; the `current` link follows the active capture. Captures
can contain private prompts and responses: preserve them in access-controlled
storage, never in Git.

### Reading the symptoms

| Observation | What to investigate |
|---|---|
| SWA pool full, ordinary evictions increasing, no boundary evictions | Expected bounded-cache pressure; fullness alone is not a failure |
| Boundary evictions increasing while FULL occupancy is low | SWA pressure, coarse marker granularity, session priority, or FULL-driven leaf cleanup; correlate with actual replay misses |
| Exact older replay misses | Check restart/flush, input IDs, namespace, surviving FULL and SWA state, and host transfer counters |
| Prompt differs before the saved boundary | A genuine prefix difference can prevent reuse; retained suffix state cannot bridge different tokens |
| Messages usage lacks a cache-read field | Use scheduler hits and independent pool restore counters; absence of that usage field does not prove a miss |
| Policy gauge is zero | Check image, flag, Python backend, HiCache enablement, SWA host pool and persistent cache mode |
| CUDA allocation retries | Separate transient allocation pressure; this retention patch does not fix the allocator or prove NCCL stability |

Known accounting caveat: the pinned paged-host-pool transfer **byte** metrics
misinterpret page/layer units as per-token units. Do not convert those raw
backup/load byte counters into physical bandwidth. Use logical pool slots and
the separately verified shape-based allocation records. This patch does not
repair that byte-metric issue.

## 7. Tests and qualification

### CPU tests without inference load

The deployment tests use actual Python TreeCore/component code and small CPU
fixtures; no model or native CUDA execution is involved. From this repository
on the qualified workstation:

```bash
docker run --rm --pull never --network none --runtime runc \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/overlay/python:/opt/dsv41/adapter \
  --mount type=bind,src=/mnt/hot/ambientlight/repos/sglang-dsv41-production-overlay,dst=/overlay,readonly \
  --workdir /overlay --entrypoint python3 \
  ambientlight/dsv41-sm120:sero-45f538a-baseline \
  -m pytest -q -p no:cacheprovider --tb=short deployment
```

For the wider API/parser/lifecycle suite, from the sibling benchmark repository:

```bash
bash scripts/dsv41_api_regressions.sh deployment
```

That script defaults to the pinned CPU baseline, records evidence under
`/mnt/hot/dsv41_state/api-tests-*`, and mounts this source read-only. Do not
casually substitute the production image: it expects its external durable
thinking-signature key for some API tests. Use the documented baseline or an
explicitly equivalent test harness, not a copied production secret.

| Test file | Coverage |
|---|---|
| [test_boot.py](test_boot.py) | Existing identity/auth/sampling plus budget bounds, RAM cache flags, interval forwarding and invalid-input rejection |
| [test_hicache_budget.py](test_hicache_budget.py) | 192/256 GB sizing, actual layout costs, 500 deterministic rounding cases, metadata, unsupported layouts and unchanged ratio fallback |
| [test_hicache_assembly.py](test_hicache_assembly.py) | Real assembler with tiny CPU tensors and mocked allocations; shape costs match assembled pools |
| [test_swa_host_retention.py](test_swa_host_retention.py) | 16 real-tree regressions: legacy misses, 64 long-prefix restores, payload/index order, chunk versus boundary marking, fallback, sessions, split locks, incremental repair, holes, buffer mode, transfer-lock scope, backend rejection and real Prometheus export |

The 64-prefix CPU fixture uses the **older** 254,976-slot SWA pool and 24,064
slots per cold request to reproduce the captured failure shape. It is not a
claim that the current GPU run replayed all 64 prompts.

Before rollout, the full suite passed **1,202 tests and 377 subtests**; 87
benchmark deployment/helper tests passed separately. The built image's own
installed cache modules passed the 16 SWA tests. Those 16 also occur in the
full suite; the counts must not be added as unique tests.

The same full suite was rerun successfully when preparing these local commits:
1,202 tests and 377 subtests passed in 153.66 seconds. The six SWA runtime/test
file hashes also matched the deployed image's frozen receipt exactly.

### Live qualification already completed

| Check | Result |
|---|---|
| Sixteen independent 256,000-token cold requests | 16/16 correct |
| Oldest exact replay after all sixteen | 255,744 cached tokens, correct answer, 1.540 s TTFT |
| Long replay FULL restore | 1,022,976 slots across TP4 = 255,744/rank |
| Long replay SWA restore | 1,024 slots across TP4 = 256/rank |
| Four header-needle replays, two each on Responses/Messages | 4/4 correct; each restored 7,168 FULL and 256 SWA slots/rank |
| Intermediate SWA slots reclaimed | 139,776/rank |
| Marked boundary slots reclaimed | 0 |
| API failures / fatal scheduler-CUDA-NCCL errors / unexpected restarts | 0 / 0 / 0 |

The pressure run used 4,352,000 input tokens and 51 output tokens across
17 Responses requests. Its mean cold TTFT was 44.206 s. A prior comparable
16-prompt run's oldest replay had zero cached tokens and 43.391 s TTFT. Seeds
and small automatically sized capacities differed: this is a behavioral
before/after result, not a controlled general throughput benchmark.

Header-needle answers were near the beginning of the cached region, not in the
uncached tail. Each replay required both FULL and SWA restore-counter deltas.
The harness checked that no unrelated HTTP inference was admitted during the
measurement. An initial harness pass stopped on unsettled HTTP-finalization
metrics after correct answers; waiting for settled counters resolved the
measurement issue. It was not an API/model failure.

There were 86 recoverable CUDA allocation-failure debug log lines during the
live qualification. These are not 86 failed requests or necessarily 86 distinct
retries. Successful restoration does not eliminate that pre-existing pressure.

To repeat GPU qualification, obtain an idle maintenance window first. The
benchmark repo supplies `scripts/dsv41_swa_retention_probe.py` (prime/replay) and
`scripts/dsv41_cache_pressure.py`. Prime both API header probes, issue enough
independent long prompts to exceed GPU FULL and host-SWA checkpoint capacity,
then replay the oldest prompt and both probes. Require correct answers, valid
streams, unchanged container identity, and **both** restore counters advancing.
Observe RAM headroom and request attribution. These commands generate expensive
inference traffic and must not be run as casual status checks.

## 8. Packaging, provenance and operations

The qualified SWA image is a five-runtime-file layer over the existing v2 image:

```text
Parent: sha256:fb61d2521beb6188652417734a8b402cabd5209675310cb3d610bfe95f08c5ca
SWA:    sha256:2d811ff02b68885294bfb8da4a75ec357ccb62bc7d95ff83c690b5575f40a5cc
```

The five files are `environ.py`, `swa_component.py`, `unified_tree_core.py`,
`metrics_reporter.py`, and `metrics_collector.py`. API/parser, main scheduler,
tokenizer-manager, SM120 attention-kernel, Engram adapter and binary dependency
bytes were preserved and verified against the parent.

The image was built from a frozen dirty-source snapshot above `47d89e6` before
these source commits were made. The snapshot receipt records the six source/
test SHA-256 hashes. Committing those same bytes does not rebuild the image or
retroactively change its provenance. The running deployment additionally mounts
the launcher, host-budget helper and assembler read-only from this repository.

Build/rollout helpers live in the sibling benchmark repository:

- `Dockerfile.swa-retention` and `dsv41_build_swa_retention_image.py` recorded the
  one-off reviewed snapshot. **That builder assumes pre-SWA HEAD matches the
  parent image. After committing SWA, it is not a generic repeatable builder; its
  baseline guard will reject a changed HEAD. Do not disable that guard.**
- `Dockerfile.api` and `dsv41_build_api_image.py` contain the expanded explicit
  source manifest for a normal clean, committed overlay build. Use a fresh
  candidate tag and verify its receipt before any future promotion.
- `dsv41_relaunch_swa_retention.py` was the one-shot guarded rollout: verify
  identities and hashes, wait for 60 continuous seconds with no active/queued
  work or new inference admission, check RAM headroom, then recreate only the
  inference service. It is not a generic restart loop.

Local commits alone do not restart or alter the running service. Preserve
the Compose profile, diagnostic entrypoint, model/state mounts, durable
thinking key, kernel cache and packet recorder when rebuilding or redeploying.
Never commit secrets, captured conversations, model weights or diagnostic dumps.

### Evidence locations on the workstation

These are external artifacts, not files expected in a fresh clone:

| Artifact | Location |
|---|---|
| Frozen image sources, SHA-256 manifest and receipt | `/mnt/hot/dsv41_state/swa-retention-image-8yswbpcd/` |
| Guarded rollout, saved profiles and qualification summary | `/mnt/hot/dsv41_state/swa-retention-rollout-grf55j8h/` |
| Live pressure results | `pressure/results.json` under the rollout directory |
| Both-API replay results | `anchors-settled/replay-20260918T053618Z/` and `anchors/replay-20260918T053622Z/` under the rollout directory |
| Wider pre-rollout regression evidence | `/mnt/hot/dsv41_state/api-tests-pUlmvL5h/` |
| Commit-preparation regression rerun | `/mnt/hot/dsv41_state/api-tests-ZLWbBnNx/` |
| Historical 64 × 256k failure capture | `/mnt/hot/dsv41_state/cache-pressure-64x256k-bjNludyH/` |
| Capture at qualification | `/mnt/hot/dsv41_dumps/capture-20260918T051306Z-956043/` |
| Expanded investigation report | Sibling repo `docker/deepseek-v41/SWA-RETENTION-2026-09-18.md` |

### Rollback

Rollback requires a drained workload and discards the in-memory KV cache. The
old v2 image and saved `rollback.compose.json` remain under the recorded image
and rollout identities. Before use, verify that the saved profile still targets
the intended model, mounts, port, service and parent image.

Recreate only service `deepseek` using the saved rollback profile with
`docker compose ... up -d --no-build --pull never --force-recreate deepseek`.
Also restore the persistent benchmark Compose image/environment/version and
release metadata; otherwise the next normal Compose launch will select the
new SWA image again. Wait for all startup smoke checks, endpoint readiness,
capture heartbeat and expected memory allocation records.

Setting the retention flag to `0` in a future drained restart disables the new
eviction preference, but does **not** undo incremental backup or split-lock
fixes. A source `git revert` alone does not roll back an already built image.
Do not edit live bind-mounted deployment sources to simulate a runtime rollback.

## 9. Boundaries of the result

- Retention favors completed-request reuse, not every possible internal branch
  point. Genuine prompt changes still invalidate the corresponding prefix.
- There is no new TTL, permanent pin, restart persistence, or NVMe KV tier.
  Capacity pressure, FULL eviction, flush and restart can remove useful state.
- Existing node granularity and session priority can change how many boundaries
  fit. No universal fixed number of retained conversations is promised.
- The patch does not change Responses/Messages wire semantics, sampling, model
  output validation, kernels or the hardware-qualified runtime dependencies.
- CPU tests do not qualify native CUDA kernels. Live tests establish targeted
  RAM restoration, not logit/KL equivalence, all Rust/Mamba configurations, a
  64-prompt GPU replay sweep, long-output stability, or sustained concurrent load.
- Recoverable allocation pressure and physical transfer-byte metric accuracy
  remain separate issues.
