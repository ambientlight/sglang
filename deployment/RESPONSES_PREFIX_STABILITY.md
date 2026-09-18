# V4.1 Responses: preserve additive instruction history

Qualification and rollout snapshot: **September 18, 2026 UTC**. This fix was
first verified locally, then deployed in a separately approved restart as
`ambientlight/dsv41-sm120:responses-prefix-v1`. The image was built before the
local commit; its manifest records the exact then-uncommitted source snapshot.
Recording those same runtime bytes in Git does not require an image rebuild.

## Failure and bounded fix

Three captured `/v1/responses` continuations had exactly additive input arrays,
unchanged top-level instructions and tools, but almost entirely missed cache.
`_construct_input_messages` normalized developer messages to system messages,
then moved **all** system messages into a combined leading block. Appended
skills or interruption notices consequently changed the start of the model's
input, invalidating hundreds of thousands of tokens of otherwise reusable KV.

For the resolved `dsv41` encoder only, the adapter now:

1. Coalesces the initial contiguous system/developer instructions as before.
2. Leaves later system/developer updates at their chronological positions.
3. Keeps the existing developer-to-system normalization and assistant/tool
   merging. It does not discard instructions or turn them into user text.

The native V4.1 encoder already supports inline `<｜System｜>` blocks. Passing
through raw `developer` roles instead would be incorrect here: this pinned
encoder renders them like user messages and can drop them under its
thinking-history policy. The API role-contract check therefore guided keeping
the existing instruction-role normalization. OpenAI's Responses documentation
places developer/system instructions above user instructions; it does not
require moving later messages to the beginning of the prompt.
([Official Responses reference](https://developers.openai.com/api/reference/python/resources/responses))

Selection uses `chat_encoding_spec`, not the public model alias. The deployment
can keep serving `deepseek-v4-flash`. Other encoders retain their existing
global system-message coalescing. Messages/Chat conversion, tool publication,
recovery, sampling, kernels, and cache policy are unchanged.

Source: [Responses conversion](../python/sglang/srt/entrypoints/openai/serving_responses.py).

## Captured-request replay with the actual tokenizer

The CPU-only replay loads the local checkpoint's tokenizer and runs the actual
`_make_request` path, including native-ID engine-prompt selection. Before
testing the patch, it reproduces **every input token of all 12 captured
requests** using the deployed conversion method. This establishes that the
offline renderer matches the recorded production inputs, rather than merely
matching lengths or decoded text.

For the three defective additive pairs:

| Response ID suffixes, previous → next | Shared tokens before fix | Shared tokens after fix | Previous patched input preserved |
|---|---:|---:|---:|
| `34b11acb…` → `a178a053…` | 20,787 | 383,871 | 100% |
| `a178a053…` → `39bd9086…` | 22,753 | 387,949 | 100% |
| `74d0f8fc…` → `cea081a1…` | 22,801 | 390,439 | 100% |

The seven other additive pairs also retain their complete previous input
prefixes. The twelfth request belongs to a separate conversation and is not
treated as an additive pair. New inline message boundaries add a small number
of tokens, so the patched totals differ slightly from the old prompt lengths.

These are **token-prefix measurements, not observed live cache-hit rates**.
The replay submits no inference requests and performs no cache operations.

Local evidence, outside Git:

- Diagnosis and captured inputs:
  `/mnt/hot/dsv41_state/responses-cache-20260918-rd8qcwDR/`.
- Replay script and metadata-only results:
  `/mnt/hot/dsv41_state/responses-prefix-fix-pllIGipT/replay.py` and
  `replay-summary.json`.
- API/parser/lifecycle/deployment test report:
  `/mnt/hot/dsv41_state/api-tests-2eECRhM8/`.

Captured prompts remain private; the new portable test fixture contains only
synthetic examples. The replay was performed in a network-disabled, CPU-only
container using the existing deployed image's dependencies and a read-only
mount of this source overlay and the local checkpoint.

| Artifact | SHA-256 |
|---|---|
| Pre-fix deployed `serving_responses.py` | `280b18d88b572c9eab24bfa6ce3855d253324d3321d15622928b6ddd6adff3c9` |
| Patched `serving_responses.py` | `f9ed5c31b6218b9dd63f786a2525ac400798c686b71b553990928b6c308a643a` |
| Checkpoint `tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |

## Regression coverage

[The new portable suite](../test/registered/unit/entrypoints/openai/test_dsv41_responses_prefix.py)
passes **9 tests plus 36 subcases**, covering:

- Initial instructions, late system/developer content lists, and input immutability.
- Appended skills, abort notices, and ordinary updates through the real encoder.
- Thinking/chat modes and text-only/vision-capable model configurations.
- Initial and inline tool-schema placement, assistant reasoning/tool merging,
  tool-result continuations, and stored response history.
- Missing initial system messages and compatibility with other encoders.
- Genuine initial-instruction changes still invalidating the prefix.

Portable tests use injective UTF-8 byte IDs instead of a constant tokenizer
mock. The separate captured-request replay above verifies actual checkpoint
token IDs. Before the runtime fix, the new suite reproduced the failures.

The broader CPU run passed **1,211 tests plus 413 subcases** (including this
suite), covering Responses, Messages, Chat, tool/reasoning parsers, lifecycle
cleanup, and deployment policy. All **five publisher encoder goldens** remain
byte-for-byte unchanged. No throughput or long-context stress test was run.

To rerun these checks from the sibling benchmark repository:

```bash
bash scripts/dsv41_api_regressions.sh deployment
python3 scripts/dsv41_encoder_reference_check.py
```

## Approved rollout and live qualification

The follow-up rollout confirmed 60 continuous idle seconds using scheduler
loads and HTTP inference counters, then recreated only `dsv41`. The image adds
one runtime file and a source manifest to the exact SWA-retention image. Hash
checks verified 25 protected API/cache/kernel/launcher files were unchanged.
The packaged code passed the new 9 tests plus 36 subcases before restart.

| Field | Observed value |
|---|---|
| Image | `ambientlight/dsv41-sm120:responses-prefix-v1` |
| Image ID | `sha256:90d0ef76ed2da4a1bae09a7b9db015bf377952ac77a4b002d3892ce993cc13c3` |
| Container | `d2818c9bc2c3c1ddf492f5586726c410641c9c9eace7f1d660e4ca0cefb0fd7d` |
| Started / launcher checks passed | 07:20:25 / 07:25:56 UTC |
| Endpoint / public model | Port 8000, `deepseek-v4-flash`, no backend key |
| Context / static fraction / concurrency | 524,288 / 0.87 / 8 |
| Host cache budget / Engram | 256,000,000,000 bytes host-total / RAM |
| GPU FULL / SWA capacity per rank | 3,373,312 / 26,112 tokens |
| Host FULL / SWA capacity per rank | 32,728,320 / 253,184 tokens |
| Restart policy / observed restarts | `unless-stopped` / 0 |

The persistent Compose file now selects the new image and deployment label;
all other resolved configuration matches the rollout candidate. The previous
image and private rollback configuration are retained. The live Responses
source hash matches the tested patched hash above. At image-build time, source
HEAD was `28cedd2361af42bcec636537a713e874a40390ff` with the fix uncommitted;
that historical identity remains in the build receipt after the local commit.

Live checks passed:

- Launcher: arithmetic, structured output, tool round trip, and native vision.
- Eight fully traced Responses calls, covering chat and thinking modes with
  appended skills, interruption notices, and system updates. All six additive
  pairs preserve the entire previous native input-token prefix. Each
  continuation reused 4,864 device-cached tokens, or 94.8–97.3% of its roughly
  5k-token input. Native and terminal API usage agree, and all answers follow
  the newest instruction. The partial final page and appended material still
  require prefill; the test did not exercise RAM restoration under pressure.
- Seven additional API calls: Responses and Messages strict tool round trips,
  omitted-thinking signature replay, and Responses native image input.

An earlier eight-call prefix probe also completed with correct answers and
cache hits, but native logging attached partway through it (07:26:29 UTC).
Its incomplete native evidence was not accepted as proof; the complete probe
above was repeated after logging attached. Both runs' artifacts are preserved.

Rollout/build/rollback evidence:
`/mnt/hot/dsv41_state/responses-prefix-rollout-ButMPOyN/`.
The fully traced prefix run is `prefix-live-20260918T072738Z/`; the API run is
`acceptance/api-20260918T072814Z/`. Capture follows the new container at
`/mnt/hot/dsv41_dumps/capture-20260918T072028Z-1135988/`, also exposed via
`/mnt/hot/dsv41_dumps/current`. Packet capture and native logging are active;
the post-check packet counter reported zero kernel drops.

## Limits

A restart clears this deployment's non-persistent GPU/RAM KV cache, and the
corrected serialization also differs from the old one for histories containing
inline instructions. Each conversation therefore needs a new warm-up request.

After warm-up, verify additive continuations with appended instructions using
native input-token prefixes and terminal API cached-token usage. Actual hits
still depend on FULL/SWA state availability, cache namespaces, page alignment,
eviction, and workload pressure. This patch does not guarantee full hits when
those conditions are unmet; see [SWA retention](HICACHE_SWA_RETENTION.md).

Actual changes to initial instructions, tool schemas, earlier history, or
encoder settings must still invalidate affected KV. The native no-tools
thinking-history policy can also rewrite prior thinking when a new user turn
arrives; this patch deliberately does not change that separate encoder policy.
