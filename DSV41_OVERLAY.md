# Image-pinned V4.1 / V12 API integration worktree

This tree starts with the Python source extracted from 0xSero's pinned base image,
with the recipe's supplied `flash_mla_sm120.py` replacement. It is not a checkout
of current upstream SGLang, and its original Git revision is unresolved.

- Recipe: `0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000` at
  `45f538a18569420721e00e37353f9a7e1af7e5da`.
- Base image: `lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5`.
- Source layer: `sha256:52f078c87b74c74ff1f1a469469a62031ce00c710700b2f5c278ad791961ab08`.
- API behavior/tests donor: local V12 fork at
  `71c498b991ce143999132aafd951d16d6b119805`, relative to
  `2c03d62532c880d17ef356c24275d1f7117c1d69`.

The Engram adapter/import hooks and all binary dependencies remain supplied by
the pinned recipe/image. Keep the overlay limited to reviewed API/parser,
deployment, and explicitly qualified cache changes; do not replace kernels or
dependency versions as part of these integrations.

The initial import commit records provenance, not hardware qualification. Prepare
and run CPU regressions independently, then qualify on top of the reproduced
four-GPU baseline. Full deployment status and recipes are maintained in
`../rtx-pro-6000-bench/docker/deepseek-v41/`.

## API port scope

The first API overlay changes thirteen Python runtime files, plus a copy of the
pinned boot script that only adds preferred sampling settings and metrics. It
ports V12 transactional tool publication, ordered fragments, parser containment,
one bounded recovery attempt, strict/non-strict schema semantics, durable omitted
thinking, failed-tool-result history, and stream cleanup. The image-pinned
scheduler/tokenizer manager already implement the ownership/abort fixes; their
files remain unchanged and are covered by the adapted V12 regression tests.

V4.1-specific adaptations:

- Build all DSML parsing boundaries from the dialect's spaced names.
- Backport the spaced structural grammar from upstream `dsv4.1` commit
  `01d34d7b404451bce7f7b35e306de6fab395024d` (#38951). Require an argument object
  for non-strict calls without enforcing their declared fields.
- Use the spaced calls marker for reasoning-to-tool transitions.
- Preserve string-parameter whitespace, matching the pinned publisher decoder.
- Preserve native inline system-message chronology; do not apply V4's
  latest-reminder transformation to V4.1.
- Share Chat/Responses engine-prompt selection. The pinned Responses adapter
  incorrectly selected the empty text field for native-ID vision encoders,
  including text-only requests on V4.1. Preserve native IDs and image metadata;
  keep standard VLMs on their existing rendered-text path. Coverage exercises
  actual request conversion through both response modes, plain/tool output and
  text/image input, not just post-generation stream parsing.
- Preserve the pinned reasoning tiers (low=25, high=50, xhigh=75, max=100).
  Updating the publisher encoding revision/tier mapping is a separate change.
- Missing sampling fields use the operator profile: temperature 1, top-p .95,
  output budget 32768. Explicit caller settings win. Responses retains its
  existing two-token framing reserve (32766 actual default generation budget).

The selected integration covers function tools, not newer upstream Responses
custom/freeform tools. No backport of their separate API surface is claimed.

CPU coverage includes V12 capture-derived corruption fixtures adapted to V4.1,
real parser-to-API streaming tests for both endpoints, compiled grammar acceptance
and rejection, encoder contracts, lifecycle cleanup and launcher policy. The
benchmark repo additionally compares all five pinned publisher prompt goldens.
Hardware and live endpoint qualification are separate gates, not inferred from
these unit tests.

See [Responses additive-prefix stability](deployment/RESPONSES_PREFIX_STABILITY.md)
for the follow-up fix to Responses' pre-encoder system-message hoisting,
captured-request token replay, regression coverage, and rollout limits. The
September 18 local qualification was followed by an approved deployment of
`ambientlight/dsv41-sm120:responses-prefix-v1`, with live additive-prefix and
API checks. This follow-up was built from an explicitly hashed uncommitted
snapshot before the local commit; its build manifest preserves that provenance.

## Deployment and HiCache reference

See [RAM HiCache and SWA retention](deployment/HICACHE_SWA_RETENTION.md) for the
complete configuration, host-total budget calculation, cache lifecycle,
upstream backport rationale, metrics, tests, qualification evidence, build
provenance, and rollback procedure.

The September 18, 2026 UTC qualified profile uses RAM Engram offload, a
256,000,000,000-byte host-wide HiCache budget, static fraction 0.87, and prefill
decode interval 4. SWA request-boundary retention is explicitly enabled on
Python TreeCore. Both FULL KV and SWA windows are stored in RAM and restored to
GPU on a hit. Retention is a bounded eviction preference, not permanent pinning.
The API containment and hardware baseline described above are preserved.

## Legacy endpoint promotion (initial profile)

The boot wrapper accepts `SERVED_MODEL_NAME` and the existing `SERVER_PORT` to
publish V4.1 behind the V12-compatible `deepseek-v4-flash` alias on port 8000.
`REQUIRE_API_KEY=0` explicitly restores V12's no-backend-key access; the default
remains authenticated for other deployments. Both startup smoke calls and health
checks follow these settings. Parser/encoder selection remains native V4.1,
independent of the public model alias; no kernel or serving implementation is
replaced by this deployment change.

The initial promoted Compose profile selected context 524288, container `dsv41`, and
`unless-stopped`, preserving TP4/EP4, 2048-token prefill, memory fraction 0.85,
eight request slots, and DSPARK block 5. The V12 durable thinking key is retained
for legacy history compatibility: its signed payload binds to the public model
alias, not cryptographically to the checkpoint revision. The embedding model
and previous inference containers remain stopped for rollback.
