# Bounded CUDA/NCCL crash diagnostics

This change addresses the September 18 DSV4.1 diagnostic gaps. It does **not**
claim to fix the initiating GPU/NCCL stall and changes no attention/MoE kernel,
NCCL algorithm/transport/version, CUDA graph mode, cache policy, or API format.

## Failure addressed

GPU 3 reported CUDA 702 at 08:38:28 PDT. Sequential, unbounded `py-spy --native`
collection delayed CUDA dump triggers until 08:47:30. The original failing rank
was already gone; only three survivors were dumped. All three were in a BF16
ring all-reduce. The other diagnostic gap was the missing PyTorch exception-path
flight-recorder opt-in, despite an enabled timeout-dump setting.

## Collection order and bounds

1. A hard scheduler watchdog logs the failure and immediately notifies its parent
   with SIGQUIT. It does not block notification on its own stack collector.
2. The parent requests PyTorch flight-recorder dumps through the configured
   current-run rank FIFOs, then CUDA user core dumps for discovered schedulers.
   There is no initial five-second sleep. These requests precede request pickling.
3. Nonblocking Python stack collection runs for all ranks before optional native
   stack collection. Commands use argv, not a shell, and own process groups.
4. Each command has a timeout. Its process group is killed on completion/timeout
   to prevent an orphan descendant keeping collection alive. Reaping is bounded;
   an uninterruptible process is logged rather than awaited indefinitely.
5. Stack collection and dump waiting share one overall window; there is no
   additional fixed one-minute wait after stack collection.
6. A daemon collector with a bounded join allows cleanup to proceed even if
   process discovery or a filesystem operation stalls. Fatal cleanup call sites
   use `finally`, so an exception while dumping requests cannot suppress cleanup.

| Setting | Default / production value | Meaning |
|---|---:|---|
| `SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS` | 60 | Overall CUDA/NCCL/stack collection window |
| `SGLANG_PYSPY_DUMP_TIMEOUT_SECS` | 5 | Maximum wait for each stack command |
| `SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS` | 20 | Overall stack-collection budget, also capped by the outer deadline |
| `SGLANG_PYSPY_DUMP_NATIVE` | 1 | Native stacks after the nonblocking Python pass |
| `SGLANG_NCCL_DUMP_BEFORE_CRASH` | 1 | Explicit best-effort flight-recorder FIFO trigger |
| `SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS` | 10 | Flight-recorder completion allowance within the same window |
| `SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS` | 60 | Existing CUDA completion allowance, now capped/shared |
| `TORCH_NCCL_EXTRA_DUMP_ON_EXEC` | 1 in deployment | Enable PyTorch's separate watchdog-exception dump path |

Negative/nonfinite budgets fall back to finite defaults; zero can disable a wait
or collection budget. Existing py-spy/CUDA enable flags remain honored. Missing,
unread, or full FIFOs do not block. Symlinks and non-FIFOs are not written.
The manual NCCL trigger selects only `<configured-prefix><numeric-rank>.pipe`.

The 60-second bound is for this diagnostic phase, **not** model reload time,
optional request-data serialization, arbitrary OS/driver teardown, or a global
kernel/GIL deadlock. Native py-spy can briefly suspend a failed worker, whereas
the first Python pass uses `--nonblocking`. No collector can guarantee evidence
from a process/device that has already exited/reset, or completed CUDA cores
before a forced shutdown. `skip_global_memory` continues to omit bulk buffers.

## PyTorch version-specific behavior

The deployed PyTorch is `2.13.0+cu130`, revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`. `Watchdog::run` calls
`dumpExtraDebuggingInfo` on a CUDA exception; that helper is gated by
`TORCH_NCCL_EXTRA_DUMP_ON_EXEC`, default false. Its manual dump avoids requesting
stack traces in that exception path to reduce GIL-deadlock risk. Keep the existing
timeout, monitoring, trace-buffer and output-prefix settings as well.

See [the pinned PyTorch implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1487-L1508).
Do not disable CUDA-error rethrow/abort just to keep a broken worker alive.
PyTorch's recorder does not cover every direct SGLang PyNccl call; native NCCL
logs and CUDA cores remain complementary evidence.

## Tests and deployment

CPU-only tests require Python and psutil, not torch/CUDA initialization:

```sh
python3 test/registered/unit/utils/test_crash_diagnostics.py -v
```

Coverage includes actual hanging subprocesses and descendants, missing tools,
permission errors, safe FIFO triggering, rank/prefix filtering, command ordering,
total deadlines, failed stages, no-reader/no-live-rank handling, soft-vs-hard
watchdog semantics, and the real tokenizer-manager fatal cleanup methods.

The deployment repository adds `Dockerfile.diagnostics` and
`scripts/dsv41_build_diagnostics_image.py`. The builder freezes four source files,
tests and this guide under `/mnt/hot/dsv41_state/`, verifies their committed
baselines against the pinned Responses-prefix image, preserves parent layers,
checks protected serving files byte-for-byte, and tests the built image with no
GPU access. It never restarts inference. The receipt records the exact source
hashes, image ID and protected-file hashes, even before changes are committed.

Production activation needs container recreation. Do not attach debuggers, write
to live CUDA dump pipes, inject GPU failures, or run throughput/pressure traffic
as routine validation. A real hardware-failure capture remains unqualified until
such a failure naturally recurs or a separate fault-injection test is approved.
