# Binary-safe, supervised deployment logs

## Failure addressed (2026-09-20)

L15's launcher reader died at 08:06:24 UTC with a `UnicodeDecodeError` from
`for line in process.stdout`. The `Popen(text=True)` reader was strict UTF-8;
the unread pipe remained open while the launcher waited for the child. The wire
capture then contained TCP acknowledgements but no HTTP/SSE payload until the
container was stopped roughly 44 minutes later. Docker needed SIGKILL after its
90-second stop timeout. There was no corresponding NCCL error in the preserved
rank logs. Which writer originated/interleaved the invalid bytes is unknown.

An isolated reproduction using the original reader produced the same exception,
a full 65,536-byte pipe, and a child blocked in `anon_pipe_write`. Draining the
pipe released the child. This is a launcher liveness bug, not evidence that the
67/33 HiCache split or multilingual model output itself is invalid.

## Implementation

`deployment/boot.py` launches SGLang with an unbuffered **binary** subprocess
pipe. `forward_log_bytes` reads at most 64 KiB at a time and writes available
bytes immediately, without decoding or waiting for a newline. Valid Unicode,
split multibyte sequences and invalid byte sequences do not kill the reader.
Docker's logging layer still performs its own framing/encoding; the independent
PCAP remains the authoritative HTTP request/response capture.

The configured backend key is redacted as bytes, including matches split over
multiple reads. The forwarder holds only a possible key prefix between reads;
it does not accumulate an entire multi-megabyte request log. Short output writes
are completed; a non-progressing or broken sink is an explicit failure.

`ServerLogForwarder` supervises the reader. A read/write exception, or pipe EOF
with a server still alive after a one-second exit grace, marks the launcher
failed, closes the unread pipe and stops the server process group. Shutdown has
a 10-second SIGTERM grace followed by SIGKILL and a five-second reap limit. The
main thread independently checks failure during startup and serving; the reader
also stops the child if failure occurs during a blocking startup smoke request.
Exception payloads are not echoed because they can contain request data/keys.
Nonzero model exit status is preserved. Ordinary external shutdown remains
graceful before the bounded kill fallback.

This is not a general solution to a Docker log sink that blocks indefinitely
without raising. It prevents the diagnosed dead-reader/open-pipe failure.

## Verification

`deployment/test_boot_logging.py` contains 17 CPU-only regressions for malformed
UTF-8 followed by output larger than pipe capacity, valid Unicode fragments,
concurrent stdout/stderr writers, cross-chunk key redaction and overlapping key
prefixes, short/failed writes, bounded large-line forwarding, immediate small-line
flushing, unexpected EOF, nonzero exits, sink failure during startup, and a child
that ignores SIGTERM. Existing launch/auth/sampling tests remain in `test_boot.py`.

Run in the pinned image without GPU access:

```sh
python3 -m pytest -q deployment/test_boot.py deployment/test_boot_logging.py
```

The deployment repository's `scripts/dsv41_relaunch_log_forwarder.py` additionally
tests the frozen boot snapshot in a network-disabled, GPU-disabled container,
requires a passing full API suite and accepts only the boot bytes plus deployment
label as changes. The model image, inference settings, diagnostics and cache
split must be identical. Relaunch waits for 60 continuous idle seconds using
both `/v1/loads` and HTTP activity counters, retains private rollback snapshots,
and makes no automatic second attempt or rollback. Normal built-in startup smoke
checks remain enabled; no throughput or cache-pressure workload is added.
