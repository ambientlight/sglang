# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Best-effort crash diagnostics with bounded external collectors.

No CUDA/torch imports: this code must still run after a device has failed.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import signal
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import psutil

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _timeout(value: float, default: float) -> float:
    if not math.isfinite(value) or value < 0:
        logger.warning("Invalid diagnostic timeout %r; using %.1fs", value, default)
        return default
    return value


def _resolve_cuda_coredump_pipe_path(proc: psutil.Process) -> Path:
    pipe_template = os.environ.get("CUDA_COREDUMP_PIPE")
    if pipe_template is None:
        pipe_path = f"corepipe.cuda.{platform.node()}.{proc.pid}"
    else:
        pipe_path = (
            pipe_template.replace("%h", platform.node())
            .replace("%p", str(proc.pid))
            .replace("%t", str(int(time.time())))
        )

    path = Path(pipe_path)
    if path.is_absolute():
        return path

    try:
        return Path(proc.cwd()) / path
    except (psutil.Error, OSError):
        return Path.cwd() / path


def _is_sglang_scheduler_process(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith("sglang::scheduler")


def collect_scheduler_processes() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_scheduler_process(proc)
    ]


def _kill_collector(proc: subprocess.Popen) -> None:
    """Kill only the process group created for this diagnostic command."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except OSError:
        logger.exception("Unable to stop diagnostic collector PID %s", proc.pid)


def _run_pyspy(cmd: List[str], timeout: float, deadline: float) -> None:
    # A temporary file avoids communicate() waiting on stdout/stderr inherited
    # by a collector's descendants. Do not use Popen's context manager: its
    # __exit__ waits without a timeout, including for uninterruptible children.
    proc = None
    with tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            proc.wait(timeout=min(timeout, max(0.0, deadline - time.monotonic())))
            output.seek(0)
            text = output.read(256 * 1024).decode("utf-8", errors="replace")
            logger.error(
                "Pyspy dump (%s), exit=%s, output capped at 256 KiB:\n%s",
                " ".join(cmd),
                proc.returncode,
                text,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "Pyspy timed out after at most %.2fs (%s)", timeout, " ".join(cmd)
            )
        except OSError:
            logger.exception("Unable to run diagnostic command %s", " ".join(cmd))
        finally:
            if proc is not None:
                # Clean up descendants even when the collector itself exited.
                _kill_collector(proc)
                try:
                    proc.wait(timeout=min(0.2, max(0.0, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    logger.error(
                        "Collector PID %s did not reap within the budget", proc.pid
                    )


def pyspy_dump_schedulers(scheduler_only=False, *, processes=None, deadline=None):
    """Python stacks for every rank first, then optional native stacks.

    Both individual commands and the entire sequence have a finite budget.
    Nonblocking Python collection avoids pausing ranks while GPU dumps run.
    Native collection is bounded but can still briefly suspend its target.
    """
    now = time.monotonic()
    total = _timeout(envs.SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS.get(), 20.0)
    deadline = min(deadline, now + total) if deadline is not None else now + total
    per_attempt = _timeout(envs.SGLANG_PYSPY_DUMP_TIMEOUT_SECS.get(), 5.0)
    if per_attempt == 0 or now >= deadline:
        return
    if scheduler_only:
        procs = processes if processes is not None else collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for py-spy dump.")
            return
        pids = [proc.pid for proc in procs]
    else:
        pids = [os.getpid()]
    flags = ["--nonblocking"]
    if envs.SGLANG_PYSPY_DUMP_NATIVE.get():
        flags.append("--native")
    for flag in flags:
        for pid in pids:
            if time.monotonic() >= deadline:
                logger.error(
                    "Pyspy total diagnostic budget exhausted; skipping remaining stacks"
                )
                return
            _run_pyspy(
                ["py-spy", "dump", flag, "--pid", str(pid)], per_attempt, deadline
            )


def _trigger_dump_pipe(path: Path, kind: str) -> bool:
    """Never block on an unread FIFO or write to a regular file/symlink."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            if not stat.S_ISFIFO(os.fstat(fd).st_mode):
                logger.error("Refusing non-FIFO %s dump target: %s", kind, path)
                return False
            os.write(fd, b"1")
        finally:
            os.close(fd)
        logger.error("Triggered %s dump via %s", kind, path)
        return True
    except OSError as exc:
        # ENOENT: process exited; ENXIO: no reader; EAGAIN: already full.
        logger.error("Unable to trigger %s dump via %s: %s", kind, path, exc)
        return False


def trigger_torch_nccl_dump(*, deadline=None) -> int:
    """Trigger only rank FIFOs under this run's explicitly configured prefix."""
    prefix = os.environ.get("TORCH_NCCL_DEBUG_INFO_PIPE_FILE")
    if not prefix:
        return 0
    base = Path(prefix)
    count = 0
    try:
        for path in sorted(base.parent.iterdir()):
            if deadline is not None and time.monotonic() >= deadline:
                break
            suffix = path.name.removeprefix(base.name)
            if (
                path.name.startswith(base.name)
                and suffix.endswith(".pipe")
                and suffix[:-5].isascii()
                and suffix[:-5].isdigit()
            ):
                count += _trigger_dump_pipe(path, "PyTorch NCCL flight recorder")
    except OSError:
        logger.exception("Unable to list NCCL dump pipes under %s", base.parent)
    return count


def trigger_cuda_user_coredump(scheduler_only=False, *, processes=None, deadline=None):
    """Trigger CUDA user-induced GPU core dumps by writing to coredump pipes."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        logger.error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )
        return 0

    if scheduler_only:
        procs = processes if processes is not None else collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for CUDA coredump.")
            return 0
    else:
        procs = [psutil.Process()]

    count = 0
    for proc in procs:
        if deadline is not None and time.monotonic() >= deadline:
            break
        count += _trigger_dump_pipe(
            _resolve_cuda_coredump_pipe_path(proc), "CUDA user core"
        )
    return count


def dump_crash_diagnostics(*, pyspy: bool, cuda: bool, nccl: bool) -> None:
    """Trigger device evidence first; never hold cleanup behind a collector.

    A daemon worker and bounded join also cover stuck process discovery or
    filesystem access. Every subprocess has its own timeout and kill group.
    This cannot recover from a kernel-wide stall or a thread holding the GIL.
    """
    budget = _timeout(envs.SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS.get(), 60.0)
    if budget == 0:
        return
    deadline = time.monotonic() + budget
    # Leave a small margin for the worker to return after its final wait.
    work_deadline = deadline - min(0.05, budget / 10)
    cancelled = threading.Event()

    def attempt(fn, **kwargs):
        try:
            return fn(**kwargs)
        except Exception:
            logger.exception(
                "Crash diagnostic stage failed; continuing other collectors"
            )
            return None

    def collect():
        try:
            wait_until = time.monotonic()
            # No initial sleep or ptrace before requesting device/collective
            # evidence. Use this run's rank pipes even if a scheduler exited.
            if nccl and attempt(trigger_torch_nccl_dump, deadline=work_deadline):
                wait_until = time.monotonic() + _timeout(
                    envs.SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS.get(), 10.0
                )
            if cancelled.is_set() or time.monotonic() >= work_deadline:
                return
            procs = attempt(collect_scheduler_processes) or []
            if cuda and attempt(
                trigger_cuda_user_coredump,
                scheduler_only=True,
                processes=procs,
                deadline=work_deadline,
            ):
                wait_until = max(
                    wait_until,
                    time.monotonic()
                    + _timeout(
                        envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS.get(), 60.0
                    ),
                )
            if pyspy and not cancelled.is_set():
                attempt(
                    pyspy_dump_schedulers,
                    scheduler_only=True,
                    processes=procs,
                    deadline=work_deadline,
                )
            # Stack collection consumes the same window as GPU dump waiting;
            # do not add another fixed 60 seconds after stacks finish.
            remaining = min(wait_until, work_deadline) - time.monotonic()
            if remaining > 0:
                logger.info("Waiting up to %.2fs for requested crash dumps", remaining)
                cancelled.wait(remaining)
        except Exception:
            logger.exception("Crash diagnostic collection failed; continuing cleanup")

    worker = threading.Thread(target=collect, name="crash-diagnostics", daemon=True)
    worker.start()
    worker.join(timeout=max(0.0, deadline - time.monotonic()))
    if worker.is_alive():
        cancelled.set()
        logger.error(
            "Crash diagnostics exceeded %.2fs; proceeding with cleanup", budget
        )
