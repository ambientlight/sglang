"""CPU-only crash-path regressions; never attach to a real scheduler or CUDA.

Run directly with Python + psutil. Real subprocess tests use stand-ins for
py-spy, and FIFO readers stand in for CUDA/PyTorch dump handlers.
"""

import ast
import asyncio
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import psutil

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


environ = load("crash_test_environ", SRT / "environ.py")
with patch.dict(sys.modules, {"sglang.srt.environ": environ}):
    diag = load("crash_test_diagnostics", SRT / "utils/cudacore_pyspy_dump_utils.py")
with patch.dict(sys.modules, {"sglang.srt.utils.cudacore_pyspy_dump_utils": diag}):
    watchdog = load("crash_test_watchdog", SRT / "utils/watchdog.py")


def manager_node(name, class_name=None):
    tree = ast.parse((SRT / "managers/tokenizer_manager.py").read_text())
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    node = next(
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    scope = dict(os=os, socket=socket, logger=Mock(), envs=environ.envs)
    module = ast.Module(body=[node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), "<real-crash-handler>", "exec"),
        scope,
    )
    return scope[name], scope


class DiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "SGLANG_PYSPY_DUMP_TIMEOUT_SECS": "0.2",
                "SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS": "1",
                "SGLANG_PYSPY_DUMP_NATIVE": "1",
                "SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS": "0.3",
                "SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS": "0",
                "SGLANG_NCCL_DUMP_BEFORE_CRASH_WAIT_SECS": "0",
            },
            clear=True,
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        quiet = patch.object(diag, "logger")
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_invalid_timeout_values_are_finite_and_zero_can_disable(self):
        for value in (float("nan"), float("inf"), -1):
            self.assertEqual(diag._timeout(value, 5), 5)
        self.assertEqual(diag._timeout(0, 5), 0)

    def test_python_stacks_for_all_ranks_precede_native_stacks_without_shell(self):
        procs = [NS(pid=101), NS(pid=102)]
        with patch.object(diag, "_run_pyspy") as run:
            diag.pyspy_dump_schedulers(True, processes=procs)
        commands = [c.args[0] for c in run.call_args_list]
        self.assertEqual(
            commands,
            [
                ["py-spy", "dump", "--nonblocking", "--pid", "101"],
                ["py-spy", "dump", "--nonblocking", "--pid", "102"],
                ["py-spy", "dump", "--native", "--pid", "101"],
                ["py-spy", "dump", "--native", "--pid", "102"],
            ],
        )

    def test_native_stacks_can_be_disabled(self):
        with (
            patch.dict(os.environ, {"SGLANG_PYSPY_DUMP_NATIVE": "0"}),
            patch.object(diag, "_run_pyspy") as run,
        ):
            diag.pyspy_dump_schedulers(True, processes=[NS(pid=1)])
        self.assertEqual(run.call_count, 1)
        self.assertIn("--nonblocking", run.call_args.args[0])

    def test_no_schedulers_and_expired_budget_do_not_spawn(self):
        with patch.object(diag, "_run_pyspy") as run:
            diag.pyspy_dump_schedulers(True, processes=[])
            diag.pyspy_dump_schedulers(
                True, processes=[NS(pid=1)], deadline=time.monotonic() - 1
            )
        run.assert_not_called()

    def test_total_budget_stops_before_later_rank_or_native_attempt(self):
        def slow(*args):
            time.sleep(0.08)

        with (
            patch.dict(os.environ, {"SGLANG_PYSPY_DUMP_TOTAL_TIMEOUT_SECS": "0.05"}),
            patch.object(diag, "_run_pyspy", side_effect=slow) as run,
        ):
            diag.pyspy_dump_schedulers(True, processes=[NS(pid=1), NS(pid=2)])
        self.assertEqual(run.call_count, 1)

    def test_missing_binary_or_permission_error_does_not_escape(self):
        for error in (FileNotFoundError(), PermissionError()):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(diag.subprocess, "Popen", side_effect=error),
            ):
                diag._run_pyspy(["not-a-real-py-spy"], 0.1, time.monotonic() + 1)

    def test_success_and_nonzero_exit_do_not_throw(self):
        for code in (0, 2):
            with self.subTest(code=code):
                diag._run_pyspy(
                    [
                        sys.executable,
                        "-c",
                        f"print('stack fixture'); raise SystemExit({code})",
                    ],
                    2,
                    time.monotonic() + 3,
                )

    def test_real_hung_collector_is_killed_and_reaped(self):
        real_popen = subprocess.Popen
        children = []

        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        start = time.monotonic()
        with patch.object(diag.subprocess, "Popen", side_effect=spawn):
            diag._run_pyspy(
                [sys.executable, "-c", "import time; time.sleep(60)"], 0.15, start + 2
            )
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(children[0].returncode, -signal.SIGKILL)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_descendant_does_not_keep_output_open_or_survive_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "child.pid"
            code = (
                "import subprocess,sys,time,pathlib; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"pathlib.Path({str(receipt)!r}).write_text(str(p.pid)); "
                "time.sleep(60)"
            )
            start = time.monotonic()
            diag._run_pyspy([sys.executable, "-c", code], 0.5, start + 2)
            self.assertLess(time.monotonic() - start, 1.5)
            child = int(receipt.read_text())
            try:
                proc = psutil.Process(child)
                for _ in range(30):
                    if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                        break
                    time.sleep(0.01)
                self.assertTrue(
                    not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE
                )
            except psutil.NoSuchProcess:
                pass

    def test_fifo_trigger_has_reader_and_never_touches_regular_file_or_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            fifo = Path(tmp) / "core.pipe"
            os.mkfifo(fifo)
            reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            try:
                self.assertTrue(diag._trigger_dump_pipe(fifo, "test"))
                self.assertEqual(os.read(reader, 1), b"1")
                link = Path(tmp) / "link.pipe"
                link.symlink_to(fifo)
                self.assertFalse(diag._trigger_dump_pipe(link, "test"))
            finally:
                os.close(reader)
            start = time.monotonic()
            self.assertFalse(diag._trigger_dump_pipe(fifo, "test"))
            self.assertLess(time.monotonic() - start, 0.5)
            regular = Path(tmp) / "regular"
            regular.write_text("preserve me")
            self.assertFalse(diag._trigger_dump_pipe(regular, "test"))
            self.assertEqual(regular.read_text(), "preserve me")
            self.assertFalse(diag._trigger_dump_pipe(Path(tmp) / "absent", "test"))

    def test_nccl_selects_only_current_prefix_numeric_rank_pipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = [
                "trace_0.pipe",
                "trace_1.pipe",
                "trace_other.pipe",
                "old_trace_0.pipe",
                "trace_0.txt",
            ]
            for name in names:
                (Path(tmp) / name).touch()
            with (
                patch.dict(
                    os.environ,
                    {"TORCH_NCCL_DEBUG_INFO_PIPE_FILE": str(Path(tmp) / "trace_")},
                ),
                patch.object(diag, "_trigger_dump_pipe", return_value=True) as trigger,
            ):
                self.assertEqual(diag.trigger_torch_nccl_dump(), 2)
                self.assertEqual(
                    [c.args[0].name for c in trigger.call_args_list], names[:2]
                )
                trigger.reset_mock()
                self.assertEqual(
                    diag.trigger_torch_nccl_dump(deadline=time.monotonic() - 1), 0
                )
                trigger.assert_not_called()

    def test_no_nccl_prefix_or_missing_directory_is_safe(self):
        self.assertEqual(diag.trigger_torch_nccl_dump(), 0)
        with patch.dict(
            os.environ, {"TORCH_NCCL_DEBUG_INFO_PIPE_FILE": "/does-not-exist/trace_"}
        ):
            self.assertEqual(diag.trigger_torch_nccl_dump(), 0)

    def test_cuda_disabled_or_no_live_process_does_not_trigger(self):
        with patch.object(diag, "_trigger_dump_pipe") as trigger:
            self.assertEqual(
                diag.trigger_cuda_user_coredump(True, processes=[NS(pid=1)]), 0
            )
            with patch.dict(os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}):
                self.assertEqual(diag.trigger_cuda_user_coredump(True, processes=[]), 0)
        trigger.assert_not_called()

    def test_cuda_uses_snapshot_and_continues_past_dead_rank(self):
        with (
            patch.dict(
                os.environ,
                {
                    "CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1",
                    "CUDA_COREDUMP_PIPE": "/tmp/test-core.%p",
                },
            ),
            patch.object(
                diag, "_trigger_dump_pipe", side_effect=[False, True]
            ) as trigger,
        ):
            self.assertEqual(
                diag.trigger_cuda_user_coredump(
                    True, processes=[NS(pid=10), NS(pid=11)]
                ),
                1,
            )
            self.assertEqual(
                [c.args[0] for c in trigger.call_args_list],
                [Path("/tmp/test-core.10"), Path("/tmp/test-core.11")],
            )

    def test_device_dumps_precede_stack_collection(self):
        stages = []
        with (
            patch.object(
                diag,
                "trigger_torch_nccl_dump",
                side_effect=lambda **k: stages.append("nccl") or 1,
            ),
            patch.object(diag, "collect_scheduler_processes", return_value=[NS(pid=1)]),
            patch.object(
                diag,
                "trigger_cuda_user_coredump",
                side_effect=lambda **k: stages.append("cuda") or 1,
            ),
            patch.object(
                diag,
                "pyspy_dump_schedulers",
                side_effect=lambda **k: stages.append("pyspy"),
            ),
        ):
            diag.dump_crash_diagnostics(pyspy=True, cuda=True, nccl=True)
        self.assertEqual(stages, ["nccl", "cuda", "pyspy"])

    def test_failed_collector_does_not_skip_remaining_collectors(self):
        with (
            patch.object(
                diag, "trigger_torch_nccl_dump", side_effect=RuntimeError("fixture")
            ),
            patch.object(diag, "collect_scheduler_processes", return_value=[]),
            patch.object(diag, "trigger_cuda_user_coredump", return_value=0) as cuda,
            patch.object(diag, "pyspy_dump_schedulers") as pyspy,
        ):
            diag.dump_crash_diagnostics(pyspy=True, cuda=True, nccl=True)
        cuda.assert_called_once()
        pyspy.assert_called_once()

    def test_hung_stage_cannot_hold_caller_past_total_budget(self):
        released = threading.Event()
        ended = threading.Event()

        def stuck(**kwargs):
            released.wait(3)
            ended.set()
            return 0

        try:
            with (
                patch.dict(
                    os.environ, {"SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS": "0.1"}
                ),
                patch.object(diag, "trigger_torch_nccl_dump", side_effect=stuck),
            ):
                start = time.monotonic()
                diag.dump_crash_diagnostics(pyspy=True, cuda=True, nccl=True)
                self.assertLess(time.monotonic() - start, 0.5)
        finally:
            released.set()
            self.assertTrue(ended.wait(1))

    def test_wait_is_shared_not_added_after_stacks(self):
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_CRASH_DIAGNOSTICS_TIMEOUT_SECS": "0.25",
                    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS": "60",
                },
            ),
            patch.object(diag, "collect_scheduler_processes", return_value=[]),
            patch.object(diag, "trigger_cuda_user_coredump", return_value=1),
            patch.object(
                diag, "pyspy_dump_schedulers", side_effect=lambda **k: time.sleep(0.15)
            ),
        ):
            start = time.monotonic()
            diag.dump_crash_diagnostics(pyspy=True, cuda=True, nccl=False)
            elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.20)
        self.assertLess(elapsed, 0.5)

    def test_no_successful_trigger_does_not_wait_for_dumps(self):
        with (
            patch.object(diag, "collect_scheduler_processes", return_value=[]),
            patch.object(diag, "trigger_torch_nccl_dump", return_value=0),
            patch.object(diag, "trigger_cuda_user_coredump", return_value=0),
        ):
            start = time.monotonic()
            diag.dump_crash_diagnostics(pyspy=False, cuda=True, nccl=True)
        self.assertLess(time.monotonic() - start, 0.2)

    def test_hard_watchdog_notifies_parent_before_optional_debug_callback(self):
        raw = watchdog.WatchdogRaw.__new__(watchdog.WatchdogRaw)
        raw.debug_name = "test"
        raw.get_counter = lambda: 0
        raw.is_active = lambda: True
        raw.watchdog_timeout = 1
        raw.soft = False
        raw.dump_info = Mock(side_effect=AssertionError("must not delay notification"))
        raw.parent_process = Mock()
        with (
            patch.object(watchdog.time, "perf_counter", side_effect=[0, 2]),
            patch.object(watchdog, "pyspy_dump_schedulers") as pyspy,
        ):
            raw._watchdog_once()
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        raw.dump_info.assert_not_called()
        pyspy.assert_not_called()

    def test_soft_watchdog_remains_nonfatal_even_if_debug_callback_fails(self):
        raw = watchdog.WatchdogRaw.__new__(watchdog.WatchdogRaw)
        raw.debug_name = "test"
        raw.get_counter = lambda: 0
        raw.is_active = lambda: True
        raw.watchdog_timeout = 1
        raw.soft = True
        raw.dump_info = Mock(side_effect=RuntimeError("fixture"))
        raw.parent_process = Mock()
        with (
            patch.object(watchdog.time, "perf_counter", side_effect=[0, 2]),
            patch.object(watchdog, "pyspy_dump_schedulers") as pyspy,
        ):
            raw._watchdog_once()
        raw.parent_process.send_signal.assert_not_called()
        pyspy.assert_called_once()

    def test_real_manager_method_runs_diagnostics_once_without_request_dump_folder(
        self,
    ):
        method, scope = manager_node("dump_requests_before_crash", "TokenizerManager")
        scope["dump_crash_diagnostics"] = Mock()
        manager = NS(crash_dump_folder=None, crash_dump_performed=False)
        method(manager)
        method(manager)
        scope["dump_crash_diagnostics"].assert_called_once_with(
            pyspy=True, cuda=True, nccl=False
        )

    def test_nccl_only_diagnostics_are_not_skipped(self):
        method, scope = manager_node("dump_requests_before_crash", "TokenizerManager")
        scope["dump_crash_diagnostics"] = Mock()
        with patch.dict(
            os.environ,
            {
                "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
                "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
                "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": "/tmp/fixture_",
            },
        ):
            method(NS(crash_dump_folder=None, crash_dump_performed=False))
        scope["dump_crash_diagnostics"].assert_called_once_with(
            pyspy=False, cuda=False, nccl=True
        )

    def test_sigquit_cleanup_survives_dump_failure(self):
        method, scope = manager_node("running_phase_sigquit_handler", "SignalHandler")
        scope["kill_process_tree"] = Mock()
        manager = NS(
            _subprocess_watchdog=None,
            dump_requests_before_crash=Mock(side_effect=RuntimeError("fixture")),
        )
        with self.assertRaises(RuntimeError):
            method(NS(tokenizer_manager=manager))
        scope["kill_process_tree"].assert_called_once_with(os.getpid())

    def test_sigquit_cleanup_survives_subprocess_watchdog_stop_failure(self):
        method, scope = manager_node("running_phase_sigquit_handler", "SignalHandler")
        scope["kill_process_tree"] = Mock()
        manager = NS(
            _subprocess_watchdog=NS(stop=Mock(side_effect=RuntimeError("fixture")))
        )
        with self.assertRaises(RuntimeError):
            method(NS(tokenizer_manager=manager))
        scope["kill_process_tree"].assert_called_once()

    def test_async_exception_cleanup_survives_dump_failure(self):
        method, scope = manager_node("print_exception_wrapper")

        class Manager:
            async def broken(self):
                raise ValueError("original")

            def dump_requests_before_crash(self):
                raise RuntimeError("dump failed")

        scope.update(
            TokenizerManager=Manager,
            get_exception_traceback=lambda: "fixture",
            kill_process_tree=Mock(),
        )
        with self.assertRaises(RuntimeError):
            asyncio.run(method(Manager().broken))
        scope["kill_process_tree"].assert_called_once_with(
            os.getpid(), include_parent=True
        )


if __name__ == "__main__":
    unittest.main()
