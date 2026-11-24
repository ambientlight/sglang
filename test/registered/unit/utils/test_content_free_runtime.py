"""CPU regressions for the native content-free policy hooks.

Run with the normal Python SGLang serving dependencies. Heavy model modules can
be excluded from these checks; the request logger's real methods are exercised.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load_class(path, name):
    source = ast.parse((SRT / path).read_text())
    node = next(
        n for n in source.body if isinstance(n, ast.ClassDef) and n.name == name
    )
    ns = {"os": os, "create_log_targets": Mock(side_effect=AssertionError)}
    # Resolve annotations lazily just like the source modules.
    tree = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node], []
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns[name]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, SRT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContentFreeRuntime(unittest.TestCase):
    def test_request_content_is_never_read_or_exported(self):
        logger_cls = load_class("observability/request_logger.py", "RequestLogger")
        exporters = load_class(
            "observability/request_metrics_exporter.py", "RequestMetricsExporterManager"
        )
        with patch.dict(os.environ, CONTENT_FREE_LOGGING="1"):
            logger = logger_cls(True, 3, "json", ["file"])
            logger.configure(True, 3, "json", ["file"])
            request = Mock()
            logger.log_received_request(request, request, request)
            logger.log_openai_received_request(request, request)
            logger.log_finished_request(request, request, request)
            self.assertFalse(logger.log_requests)
            self.assertEqual(logger.targets, [])
            self.assertEqual(request.mock_calls, [])
            with patch.object(
                exporters, "_create_exporters", side_effect=AssertionError
            ):
                manager = exporters(Mock())
            self.assertFalse(manager.exporter_enabled())

    def test_http_metrics_use_fixed_method_and_route_template(self):
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.constants": SimpleNamespace(
                    ROUTING_KEY_HEADER="x-test-route"
                )
            },
        ):
            metrics = load_module(
                "policy_http_metrics", "observability/http_request_metrics.py"
            )
        middleware = load_module("policy_middleware", "utils/http_middleware_patch.py")
        app = FastAPI()
        with (
            patch.dict(
                sys.modules, {"sglang.srt.utils.http_middleware_patch": middleware}
            ),
            patch.dict(os.environ, CONTENT_FREE_LOGGING="1"),
        ):
            metrics.add_http_request_metrics_middleware(app)

        @app.get("/sample/{record}")
        def sample(record: str):
            return {"ok": True}

        metrics._http_request_metrics = Mock()
        with TestClient(app) as client:
            self.assertEqual(
                client.get(
                    "/sample/private-value", headers={"x-test-route": "private-header"}
                ).status_code,
                200,
            )
            self.assertEqual(
                client.request("PRIVATE-METHOD", "/private-path").status_code, 404
            )
        requests = metrics._http_request_metrics.requests_total.labels.call_args_list
        self.assertEqual(
            [c.kwargs for c in requests],
            [
                dict(endpoint="/sample/{record}", method="GET"),
                dict(endpoint="unmatched", method="OTHER"),
            ],
        )
        self.assertNotIn(
            "private", str(metrics._http_request_metrics.mock_calls).lower()
        )

    def test_diagnostic_raw_collector_and_cuda_trigger_disabled(self):
        env = SimpleNamespace(envs=Mock())
        with patch.dict(sys.modules, {"sglang.srt.environ": env}):
            diag = load_module("policy_core", "utils/cudacore_pyspy_dump_utils.py")
        policy = SimpleNamespace(collect_pyspy=Mock())
        with (
            patch.dict(
                os.environ,
                CONTENT_FREE_LOGGING="1",
                CUDA_ENABLE_USER_TRIGGERED_COREDUMP="1",
            ),
            patch.dict(sys.modules, {"content_free_logging": policy}),
            patch.object(diag.tempfile, "TemporaryFile", side_effect=AssertionError),
            patch.object(diag, "_trigger_dump_pipe", side_effect=AssertionError),
        ):
            cmd = ["py-spy", "dump", "--nonblocking", "--pid", "111"]
            diag._run_pyspy(cmd, 1, 3)
            policy.collect_pyspy.assert_called_once_with(cmd, 1, 3)
            self.assertEqual(diag.trigger_cuda_user_coredump(), 0)


if __name__ == "__main__":
    unittest.main()
