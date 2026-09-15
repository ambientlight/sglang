"""Deployment identity/auth controls must retain native V4.1 engine behavior."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "candidate_boot", Path(__file__).with_name("boot.py")
)
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


class LaunchPolicyTest(unittest.TestCase):
    def launch(self, overrides):
        process = Mock(stdout=[])
        process.poll.side_effect = [None, 0]
        process.wait.return_value = 0
        env = {
            "OFFLOAD_MODE": "nvme",
            "CONTEXT_LENGTH": "409600",
            "CHUNKED_PREFILL_SIZE": "2048",
            "MAX_RUNNING_REQUESTS": "8",
            "MEMORY_FRACTION": "0.85",
            "MAX_TOTAL_TOKENS": "",
            "SERVED_MODEL_NAME": "deepseek-v4.1-flash",
            "REQUIRE_API_KEY": "1",
            "SERVER_PORT": "8010",
            **overrides,
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(boot.os.environ, env),
            patch.object(boot, "STATE", Path(temp)),
            patch.object(boot, "PORT", int(env["SERVER_PORT"])),
            patch.object(
                boot.subprocess,
                "check_output",
                return_value="NVIDIA RTX PRO 6000, 97887\n" * 4,
            ),
            patch.object(boot.subprocess, "Popen", return_value=process) as start,
            patch.object(boot, "key", return_value="fixture-key"),
            patch.object(boot, "request"),
            patch.object(boot, "smoke"),
            patch.object(boot.signal, "signal"),
        ):
            self.assertEqual(boot.serve(), 0)
            self.command = start.call_args.args[0]
            self.manifest = json.loads((Path(temp) / "launch.json").read_text())
            return self.manifest["args"]

    def test_defaults_and_hardware_profile(self):
        args = self.launch({})
        self.assertEqual(
            json.loads(args[args.index("--preferred-sampling-params") + 1]),
            {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 32768},
        )
        for flag, value in (
            ("--tp", "4"),
            ("--ep-size", "4"),
            ("--context-length", "409600"),
            ("--chunked-prefill-size", "2048"),
            ("--speculative-algorithm", "DSPARK"),
            ("--speculative-dspark-block-size", "5"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--enable-decoder-swa-bounded-replay", args)
        self.assertIn("--enable-metrics", args)
        self.assertNotIn("fixture-key", json.dumps(args))
        self.assertIn("--api-key", self.command)
        self.assertTrue(self.manifest["require_api_key"])

    def test_legacy_endpoint_keeps_native_v41_parsers(self):
        args = self.launch({
            "SERVED_MODEL_NAME": "deepseek-v4-flash",
            "CONTEXT_LENGTH": "524288",
            "SERVER_PORT": "8000",
            "REQUIRE_API_KEY": "0",
        })
        for flag, expected in (
            ("--served-model-name", "deepseek-v4-flash"),
            ("--context-length", "524288"),
            ("--port", "8000"),
            ("--tool-call-parser", "deepseekv41"),
            ("--reasoning-parser", "deepseek-v41"),
            ("--max-running-requests", "8"),
            ("--mem-fraction-static", "0.85"),
            ("--chunked-prefill-size", "2048"),
        ):
            self.assertEqual(args[args.index(flag) + 1], expected)
        self.assertNotIn("--api-key", self.command)
        self.assertNotIn("fixture-key", self.command)
        self.assertFalse(self.manifest["require_api_key"])

    def test_invalid_auth_mode_is_not_silently_disabled(self):
        with patch.dict(boot.os.environ, {"REQUIRE_API_KEY": "false"}):
            with self.assertRaises(AssertionError):
                boot.require_api_key()

    def test_empty_served_alias_is_rejected(self):
        with patch.dict(boot.os.environ, {"SERVED_MODEL_NAME": " "}):
            with self.assertRaises(AssertionError):
                boot.served_model_name()

    def test_health_client_omits_auth_in_legacy_mode(self):
        response = Mock()
        response.read.return_value = b'{}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with (
            patch.dict(boot.os.environ, {"REQUIRE_API_KEY": "0"}),
            patch.object(boot, "PORT", 8000),
            patch.object(boot, "key") as key,
            patch.object(boot.urllib.request, "urlopen", return_value=context) as call,
        ):
            boot.request('/health')
        req = call.call_args.args[0]
        self.assertEqual(req.full_url, 'http://127.0.0.1:8000/health')
        self.assertFalse(req.has_header('Authorization'))
        key.assert_not_called()

    def test_all_startup_smokes_use_the_served_alias(self):
        call = {'id':'call_fixture','type':'function','function':{
            'name':'lookup_fixture','arguments':'{"key":"alpha"}'}}
        outputs = [
            {'choices':[{'message':{'content':'42'},'finish_reason':'stop'}]},
            {'choices':[{'message':{'content':'{"answer":42}'}}]},
            {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[call]}}]},
            {'choices':[{'message':{'content':'42'}}]},
            {'choices':[{'message':{'content':'red circle left of blue square'}}],
             'usage':{'prompt_tokens_details':{'image_tokens':1024}}},
        ]
        with (
            patch.dict(boot.os.environ, {'SERVED_MODEL_NAME':'deepseek-v4-flash'}),
            patch.object(boot, 'request', side_effect=outputs) as request,
            patch.object(boot, 'save'),
        ):
            boot.smoke()
        self.assertEqual(len(request.call_args_list), 5)
        for invocation in request.call_args_list:
            self.assertEqual(invocation.args[1]['model'], 'deepseek-v4-flash')

    def test_operator_can_override_missing_field_policy(self):
        args = self.launch(
            {
                "DSV41_PREFERRED_SAMPLING_PARAMS": '{"temperature":0.7,"max_new_tokens":4096}'
            }
        )
        self.assertEqual(
            json.loads(args[args.index("--preferred-sampling-params") + 1]),
            {"temperature": 0.7, "max_new_tokens": 4096},
        )


if __name__ == "__main__":
    unittest.main()
