"""The API launch profile may add sampling/metrics, not change hardware flags."""

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
            **overrides,
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(boot.os.environ, env),
            patch.object(boot, "STATE", Path(temp)),
            patch.object(
                boot.subprocess,
                "check_output",
                return_value="NVIDIA RTX PRO 6000, 97887\n" * 4,
            ),
            patch.object(boot.subprocess, "Popen", return_value=process),
            patch.object(boot, "key", return_value="fixture-key"),
            patch.object(boot, "request"),
            patch.object(boot, "smoke"),
            patch.object(boot.signal, "signal"),
        ):
            self.assertEqual(boot.serve(), 0)
            return json.loads((Path(temp) / "launch.json").read_text())["args"]

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
