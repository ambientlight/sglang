"""Pure-Python V4.1 encoder contract checks, runnable without CUDA dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "dsv41_encoder_under_test",
    ROOT / "python/sglang/srt/entrypoints/openai/encoding_dsv41.py",
)
encoder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(encoder)


class EncoderOverlayTest(unittest.TestCase):
    def test_native_inline_system_chronology(self):
        messages = [
            {"role": "system", "content": "initial instruction"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "system", "content": "later instruction"},
            {"role": "user", "content": "second question"},
        ]
        before = copy.deepcopy(messages)
        prompt = encoder.encode_messages(messages, thinking_mode="chat")
        self.assertEqual(messages, before)
        self.assertIn("<｜System｜>later instruction", prompt)
        self.assertNotIn("<｜latest_reminder｜>", prompt)
        self.assertLess(prompt.index("first answer"), prompt.index("later instruction"))
        self.assertLess(
            prompt.index("later instruction"), prompt.index("second question")
        )

    def test_tool_error_is_preserved_without_changing_success(self):
        for is_error in (False, True):
            with self.subTest(is_error=is_error):
                messages = [
                    {"role": "system", "content": ""},
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "fixture result",
                        "is_error": is_error,
                    },
                ]
                before = copy.deepcopy(messages)
                prompt = encoder.encode_messages(messages, thinking_mode="chat")
                self.assertEqual(messages, before)
                self.assertEqual("Tool execution failed." in prompt, is_error)
                self.assertIn("fixture result</tool_result>", prompt)
                if not is_error:
                    self.assertIn("<tool_result>fixture result</tool_result>", prompt)

    def test_existing_tool_result_blocks_keep_error_flag(self):
        prompt = encoder.encode_messages(
            [
                {
                    "role": "user",
                    "content_blocks": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "c",
                            "is_error": True,
                            "content": [{"type": "text", "text": "permission denied"}],
                        },
                        {"type": "text", "text": "try another method"},
                    ],
                },
            ],
            thinking_mode="chat",
        )
        self.assertIn("<tool_result>Tool execution failed.", prompt)
        self.assertIn("permission denied</tool_result>", prompt)
        self.assertLess(
            prompt.index("permission denied"), prompt.index("try another method")
        )

    def test_baseline_reasoning_mapping_stays_pinned(self):
        for effort, budget in (
            ("low", 25),
            ("high", 50),
            ("xhigh", 75),
            ("max", 100),
            (75, 75),
        ):
            with self.subTest(effort=effort):
                prompt = encoder.encode_messages(
                    [{"role": "user", "content": "question"}],
                    thinking_mode="thinking",
                    reasoning_effort=effort,
                )
                self.assertIn(f"Reasoning Effort: {budget} ", prompt)
        prompt = encoder.encode_messages(
            [{"role": "user", "content": "question"}], thinking_mode="chat"
        )
        self.assertNotIn("Reasoning Effort:", prompt)

    def test_tool_history_uses_spaced_markers_and_preserves_json_values(self):
        args = {"value": "  meaningful whitespace\n", "count": 3}
        encoded = encoder.encode_arguments_to_dsml({"arguments": json.dumps(args)})
        self.assertIn(
            '<｜DSML｜ parameter name="value" string="true">  meaningful whitespace\n</｜DSML｜ parameter>',
            encoded,
        )
        self.assertIn(
            '<｜DSML｜ parameter name="count" string="false">3</｜DSML｜ parameter>',
            encoded,
        )


if __name__ == "__main__":
    unittest.main()
