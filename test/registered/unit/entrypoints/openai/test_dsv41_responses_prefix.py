"""Responses chronology through the real V4.1 request/encoder path.

Unit tests use injective UTF-8 byte IDs instead of a constant tokenizer mock.
Real checkpoint-tokenizer replay of private captures is a separate CPU check;
no private prompts or model files are required by this portable suite.
"""

import asyncio
import copy
import unittest
from types import SimpleNamespace

from test_dsv41_api_contract import SCHEMA, serving
from utils import make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest


TOOLS = [{"type": "function", "name": "Bash", "parameters": SCHEMA}]
UPDATES = (
    "<skills_instructions>Use the project skill.</skills_instructions>",
    "<turn_aborted>The previous turn was interrupted.</turn_aborted>",
    "Use the revised acceptance criteria for the next step.",
)


def handler(multimodal=True):
    result = serving()
    result.default_chat_template_kwargs = {}
    result.template_manager.chat_template_name = None
    result.tokenizer_manager.model_config.is_multimodal = multimodal
    result.tokenizer_manager.model_config.hf_config.image_token_id = 128100
    tokenizer = result.tokenizer_manager.tokenizer
    tokenizer.convert_ids_to_tokens.return_value = "<image>"
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text.encode("utf-8"))
    return result


def request(items, *, instructions="Stable instructions.", tools=True, thinking=True):
    return ResponsesRequest(
        model="deepseek-v4-flash",
        instructions=instructions,
        input=copy.deepcopy(items),
        tools=copy.deepcopy(TOOLS) if tools else [],
        tool_choice="auto" if tools else "none",
        chat_template_kwargs={"thinking": thinking},
        store=False,
    )


def encode(result, req, previous=None):
    converted = asyncio.run(
        result._make_request(req, previous, result.tokenizer_manager.tokenizer)
    )
    text = result.tokenizer_manager.tokenizer.encode.call_args.args[0]
    ids = converted[3].prompt_ids
    assert converted[2] == [ids]
    return converted, text, ids


def seed_items():
    return [
        {"role": "developer", "content": "Initial permissions."},
        {"role": "system", "content": "Initial workspace."},
        {"role": "user", "content": "First question."},
        {"role": "assistant", "content": "First answer."},
        {"role": "developer", "content": "Earlier inline update."},
        {"role": "user", "content": "Stable long prefix. " * 512},
    ]


class V41ResponsesPrefixTests(unittest.TestCase):
    def test_merge_initial_instructions_but_keep_later_instruction_roles(self):
        result = handler()
        items = seed_items() + [
            {"role": "assistant", "content": "Second answer."},
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Latest update."}],
            },
            {"role": "system", "content": "Another update."},
            {"role": "user", "content": "Next question."},
        ]
        req = request(items)
        before = copy.deepcopy(req.input)
        messages = result._construct_input_messages(req)
        self.assertEqual(
            messages[0],
            {
                "role": "system",
                "content": "Stable instructions.\n\nInitial permissions.\n\nInitial workspace.",
            },
        )
        self.assertEqual(
            [item["role"] for item in messages],
            [
                "system",
                "user",
                "assistant",
                "system",
                "user",
                "assistant",
                "system",
                "system",
                "user",
            ],
        )
        self.assertEqual(messages[3]["content"], "Earlier inline update.")
        self.assertEqual(
            messages[6]["content"], [{"type": "text", "text": "Latest update."}]
        )
        self.assertEqual(messages[7]["content"], "Another update.")
        self.assertEqual(req.input, before)

    def test_generic_and_other_encoder_coalescing_is_unchanged(self):
        for spec in (None, "dsv4", "dsv32", "inkling", "kimi_k3"):
            with self.subTest(spec=spec):
                result = make_serving()
                result.chat_encoding_spec = spec
                messages = result._construct_input_messages(request(seed_items()))
                self.assertEqual(
                    [item["role"] for item in messages],
                    ["system", "user", "assistant", "user"],
                )
                self.assertEqual(
                    messages[0]["content"],
                    "Stable instructions.\n\nInitial permissions.\n\nInitial workspace.\n\nEarlier inline update.",
                )

    def test_initial_only_construction_matches_legacy_conversion(self):
        for items in (
            seed_items()[:3],
            seed_items()[:4],
            [{"role": "user", "content": "No developer messages."}],
        ):
            with self.subTest(items=len(items)):
                self.assertEqual(
                    handler()._construct_input_messages(request(items)),
                    make_serving()._construct_input_messages(request(items)),
                )

    def test_additive_updates_preserve_real_encoder_input_prefix(self):
        for update in UPDATES:
            for role in ("developer", "system"):
                for thinking in (False, True):
                    for multimodal in (False, True):
                        with self.subTest(
                            update=update,
                            role=role,
                            thinking=thinking,
                            multimodal=multimodal,
                        ):
                            result = handler(multimodal)
                            old = request(seed_items(), thinking=thinking)
                            appended = [
                                {
                                    "type": "reasoning",
                                    "summary": [
                                        {
                                            "type": "summary_text",
                                            "text": "Reasoned answer.",
                                        }
                                    ],
                                },
                                {"role": "assistant", "content": "Completed answer."},
                                {
                                    "role": role,
                                    "content": [{"type": "input_text", "text": update}],
                                },
                                {
                                    "role": "user",
                                    "content": "Continue the same conversation.",
                                },
                            ]
                            new = request(seed_items() + appended, thinking=thinking)
                            before = copy.deepcopy(new.input)
                            _, old_text, old_ids = encode(result, old)
                            converted, new_text, new_ids = encode(result, new)
                            self.assertGreater(len(old_ids), 8000)
                            self.assertEqual(new_ids[: len(old_ids)], old_ids)
                            self.assertTrue(new_text.startswith(old_text))
                            self.assertIn("<｜System｜>" + update, new_text)
                            self.assertLess(
                                new_text.index("Completed answer."),
                                new_text.index(update),
                            )
                            self.assertEqual(
                                new_text.count("### Available Tool Schemas"), 1
                            )
                            self.assertLess(
                                new_text.index("### Available Tool Schemas"),
                                new_text.index("First question."),
                            )
                            self.assertEqual(new.input, before)
                            self.assertEqual(converted[0][-2]["role"], "system")

    def test_no_tools_keeps_chronology_with_native_thinking_drop_policy(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                result = handler()
                old = request(seed_items(), tools=False, thinking=thinking)
                new = request(
                    seed_items()
                    + [
                        {"role": "assistant", "content": "Completed answer."},
                        {"role": "developer", "content": UPDATES[0]},
                        {"role": "user", "content": "Continue."},
                    ],
                    tools=False,
                    thinking=thinking,
                )
                _, old_text, old_ids = encode(result, old)
                _, new_text, new_ids = encode(result, new)
                stable_ids = old_ids[: -len("<think>")] if thinking else old_ids
                self.assertEqual(new_ids[: len(stable_ids)], stable_ids)
                self.assertIn("<｜System｜>" + UPDATES[0], new_text)
                self.assertNotIn("### Available Tool Schemas", new_text)
                if thinking:
                    self.assertTrue(old_text.endswith("<think>"))

    def test_late_system_does_not_become_leading_system_when_no_initial_one(self):
        for tools in (False, True):
            with self.subTest(tools=tools):
                result = handler()
                old_items = [{"role": "user", "content": "First question."}]
                new_items = old_items + [
                    {"role": "assistant", "content": "First answer."},
                    {"role": "developer", "content": "Late instruction."},
                    {"role": "user", "content": "Next question."},
                ]
                _, _, old_ids = encode(
                    result,
                    request(old_items, instructions=None, tools=tools, thinking=False),
                )
                converted, new_text, new_ids = encode(
                    result,
                    request(new_items, instructions=None, tools=tools, thinking=False),
                )
                self.assertEqual(converted[0][0]["role"], "user")
                self.assertEqual(new_ids[: len(old_ids)], old_ids)
                self.assertLess(
                    new_text.index("First answer."),
                    new_text.index("<｜System｜>Late instruction."),
                )

    def test_inline_update_separates_consecutive_assistant_and_tool_items(self):
        items = seed_items() + [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Tool reasoning."}],
            },
            {"role": "assistant", "content": "Running the tool."},
            {
                "type": "function_call",
                "call_id": "call_a",
                "name": "Bash",
                "arguments": '{"command":"pwd"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_a",
                "output": "/workspace",
            },
            {"role": "developer", "content": "Instruction after tool result."},
            {"role": "assistant", "content": "Later assistant."},
            {"role": "user", "content": "Continue."},
        ]
        result = handler()
        converted, prompt, _ = encode(result, request(items))
        messages = converted[0]
        tool_index = next(
            i for i, item in enumerate(messages) if item["role"] == "tool"
        )
        before_tool = messages[tool_index - 1]
        self.assertEqual(before_tool["reasoning_content"], "Tool reasoning.")
        self.assertEqual(before_tool["content"], "Running the tool.")
        self.assertEqual(before_tool["tool_calls"][0]["id"], "call_a")
        self.assertEqual(messages[tool_index + 1]["role"], "system")
        self.assertEqual(messages[tool_index + 2]["content"], "Later assistant.")
        self.assertLess(
            prompt.index("/workspace</tool_result>"),
            prompt.index("Instruction after tool result."),
        )
        self.assertLess(
            prompt.index("Instruction after tool result."),
            prompt.index("Later assistant."),
        )

    def test_stored_history_preserves_inline_updates_without_mutation(self):
        result = handler()
        stored = [
            {"role": "user", "content": "Old question."},
            {"role": "system", "content": "Old inline instruction."},
            {"role": "user", "content": "Recent question."},
        ]
        result.msg_store["resp_previous"] = copy.deepcopy(stored)
        previous = SimpleNamespace(
            id="resp_previous",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Previous answer."}],
                }
            ],
        )
        req = request(
            [
                {"role": "developer", "content": "New inline instruction."},
                {"role": "user", "content": "Next question."},
            ]
        )
        messages = result._construct_input_messages(req, previous)
        self.assertEqual(
            [item["role"] for item in messages],
            ["system", "user", "system", "user", "assistant", "system", "user"],
        )
        self.assertEqual(messages[0]["content"], "Stable instructions.")
        self.assertEqual(messages[2]["content"], "Old inline instruction.")
        self.assertEqual(messages[-2]["content"], "New inline instruction.")
        self.assertEqual(result.msg_store["resp_previous"], stored)

    def test_changed_initial_instructions_still_invalidate_prefix(self):
        result = handler()
        _, _, old_ids = encode(result, request(seed_items()))
        _, _, new_ids = encode(
            result, request(seed_items(), instructions="Actually changed instructions.")
        )
        self.assertNotEqual(new_ids[: len(old_ids)], old_ids)


if __name__ == "__main__":
    unittest.main()
