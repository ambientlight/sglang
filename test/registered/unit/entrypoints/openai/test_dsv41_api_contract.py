"""V4.1 raw model text through real Chat/Messages/Responses adapter layers.

Only the GPU generator is replaced. Parsers and API stream ownership are real.
"""

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from utils import (
    StreamFixture,
    collect_stream_events,
    engine_chunk,
    event_payloads,
    make_serving,
)

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.chat_encoding import InlineSystemStrategy
from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.parser.reasoning_parser import ReasoningParser

SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
    "additionalProperties": False,
}


def protocol(arguments='{"command":"pwd"}'):
    return (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="Bash">\n'
        + arguments
        + "\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )


def serving():
    result = make_serving()
    result.tool_call_parser = "deepseekv41"
    result.reasoning_parser = "deepseek-v41"
    result._reasoning_detector = ReasoningParser("deepseek-v41").detector
    result.chat_encoding_spec = "dsv41"
    result._dsv41_default_reasoning_effort = "high"
    result._dsv41_unsupported_efforts_warned = set()
    result._tool_recovery_enabled = False
    result.tokenizer_manager.server_args.tool_call_parser = "deepseekv41"
    result.tokenizer_manager.server_args.reasoning_parser = "deepseek-v41"
    return result


def chunks(text, size, *, length=False):
    values = []
    for end in range(size, len(text) + size, size):
        final = end >= len(text)
        chunk = engine_chunk(text[:end], completion_tokens=end, finish=final)
        if final and length:
            chunk["meta_info"]["finish_reason"] = {"type": "length"}
        values.append(chunk)
    return values


class V41ResponsesContract(unittest.TestCase):
    def test_real_request_conversion_preserves_native_ids_on_vision_model(self):
        # Regression from the first live V4.1 Responses request. The previous
        # stream fixtures began after conversion and missed prompt="" with
        # populated native IDs on an is_multimodal=True model. Exercise the
        # real encoder, create_responses and API generator; stub only tokenize
        # and GPU generation. Cover text/image, plain/tool, full/stream paths.
        for with_image in (False, True):
            for with_tool in (False, True):
                for stream in (False, True):
                    with self.subTest(image=with_image, tool=with_tool, stream=stream):
                        handler = serving()
                        handler.default_chat_template_kwargs = {}
                        handler.tokenizer_manager.model_config.is_multimodal = True
                        handler.template_manager.chat_template_name = None
                        handler.tokenizer_manager.model_config.hf_config.image_token_id = (
                            128100
                        )
                        tokenizer = handler.tokenizer_manager.tokenizer
                        tokenizer.convert_ids_to_tokens.return_value = "<image>"
                        native_ids = [128000, 100, 200, 128001]
                        tokenizer.encode.return_value = native_ids
                        captured = []

                        async def generate(adapted, *_args, **_kwargs):
                            captured.append(adapted)
                            self.assertIsNone(adapted.text)
                            self.assertEqual(adapted.input_ids, native_ids)
                            self.assertEqual(
                                adapted.image_data,
                                (
                                    ["http://example.com/fixture.png"]
                                    if with_image
                                    else None
                                ),
                            )
                            if with_tool:
                                self.assertIn("structural_tag", adapted.sampling_params)
                            yield engine_chunk(
                                protocol() if with_tool else "42", finish=True
                            )

                        handler.tokenizer_manager.generate_request = Mock(
                            side_effect=generate
                        )
                        content = [
                            {
                                "type": "input_text",
                                "text": "Run pwd." if with_tool else "Reply 42.",
                            }
                        ]
                        if with_image:
                            content.append(
                                {
                                    "type": "input_image",
                                    "image_url": "http://example.com/fixture.png",
                                }
                            )
                        request = ResponsesRequest(
                            model="deepseek-v4.1-flash",
                            input=[{"role": "user", "content": content}],
                            stream=stream,
                            store=False,
                            max_output_tokens=1024,
                            chat_template_kwargs={"thinking": False},
                            tools=(
                                [
                                    {
                                        "type": "function",
                                        "name": "Bash",
                                        "parameters": SCHEMA,
                                        "strict": True,
                                    }
                                ]
                                if with_tool
                                else []
                            ),
                            tool_choice="required" if with_tool else "none",
                        )

                        async def run():
                            response = await handler.create_responses(request)
                            self.assertFalse(
                                hasattr(response, "status_code"),
                                getattr(response, "body", None),
                            )
                            if stream:
                                payloads = event_payloads(
                                    await collect_stream_events(response)
                                )
                                self.assertEqual(
                                    payloads[-1]["type"], "response.completed"
                                )
                                return payloads[-1]["response"]
                            return response.model_dump()

                        response = asyncio.run(run())
                        self.assertEqual(response["status"], "completed")
                        self.assertEqual(len(captured), 1)
                        self.assertEqual(
                            response["output"][0]["type"],
                            "function_call" if with_tool else "message",
                        )

    def run_stream(self, text, size=7, *, strict=False, length=False):
        handler = serving()
        request = ResponsesRequest(
            model="deepseek-v4.1-flash",
            input="run pwd",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "Bash",
                    "parameters": SCHEMA,
                    "strict": strict,
                }
            ],
        )
        events = StreamFixture(handler, request).run(chunks(text, size, length=length))
        return event_payloads(events)

    def test_tool_transaction_at_every_chunk_size(self):
        for size in (1, 2, 7, 31, 1000):
            with self.subTest(size=size):
                payloads = self.run_stream("I will check.\n" + protocol(), size)
                self.assertEqual(payloads[-1]["type"], "response.completed")
                response = payloads[-1]["response"]
                calls = [
                    item
                    for item in response["output"]
                    if item["type"] == "function_call"
                ]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["name"], "Bash")
                self.assertEqual(json.loads(calls[0]["arguments"]), {"command": "pwd"})
                self.assertNotIn("DSML", json.dumps(payloads, ensure_ascii=False))
                self.assertEqual(
                    [p["sequence_number"] for p in payloads], list(range(len(payloads)))
                )

    def test_malformed_tail_discards_pending_tools(self):
        for tail in ("<｜DSML｜ broken>", "<｜DSML｜ calls>", "<｜DSML｜ cal"):
            with self.subTest(tail=tail):
                payloads = self.run_stream(protocol() + tail)
                self.assertEqual(payloads[-1]["type"], "response.failed")
                self.assertFalse(
                    any(
                        p["type"] == "response.function_call_arguments.delta"
                        for p in payloads
                    )
                )
                self.assertEqual(payloads[-1]["response"]["output"], [])

    def test_schema_is_enforced_only_when_strict(self):
        for strict in (False, True):
            payloads = self.run_stream(protocol('{"command":42}'), strict=strict)
            self.assertEqual(
                payloads[-1]["type"],
                "response.failed" if strict else "response.completed",
            )

    def test_truncated_invoke_never_becomes_completed_tool(self):
        payloads = self.run_stream(
            '<｜DSML｜ calls>\n<｜DSML｜ invoke name="Bash">{"command":', length=True
        )
        self.assertEqual(payloads[-1]["type"], "response.failed")
        self.assertFalse(
            any(p["type"] == "response.function_call_arguments.delta" for p in payloads)
        )


class V41MessagesContract(unittest.TestCase):
    def run_stream(self, text, size, *, strict=False):
        chat = serving()
        closed = []

        async def generate(*_args):
            try:
                for chunk in chunks(text, size):
                    yield chunk
            finally:
                closed.append(True)

        chat.tokenizer_manager.generate_request = Mock(side_effect=generate)
        handler = AnthropicServing(chat)
        handler._tool_recovery_enabled = False
        request = AnthropicMessagesRequest(
            model="deepseek-v4.1-flash",
            max_tokens=1024,
            messages=[{"role": "user", "content": "run pwd"}],
            stream=True,
            tools=[{"name": "Bash", "input_schema": SCHEMA, "strict": strict}],
        )
        processed = handler._convert_to_chat_completion_request(request)
        events = asyncio.run(
            collect_stream_events(
                handler._generate_anthropic_stream(
                    SimpleNamespace(rid="fixture-rid"), processed, request, None
                )
            )
        )
        self.assertEqual(closed, [True])
        return event_payloads(events)

    def test_tool_arguments_close_before_postamble(self):
        for size in (1, 7, 1000):
            with self.subTest(size=size):
                events = self.run_stream(protocol() + "Queued.", size)
                self.assertEqual(events[-1]["type"], "message_stop")
                self.assertFalse(any(e["type"] == "error" for e in events))
                opened, stopped, arguments = {}, set(), ""
                for event in events:
                    kind, index = event["type"], event.get("index")
                    if kind == "content_block_start":
                        opened[index] = event["content_block"]["type"]
                    elif kind == "content_block_stop":
                        stopped.add(index)
                    elif kind == "content_block_delta":
                        self.assertNotIn(index, stopped)
                        self.assertIn(index, opened)
                        delta = event["delta"]
                        if delta["type"] == "input_json_delta":
                            arguments += delta["partial_json"]
                        elif delta["type"] == "text_delta":
                            self.assertTrue(
                                any(opened[i] == "tool_use" for i in stopped)
                            )
                self.assertEqual(json.loads(arguments), {"command": "pwd"})
                self.assertNotIn("DSML", json.dumps(events, ensure_ascii=False))

    def test_malformed_suffix_fails_closed_and_closes_generator(self):
        events = self.run_stream(protocol() + "<｜DSML｜ broken>", 7)
        self.assertEqual([e["type"] for e in events][-2:], ["error", "message_stop"])
        self.assertFalse(any(e["type"] == "message_delta" for e in events))
        self.assertFalse(
            any(e.get("content_block", {}).get("type") == "tool_use" for e in events)
        )

    def test_native_inline_system_order_and_failed_result_survive_conversion(self):
        handler = AnthropicServing(serving())
        self.assertEqual(
            handler._inline_system_strategy, InlineSystemStrategy.PASSTHROUGH
        )
        request = AnthropicMessagesRequest(
            model="deepseek-v4.1-flash",
            max_tokens=1024,
            system="initial",
            messages=[
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "c",
                            "name": "Bash",
                            "input": {"command": "pwd"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "c",
                            "is_error": True,
                            "content": "permission denied",
                        }
                    ],
                },
                {"role": "system", "content": "later"},
                {"role": "assistant", "content": "acknowledged"},
                {"role": "user", "content": "continue"},
            ],
        )
        converted = handler._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.content for m in converted.messages if m.role == "system"],
            ["initial", "later"],
        )
        tool = next(m for m in converted.messages if m.role == "tool")
        self.assertTrue(tool.is_error)


if __name__ == "__main__":
    unittest.main()
