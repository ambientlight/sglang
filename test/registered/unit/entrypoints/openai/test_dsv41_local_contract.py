"""Exercise the upstream DeepSeek recipe with the deployment's own HTTP behavior."""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_serving_deepseek_recipe import _serving, _tokenizer_file
from transformers import PreTrainedTokenizerFast
from utils import engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.dsv41_tool_recovery import ResponsesToolRecovery
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.function_call.core_types import MalformedToolCallError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TOOL = {
    "name": "lookup",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    },
}
CALL = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
    '<｜DSML｜ parameter name="q" string="true">value</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


class LocalDeepSeekTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file())

    def setUp(self):
        env = patch.dict(
            os.environ,
            {
                "SGLANG_DSV41_TOOL_CONTAINMENT": "1",
                "SGLANG_RESPONSES_TOOL_RECOVERY": "1",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.chat = _serving(self, self.tokenizer)
        self.original = ChatCompletionRequest(
            messages=[{"role": "user", "content": "query"}],
            tools=[{"type": "function", "function": TOOL}],
        )

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def test_transaction_keeps_all_tool_calls_private_until_valid(self):
        for suffix in (
            CALL,
            CALL.replace('name="lookup"', 'name="unknown"'),
            CALL[:60],
            CALL.replace('string="true">value', 'string="false">7'),
        ):
            parser = self.chat._encoding_output()._new_output_parser(self.original)
            pieces = [parser.feed([token]) for token in self.ids("Working" + suffix)]
            self.assertEqual("".join(p.content for p in pieces), "Working")
            self.assertFalse(any(p.tool_calls for p in pieces))
            if suffix == CALL:
                self.assertTrue(parser.finish([], {"type": "stop"}).tool_calls)
            else:
                with self.assertRaises(MalformedToolCallError):
                    parser.finish([], {"type": "stop"})

    def test_local_schema_references_for_both_tool_apis(self):
        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
        from sglang.srt.entrypoints.anthropic.serving import (
            _validate_anthropic_tool_input,
        )
        from sglang.srt.parser.dsv41_tool_transaction import validate_call

        def check(ref):
            schema = {
                "type": "object",
                "properties": {"q": {"$ref": ref}},
                "$defs": {"query": {"type": "string"}},
            }
            return (
                lambda: validate_call(
                    {"lookup": SimpleNamespace(strict=True, parameters=schema)},
                    "lookup",
                    '{"q":"value"}',
                ),
                lambda: _validate_anthropic_tool_input(
                    AnthropicMessagesRequest(
                        model="model",
                        max_tokens=8,
                        messages=[{"role": "user", "content": "query"}],
                        tools=[
                            {"name": "lookup", "input_schema": schema, "strict": True}
                        ],
                    ),
                    "lookup",
                    {"q": "value"},
                ),
            )

        for validate in check("#/$defs/query"):
            validate()
        for validate in check("urn:fixture:unregistered-schema"):
            with self.assertRaisesRegex(MalformedToolCallError, "schema is invalid"):
                validate()

    def test_error_result_rendering_and_retrocompatible_prompt_ids(self):
        def render(error):
            request = self.original.model_copy(
                update={
                    "messages": ChatCompletionRequest(
                        messages=[
                            {"role": "user", "content": "query"},
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "id": "a",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "a",
                                "content": "fixture",
                                **error,
                            },
                            {"role": "user", "content": "continue"},
                        ]
                    ).messages
                }
            )
            return self.chat._process_messages(request, False).prompt_ids

        self.assertEqual(render({}), render({"is_error": False}))
        text = self.tokenizer.decode(render({"is_error": True}))
        self.assertIn(
            "Tool execution failed. Treat the following as an error result:\nfixture",
            text,
        )

    def test_progress_before_gpu_final_and_one_bounded_retry(self):
        server = make_serving()
        server._encoding_adapter = self.chat._encoding_adapter
        server.chat_encoding_spec = "dsv41"
        server.tool_call_parser = "deepseekv41"
        server.reasoning_parser = "deepseek-v41"
        server._dsv41_recovery = ResponsesToolRecovery()
        request = ResponsesRequest(
            model="model",
            input="query",
            stream=True,
            store=False,
            tools=[{"type": "function", **TOOL}],
        )
        server._convert_to_internal_request = Mock(
            side_effect=lambda req, raw: (SimpleNamespace(), req)
        )
        server._handle_non_streaming_request = AsyncMock(
            return_value=ChatCompletionResponse(
                id="r",
                model="model",
                created=1,
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "ok",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":"repaired"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                usage={"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            )
        )

        async def compare():
            release = asyncio.Event()

            async def source():
                initial = self.ids("Working ")
                head = engine_chunk("Working ", len(initial))
                head["output_ids"] = initial
                yield head
                await asyncio.wait_for(release.wait(), 3)
                ids = self.ids(
                    "Working " + CALL.replace('string="true">value', 'string="false">7')
                )
                end = engine_chunk("", len(ids), finish=True)
                end["output_ids"] = ids
                yield end

            events = []
            async for frame in server.responses_stream_generator_non_harmony(
                request,
                {},
                source(),
                "model",
                self.tokenizer,
                RequestResponseMetadata(request_id=request.request_id),
                require_reasoning=False,
                recovery_context=(self.original, None),
            ):
                for item in event_payloads([frame]):
                    events.append(item)
                    if item["type"] == "response.output_text.delta":
                        release.set()
            return events

        events = asyncio.run(compare())
        self.assertEqual(events[-1]["type"], "response.completed", events[-1])
        calls = [
            i for i in events[-1]["response"]["output"] if i["type"] == "function_call"
        ]
        self.assertEqual(
            [json.loads(c["arguments"]) for c in calls], [{"q": "repaired"}]
        )
        phases = [
            e["item"]["phase"]
            for e in events
            if e["type"] == "response.output_item.added"
            and e["item"]["type"] == "message"
        ]
        self.assertEqual(phases, ["commentary"])
        self.assertEqual(server._handle_non_streaming_request.await_count, 1)
        self.assertTrue(
            server._convert_to_internal_request.call_args.args[0]
            .tools[0]
            .function.strict
        )
