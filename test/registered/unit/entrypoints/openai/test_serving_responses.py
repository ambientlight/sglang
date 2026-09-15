import asyncio
import json
import pathlib
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from utils import (
    collect_stream_events,
    engine_chunk,
    event_payloads,
    event_types,
    make_serving,
)

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    MessageProcessingResult,
    RequestResponseMetadata,
    ResponsesRequest,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_responses import (
    OpenAIServingResponses,
    _build_output_text_logprobs,
    _RecoveredResponseToolCall,
    _ResponsesToolRecoveryResult,
    _should_emit_normal_text_as_message,
)
from sglang.srt.function_call.core_types import MalformedToolCallError, ToolCallItem
from sglang.srt.parser.template_detection import ReasoningToggleConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class InputMessageConstructionTestCase(CustomTestCase):
    def test_previous_response_replays_assistant_text_not_instructions(self):
        serving = make_serving()
        prev_response = Mock(id="resp_prev")
        prev_response.output = [
            ResponseReasoningItem(
                id="rs_prev", summary=[], type="reasoning", content=None, status=None
            ),
            ResponseOutputMessage(
                id="msg_prev",
                content=[
                    ResponseOutputText(
                        text="first answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                    ResponseOutputText(
                        text="second answer part",
                        annotations=[],
                        type="output_text",
                        logprobs=None,
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
        serving.msg_store["resp_prev"] = [{"role": "user", "content": "old input"}]

        request = ResponsesRequest(
            model="x",
            instructions="Be brief",
            previous_response_id="resp_prev",
            input="new input",
            store=False,
        )

        messages = serving._construct_input_messages(request, prev_response)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "old input"},
                {
                    "role": "assistant",
                    "content": "first answer part\nsecond answer part",
                },
                {"role": "user", "content": "new input"},
            ],
        )

    def test_input_parts_normalized_for_chat_templates(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            store=False,
        )

        messages = serving._construct_input_messages(request)

        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "http://example.com/cat.png",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
        )

    def test_previous_response_id_input_list_does_not_call_copy_module(self):
        serving = make_serving()
        serving.use_harmony = True
        prev = Mock(id="resp_prev")
        prev.output = [
            ResponseFunctionToolCall(
                arguments="{}",
                call_id="call_x",
                name="t",
                type="function_call",
                id="fc_x",
                status="completed",
            )
        ]
        request = ResponsesRequest(
            model="x",
            input=[{"role": "user", "content": "hi"}],
            previous_response_id="resp_prev",
            store=False,
        )
        try:
            serving._construct_input_messages_with_harmony(request, prev)
        except TypeError as exc:
            self.fail(f"copy() module-call regression: {exc}")
        except Exception:
            pass


class ChatToolForwardingTestCase(CustomTestCase):
    def test_make_request_passes_function_tools_to_chat_processing(self):
        serving = make_serving()
        seen = {}

        def fake_process(chat_request, is_multimodal):
            seen["tools"] = chat_request.tools
            seen["tool_choice"] = chat_request.tool_choice
            seen["parallel_tool_calls"] = chat_request.parallel_tool_calls
            return MessageProcessingResult(
                prompt="prompt",
                prompt_ids=[1, 2, 3],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=["</s>"],
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

        serving._process_messages = Mock(side_effect=fake_process)
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="required",
            parallel_tool_calls=False,
            store=False,
        )

        messages, request_prompts, engine_prompts, processed, chat_request = (
            asyncio.run(
                serving._make_request(
                    request, None, serving.tokenizer_manager.tokenizer
                )
            )
        )

        self.assertEqual(messages, [{"role": "user", "content": "call the tool"}])
        self.assertEqual(request_prompts, [[1, 2, 3]])
        self.assertEqual(engine_prompts, [[1, 2, 3]])
        self.assertEqual(seen["tools"][0].function.name, "lookup")
        self.assertEqual(seen["tool_choice"], "required")
        self.assertFalse(seen["parallel_tool_calls"])
        self.assertEqual(processed.tool_call_constraint[0], "json_schema")
        self.assertEqual(chat_request.max_tokens, request.max_output_tokens)

    def test_required_tool_choice_without_function_tool_returns_400(self):
        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[{"type": "web_search"}, {"type": "mcp"}],
            store=False,
        )
        result = asyncio.run(serving.create_responses(request, raw_request=None))
        self.assertEqual(getattr(result, "status_code", None), 400)

    def test_kimi_k3_request_uses_chat_encoder_fields(self):
        serving = make_serving()
        serving.chat_encoding_spec = "kimi_k3"
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        serving.tokenizer_manager.tokenizer.apply_chat_template.return_value = [4, 5, 6]
        request = ResponsesRequest(
            model="x",
            input="Explain <|kimi_image_placeholder|>",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            tool_choice="required",
            reasoning={"effort": "high"},
            store=False,
        )

        _, request_prompts, engine_prompts, _, _ = asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )

        call = serving.tokenizer_manager.tokenizer.apply_chat_template.call_args
        self.assertEqual(
            call.args[0][0]["content"], "Explain <| kimi_image_placeholder |>"
        )
        self.assertEqual(call.kwargs["thinking_effort"], "high")
        self.assertEqual(call.kwargs["tool_choice"], "required")
        self.assertEqual(call.kwargs["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(request_prompts, [[4, 5, 6]])
        self.assertEqual(engine_prompts, [[4, 5, 6]])


class ReasoningRequestForwardingTestCase(unittest.TestCase):
    def test_create_responses_uses_processed_reasoning_state(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.default_chat_template_kwargs = {"thinking": False}
        serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        rendered = MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["adapted_request"] = adapted_request
            context.append_output(
                {
                    "text": "done",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input="answer",
            request_id="resp_reasoning",
            store=False,
        )

        with (
            patch.object(
                serving, "_apply_conversation_template", return_value=rendered
            ),
            patch(
                "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
            ) as parser_cls,
        ):
            parser_cls.return_value.parse_non_stream.return_value = (None, "done")
            response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        self.assertFalse(captured["adapted_request"].require_reasoning)
        self.assertFalse(parser_cls.call_args.kwargs["force_reasoning"])


class SkipSpecialTokensForwardingTestCase(CustomTestCase):
    """The skip_special_tokens override from _process_messages must reach the
    engine sampling params; muse's channel markers die in detok otherwise."""

    def _create_responses_sampling_params(self, serving):
        serving.default_chat_template_kwargs = None
        rendered = MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["sampling_params"] = sampling_params
            context.append_output(
                {
                    "text": "done",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input="answer",
            request_id="resp_skip_special",
            store=False,
        )

        with (
            patch.object(
                serving, "_apply_conversation_template", return_value=rendered
            ),
            patch(
                "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
            ) as parser_cls,
        ):
            parser_cls.return_value.parse_non_stream.return_value = (None, "done")
            response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        return captured["sampling_params"]

    def test_marker_preserving_parser_disables_skip_special_tokens(self):
        serving = make_serving()
        serving.reasoning_parser = "muse"
        params = self._create_responses_sampling_params(serving)
        self.assertFalse(params["skip_special_tokens"])

    def test_default_parser_keeps_skip_special_tokens(self):
        serving = make_serving()
        params = self._create_responses_sampling_params(serving)
        # The chat request's True is a synthesized default (ResponsesRequest has
        # no such field), so leave it unset for --preferred-sampling-params.
        self.assertNotIn("skip_special_tokens", params)


class InputItemNormalizationTestCase(CustomTestCase):
    def test_function_call_becomes_assistant_tool_call(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "lookup",
                "arguments": '{"key": "val"}',
                "status": "completed",
            }
        )
        self.assertEqual(
            normalized,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key": "val"}',
                        },
                    }
                ],
            },
        )

    def test_developer_role_becomes_system(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {"role": "developer", "content": "Be terse."}
        )
        self.assertEqual(normalized, {"role": "system", "content": "Be terse."})

    def test_function_call_output_becomes_tool_message(self):
        normalized = OpenAIServingResponses._normalize_response_message_for_chat(
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "42",
            }
        )
        self.assertEqual(
            normalized,
            {"role": "tool", "tool_call_id": "call_abc", "content": "42"},
        )

    def test_unknown_input_item_type_raises(self):
        with self.assertRaises(ValueError):
            OpenAIServingResponses._normalize_response_message_for_chat(
                {"type": "web_search_call", "id": "ws_1"}
            )


class FullResponseUsageTestCase(CustomTestCase):
    def test_full_response_uses_dict_meta_info_for_usage(self):
        serving = make_serving()
        context = SimpleContext()
        context.last_output = {
            "text": "done",
            "meta_info": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "cached_tokens": 3,
                "reasoning_tokens": 2,
            },
        }
        request = ResponsesRequest(
            model="x", input="hello", request_id="resp_usage", store=False
        )
        metadata = RequestResponseMetadata(request_id=request.request_id)

        async def empty_generator():
            for _ in ():
                yield None

        response = asyncio.run(
            serving.responses_full_generator(
                request,
                sampling_params={},
                result_generator=empty_generator(),
                context=context,
                model_name="x",
                tokenizer=serving.tokenizer_manager.tokenizer,
                request_metadata=metadata,
                created_time=123,
                require_reasoning=False,
            )
        )

        self.assertEqual(response.usage.prompt_tokens, 11)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.usage.reasoning_tokens, 2)
        self.assertEqual(metadata.final_usage_info, response.usage)


class MultimodalRequestTestCase(CustomTestCase):
    def test_shared_prompt_selection_matches_each_encoder_contract(self):
        serving = make_serving(is_multimodal=True)
        processed = MessageProcessingResult(
            prompt="standard vision text",
            prompt_ids=[10, 20],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        for spec, expected in (
            (None, processed.prompt),
            ("dsv41", [10, 20]),
            ("kimi_k3", [10, 20]),
            ("inkling", [10, 20]),
        ):
            serving.chat_encoding_spec = spec
            self.assertEqual(serving._select_engine_prompt(processed, True), expected)
            self.assertEqual(serving._select_engine_prompt(processed, False), [10, 20])
            self.assertEqual(
                serving._select_engine_prompt(processed, True, input_ids_provided=True),
                [10, 20],
            )
        processed.prompt_ids = []
        self.assertEqual(
            serving._select_engine_prompt(processed, True), processed.prompt
        )
        processed.prompt_ids = "text-only pre-rendered input"
        self.assertEqual(
            serving._select_engine_prompt(processed, False), processed.prompt_ids
        )

    def test_text_only_create_responses_rejects_media_before_generation(self):
        serving = make_serving()
        serving._process_messages = Mock()
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe it"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            store=False,
        )

        response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"received unsupported content type 'image_url'", response.body)
        serving._process_messages.assert_not_called()
        serving.tokenizer_manager.generate_request.assert_not_called()

    def test_multimodal_create_responses_sends_text_and_media_to_engine(self):
        serving = make_serving(is_multimodal=True)
        captured = {}

        serving._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="rendered multimodal prompt",
                prompt_ids=[9, 9, 9],
                image_data=["http://example.com/cat.png"],
                audio_data=None,
                video_data=None,
                modalities=["image"],
                stop=[],
            )
        )

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["request_prompt"] = request_prompt
            captured["adapted_request"] = adapted_request
            context.append_output(
                {
                    "text": "looks like a cat",
                    "meta_info": {
                        "prompt_tokens": 5,
                        "completion_tokens": 4,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        request = ResponsesRequest(
            model="x",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe it"},
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/cat.png",
                        },
                    ],
                }
            ],
            request_id="resp_mm",
            store=False,
        )

        response = asyncio.run(serving.create_responses(request))

        self.assertEqual(response.status, "completed")
        self.assertEqual(captured["request_prompt"], "rendered multimodal prompt")
        self.assertEqual(captured["adapted_request"].text, "rendered multimodal prompt")
        self.assertIsNone(captured["adapted_request"].input_ids)
        self.assertEqual(
            captured["adapted_request"].image_data, ["http://example.com/cat.png"]
        )
        self.assertEqual(captured["adapted_request"].modalities, ["image"])


class OutputItemsTestCase(CustomTestCase):
    def setUp(self):
        # qwen3_coder is the default for this class; the one no-native-parser
        # case overrides it.
        self.serving = make_serving()
        self.serving.tool_call_parser = "qwen3_coder"

    def _function_tool_request(self):
        return ResponsesRequest(
            model="x",
            input="weather?",
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

    def test_function_tool_call_extracted_via_parser(self):
        serving = self.serving
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = ("trailing text", [fake_call])
            output_items = serving._make_response_output_items(
                self._function_tool_request(),
                "raw model output with <tool_call>",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')

        message_items = [
            item for item in output_items if isinstance(item, ResponseOutputMessage)
        ]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0].content[0].text, "trailing text")

    def test_prose_emitted_before_tool_call_item(self):
        serving = self.serving
        fake_call = ToolCallItem(
            tool_index=0, name="get_weather", parameters='{"city": "Beijing"}'
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            instance = parser_cls.return_value
            instance.has_tool_call.return_value = True
            instance.parse_non_stream.return_value = (
                "I'll check the weather.",
                [fake_call],
            )
            output_items = serving._make_response_output_items(
                self._function_tool_request(),
                "raw model output",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        types = [type(item).__name__ for item in output_items]
        self.assertEqual(types, ["ResponseOutputMessage", "ResponseFunctionToolCall"])

    def test_required_tool_choice_parses_json_array_without_native_parser(self):
        serving = self.serving
        serving.tool_call_parser = None
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
            store=False,
        )
        raw = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        output_items = serving._make_response_output_items(
            request, raw, tokenizer=Mock(), require_reasoning=False
        )

        tool_calls = [
            item for item in output_items if isinstance(item, ResponseFunctionToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, '{"city": "Beijing"}')
        self.assertEqual(
            [item for item in output_items if isinstance(item, ResponseOutputMessage)],
            [],
        )

    def test_required_tool_choice_skips_json_fallback_for_native_parser(self):
        """muse reports parses_required_natively, so required output must not
        be pushed through the orjson JSON-array fallback (mirrors chat)."""
        serving = self.serving
        serving.tool_call_parser = "muse"
        request = ResponsesRequest(
            model="x",
            input="hi",
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
            store=False,
        )
        raw = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        output_items = serving._make_response_output_items(
            request, raw, tokenizer=Mock(), require_reasoning=False
        )

        self.assertEqual(
            [
                item
                for item in output_items
                if isinstance(item, ResponseFunctionToolCall)
            ],
            [],
        )
        message_items = [
            item for item in output_items if isinstance(item, ResponseOutputMessage)
        ]
        self.assertEqual(len(message_items), 1)
        self.assertEqual(message_items[0].content[0].text, raw)

    def test_no_tool_call_extraction_when_tool_choice_none(self):
        serving = self.serving
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            tool_choice="none",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            output_items = serving._make_response_output_items(
                request,
                "just a plain answer",
                tokenizer=Mock(),
                require_reasoning=False,
            )
            parser_cls.assert_not_called()

        self.assertEqual(len(output_items), 1)
        self.assertIsInstance(output_items[0], ResponseOutputMessage)


class HarmonyResponsesTestCase(CustomTestCase):
    def test_developer_message_skips_unsupported_tool_types(self):
        from sglang.srt.entrypoints.harmony_utils import get_developer_message
        from sglang.srt.entrypoints.openai.protocol import ResponseTool

        tools = [
            ResponseTool(
                type="function",
                name="get_weather",
                description="Look up weather.",
                parameters={"type": "object"},
            ),
            ResponseTool(type="web_search"),
            ResponseTool(type="namespace", name="codex"),
            ResponseTool(type="mcp"),
        ]
        msg = get_developer_message(instructions="be helpful", tools=tools)
        self.assertIsNotNone(msg)


class StatusFromFinishReasonTestCase(CustomTestCase):
    def test_only_length_maps_to_incomplete(self):
        fn = OpenAIServingResponses._status_from_finish_reason
        self.assertEqual(fn({"type": "length"}), "incomplete")
        self.assertEqual(fn("length"), "incomplete")
        for other in ({"type": "stop"}, {"type": "tool_calls"}, "stop", None):
            self.assertEqual(fn(other), "completed", other)


class BuildOutputTextLogprobsTestCase(CustomTestCase):
    def test_tokens_and_top_logprobs_are_converted(self):
        meta_info = {
            "output_token_logprobs": [(-0.1, 10, "Hello"), (-0.2, 11, " world")],
            "output_top_logprobs": [
                [(-0.1, 10, "Hello"), (-2.0, 12, "Hi")],
                [(-0.2, 11, " world"), (-3.0, 13, " earth")],
            ],
        }
        out = _build_output_text_logprobs(meta_info)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].token, "Hello")
        self.assertEqual(out[0].logprob, -0.1)
        self.assertEqual(out[0].bytes, list("Hello".encode("utf-8")))
        self.assertEqual(len(out[0].top_logprobs), 2)
        self.assertEqual(out[0].top_logprobs[0].token, "Hello")
        self.assertEqual(out[1].token, " world")

    def test_no_top_logprobs_yields_empty_lists(self):
        meta_info = {
            "output_token_logprobs": [(-0.5, 7, "hi")],
            "output_top_logprobs": None,
        }
        out = _build_output_text_logprobs(meta_info)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].top_logprobs, [])


class ChatToolChoiceConversionTestCase(CustomTestCase):
    def test_conversion(self):
        fn = OpenAIServingResponses._chat_tool_choice
        for s in ("auto", "required", "none"):
            self.assertEqual(fn(s), s)
        # Input is an effective_tool_choice() result, so the only object form
        # reaching here is a named function; degrading the rest to "auto"
        # happens there, once, so the echoed and the honored value agree.
        self.assertEqual(
            fn({"type": "function", "name": "get_weather"}),
            {"type": "function", "function": {"name": "get_weather"}},
        )


class ShouldEmitNormalTextTestCase(CustomTestCase):
    def test_whitespace_suppressed_only_while_a_tool_is_open(self):
        emit = _should_emit_normal_text_as_message
        self.assertFalse(emit("", any_tool_call_in_progress=False))
        # whitespace between tool blocks is an inter-call separator, not content
        self.assertFalse(emit("\n", any_tool_call_in_progress=True))
        self.assertTrue(emit("\n", any_tool_call_in_progress=False))
        self.assertTrue(emit("hello", any_tool_call_in_progress=True))


class ResponsesToolTransactionTestCase(CustomTestCase):
    @staticmethod
    def _request(*, strict=False):
        return ResponsesRequest(
            model="x",
            input="run pwd",
            request_id="resp_transaction",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "Bash",
                    "strict": strict,
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                }
            ],
        )

    @staticmethod
    def _parser_patch(serving, *, parse_side_effect, parse_end=("", [])):
        serving.tool_call_parser = "deepseekv4"
        parser = Mock()
        parser.detector.supports_structural_tag.return_value = True
        parser.parse_stream_chunk.side_effect = parse_side_effect
        parser.parse_stream_end.return_value = parse_end
        return patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser",
            return_value=parser,
        )

    @staticmethod
    def _run(serving, request, chunks, *, processed_request=None):
        async def result_generator():
            for chunk in chunks:
                yield chunk

        async def collect():
            return await collect_stream_events(
                serving.responses_stream_generator_non_harmony(
                    request,
                    sampling_params={},
                    result_generator=result_generator(),
                    model_name="x",
                    tokenizer=Mock(),
                    request_metadata=RequestResponseMetadata(
                        request_id=request.request_id
                    ),
                    require_reasoning=False,
                    processed_request=processed_request,
                )
            )

        return asyncio.run(collect())

    @staticmethod
    def _event_pairs(events):
        return list(zip(event_types(events), event_payloads(events)))

    def test_malformed_suffix_never_publishes_buffered_tool_fragments(self):
        serving = make_serving()
        request = self._request()
        parsed_call = ToolCallItem(
            tool_index=0,
            name="Bash",
            parameters='{"command":"pwd"}',
        )
        parser_patch = self._parser_patch(
            serving,
            parse_side_effect=[
                ("", [parsed_call]),
                MalformedToolCallError(
                    "Model emitted malformed DeepSeek tool-call protocol"
                ),
            ],
        )

        with parser_patch:
            events = self._run(
                serving,
                request,
                [engine_chunk("first"), engine_chunk("firstbad")],
            )

        pairs = self._event_pairs(events)
        types = [event_type for event_type, _ in pairs]
        self.assertEqual(
            types, ["response.created", "response.in_progress", "response.failed"]
        )
        failed = pairs[-1][1]["response"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["output"], [])
        self.assertEqual(failed["error"]["sglang_code"], "malformed_protocol")
        self.assertIn("malformed DeepSeek", failed["error"]["message"])
        self.assertEqual(
            [payload["sequence_number"] for _, payload in pairs],
            list(range(len(pairs))),
        )

    def test_malformed_suffix_is_replaced_by_one_validated_recovery(self):
        serving = make_serving()
        serving._tool_recovery_enabled = True
        request = self._request()
        parsed_call = ToolCallItem(
            tool_index=0,
            name="Bash",
            parameters='{"command":"damaged"}',
        )
        recovered = _ResponsesToolRecoveryResult(
            calls=(
                _RecoveredResponseToolCall(
                    call_id="call_recovered",
                    name="Bash",
                    arguments='{"command":"pwd"}',
                ),
            ),
            usage=UsageInfo(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
            recovery_rid="fresh_toolrecovery",
        )
        serving._attempt_response_tool_recovery = AsyncMock(return_value=recovered)
        parser_patch = self._parser_patch(
            serving,
            parse_side_effect=[
                ("", [parsed_call]),
                MalformedToolCallError(
                    "Model emitted malformed DeepSeek tool-call protocol"
                ),
            ],
        )

        with parser_patch:
            events = self._run(
                serving,
                request,
                [engine_chunk("first"), engine_chunk("firstbad", 2)],
                processed_request=Mock(),
            )

        pairs = self._event_pairs(events)
        types = [event_type for event_type, _ in pairs]
        self.assertNotIn("response.failed", types)
        self.assertEqual(types[-1], "response.completed")
        self.assertEqual(types.count("response.output_item.added"), 1)
        self.assertEqual(types.count("response.function_call_arguments.delta"), 1)
        completed = pairs[-1][1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(len(completed["output"]), 1)
        self.assertEqual(completed["output"][0]["name"], "Bash")
        self.assertEqual(
            json.loads(completed["output"][0]["arguments"]),
            {"command": "pwd"},
        )
        self.assertEqual(completed["usage"]["input_tokens"], 16)
        self.assertEqual(completed["usage"]["output_tokens"], 5)
        callback = serving._attempt_response_tool_recovery.await_args
        self.assertEqual(callback.args[3], "malformed_protocol")
        self.assertEqual(callback.args[5], ("Bash",))

    def test_strict_schema_mismatch_fails_before_any_tool_event(self):
        serving = make_serving()
        request = self._request(strict=True)
        parser_patch = self._parser_patch(
            serving,
            parse_side_effect=[
                (
                    "",
                    [
                        ToolCallItem(
                            tool_index=0,
                            name="Bash",
                            parameters="{}",
                        )
                    ],
                )
            ],
        )

        with parser_patch:
            events = self._run(serving, request, [engine_chunk("call", finish=True)])

        pairs = self._event_pairs(events)
        self.assertEqual(
            [event_type for event_type, _ in pairs],
            ["response.created", "response.in_progress", "response.failed"],
        )
        error = pairs[-1][1]["response"]["error"]
        self.assertEqual(error["sglang_code"], "schema_mismatch")
        self.assertIn("declared schema", error["message"])

    def test_recovery_closes_visible_prefix_before_publishing_recovered_tool(self):
        serving = make_serving()
        serving._tool_recovery_enabled = True
        request = self._request()
        serving._attempt_response_tool_recovery = AsyncMock(
            return_value=_ResponsesToolRecoveryResult(
                calls=(
                    _RecoveredResponseToolCall(
                        call_id="call_recovered",
                        name="Bash",
                        arguments='{"command":"pwd"}',
                    ),
                ),
                usage=UsageInfo(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
                recovery_rid="fresh_toolrecovery",
            )
        )
        parser_patch = self._parser_patch(
            serving,
            parse_side_effect=[
                ("visible prefix", []),
                MalformedToolCallError(
                    "Model emitted malformed DeepSeek tool-call protocol"
                ),
            ],
        )

        with parser_patch:
            events = self._run(
                serving,
                request,
                [engine_chunk("first"), engine_chunk("firstbad")],
                processed_request=Mock(),
            )

        pairs = self._event_pairs(events)
        message_done = next(
            index
            for index, (event_type, payload) in enumerate(pairs)
            if event_type == "response.output_item.done"
            and payload["item"]["type"] == "message"
        )
        tool_added = next(
            index
            for index, (event_type, payload) in enumerate(pairs)
            if event_type == "response.output_item.added"
            and payload["item"]["type"] == "function_call"
        )
        self.assertLess(message_done, tool_added)
        completed_output = pairs[-1][1]["response"]["output"]
        self.assertEqual(
            [item["type"] for item in completed_output],
            ["message", "function_call"],
        )
        callback = serving._attempt_response_tool_recovery.await_args
        self.assertEqual(callback.args[5], ())

    def test_non_strict_schema_mismatch_preserves_responses_api_semantics(self):
        serving = make_serving()
        request = self._request(strict=False)
        parser_patch = self._parser_patch(
            serving,
            parse_side_effect=[
                (
                    "",
                    [
                        ToolCallItem(
                            tool_index=0,
                            name="Bash",
                            parameters="{}",
                        )
                    ],
                )
            ],
        )

        with parser_patch:
            events = self._run(serving, request, [engine_chunk("call", finish=True)])

        pairs = self._event_pairs(events)
        self.assertEqual(pairs[-1][0], "response.completed")
        self.assertEqual(pairs[-1][1]["response"]["output"][0]["arguments"], "{}")

    def test_failure_dump_contains_only_bounded_raw_pre_parser_tail(self):
        serving = make_serving()
        serving.tool_call_parser = "deepseekv4"
        serving._tool_failure_tail_chars = 12
        request = self._request()
        parser = Mock()
        parser.detector.supports_structural_tag.return_value = True
        parser.parse_stream_chunk.side_effect = MalformedToolCallError(
            "Model emitted malformed DeepSeek tool-call protocol"
        )

        with tempfile.TemporaryDirectory() as directory:
            serving._tool_failure_dump_dir = pathlib.Path(directory)
            with patch(
                "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser",
                return_value=parser,
            ):
                self._run(
                    serving,
                    request,
                    [engine_chunk("prefix-sensitive-tail")],
                )

            dumps = list(pathlib.Path(directory).glob("*.json"))
            self.assertEqual(len(dumps), 1)
            payload = json.loads(dumps[0].read_text())
            self.assertEqual(payload["request_id"], request.request_id)
            self.assertEqual(payload["tail_chars"], 12)
            self.assertTrue(
                "prefix-sensitive-tail".endswith(payload["raw_pre_parser_tail"])
            )
            self.assertEqual(
                stat.S_IMODE(dumps[0].stat().st_mode),
                0o600,
            )

    def test_recovery_is_fresh_bounded_strict_and_intent_constrained(self):
        serving = make_serving()
        serving._tool_recovery_enabled = True
        request = self._request()
        chat_tools = serving._response_tools_to_chat_tools(request)
        chat_tools.append(
            chat_tools[0].model_copy(
                update={
                    "function": chat_tools[0].function.model_copy(
                        update={"name": "Read"}
                    )
                }
            )
        )
        processed_request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "run pwd"}],
            tools=chat_tools,
            tool_choice="auto",
            stream=True,
            max_tokens=50000,
            session_id="shared-session",
            session_params={"offset": 1},
        )
        adapted = SimpleNamespace(
            rid="header-overridden-rid",
            received_time=None,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            disagg_prefill_dp_rank=None,
        )
        serving._convert_to_internal_request = Mock(
            return_value=(adapted, processed_request)
        )
        serving._handle_non_streaming_request = AsyncMock(
            return_value=ChatCompletionResponse.model_validate(
                {
                    "id": "chatcmpl-recovery",
                    "model": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_recovered",
                                        "type": "function",
                                        "function": {
                                            "name": "Bash",
                                            "arguments": '{"command":"pwd"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                }
            )
        )

        result = asyncio.run(
            serving._attempt_response_tool_recovery(
                processed_request,
                request,
                None,
                "malformed_protocol",
                "original-rid",
                ("Bash",),
            )
        )

        self.assertIsNotNone(result)
        recovery_request = serving._convert_to_internal_request.call_args.args[0]
        self.assertFalse(recovery_request.stream)
        self.assertIsNone(recovery_request.session_id)
        self.assertIsNone(recovery_request.session_params)
        self.assertEqual(recovery_request.max_tokens, 8192)
        self.assertEqual(
            [tool.function.name for tool in recovery_request.tools],
            ["Bash"],
        )
        self.assertTrue(recovery_request.tools[0].function.strict)
        self.assertEqual(recovery_request.tool_choice.function.name, "Bash")
        self.assertNotEqual(recovery_request.rid, "original-rid")
        self.assertTrue(recovery_request.rid.endswith("_toolrecovery"))
        self.assertEqual(adapted.rid, recovery_request.rid)


class ResponsesGenerationCleanupTestCase(CustomTestCase):
    def test_closing_outer_stream_closes_suspended_tokenizer_stream(self):
        serving = make_serving()
        child_closed = False

        async def child_stream():
            nonlocal child_closed
            try:
                yield {"text": "partial", "meta_info": {}}
                await asyncio.Future()
            finally:
                child_closed = True

        serving.tokenizer_manager.generate_request = Mock(
            side_effect=lambda *_: child_stream()
        )
        adapted = SimpleNamespace()
        context = SimpleContext()

        async def consume_then_close():
            outer = serving._generate_with_builtin_tools(
                "rid",
                [],
                adapted,
                {},
                context,
            )
            await outer.__anext__()
            await outer.aclose()

        asyncio.run(consume_then_close())
        self.assertTrue(child_closed)


class EnginePassthroughTestCase(CustomTestCase):
    """Both flags cross hops with no type contract, and dropping either fails
    silently."""

    def _capture(self, serving, request):
        # Let the real _process_messages run: it is the hop that turns
        # skip_special_tokens off, so mocking it would make that assertion vacuous.
        # chat_template_name=None routes it through the tokenizer's template
        # (mocked) instead of the conversation registry, which has no fixture entry.
        serving.default_chat_template_kwargs = {}
        serving.template_manager.chat_template_name = None
        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["adapted_request"] = adapted_request
            captured["sampling_params"] = sampling_params
            context.append_output(
                {
                    "text": "ok",
                    "meta_info": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        asyncio.run(serving.create_responses(request))
        return captured

    def test_require_reasoning_forwarded_when_reasoning_parser_configured(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )

        captured = self._capture(
            serving, ResponsesRequest(model="x", input="hi", store=False)
        )

        self.assertTrue(captured["adapted_request"].require_reasoning)

    def test_prefilled_think_template_opens_the_parser(self):
        """``force_reasoning`` is a template property, not a request one, so it
        drives the parser but never the engine flag -- as on the chat path."""
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.template_manager.force_reasoning = True

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_non_stream.return_value = (None, "hi")
            serving._make_response_output_items(
                ResponsesRequest(model="x", input="hi", store=False),
                "hi",
                tokenizer=Mock(),
                require_reasoning=False,
            )

        self.assertTrue(parser_cls.call_args.kwargs["force_reasoning"])

    def test_require_reasoning_false_without_reasoning_parser(self):
        serving = make_serving()
        serving.reasoning_parser = None

        captured = self._capture(
            serving, ResponsesRequest(model="x", input="hi", store=False)
        )

        self.assertFalse(captured["adapted_request"].require_reasoning)

    def test_skip_special_tokens_disabled_for_tool_requests(self):
        # _process_messages turns it off so tool-call markers survive detokenize;
        # create_responses must re-apply it to the engine sampling dict.
        serving = make_serving()
        serving.tool_call_parser = "qwen25"

        captured = self._capture(
            serving,
            ResponsesRequest(
                model="x",
                input="weather",
                store=False,
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    }
                ],
            ),
        )

        self.assertFalse(captured["sampling_params"]["skip_special_tokens"])


class CancelIdempotencyTestCase(CustomTestCase):
    def test_cancelling_a_terminal_response_returns_it_not_an_error(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        for status in ("cancelled", "completed"):
            serving = make_serving()
            resp = ResponsesResponse.from_request(
                ResponsesRequest(model="x", input="hi", store=False),
                sampling_params={},
                model_name="x",
                created_time=0,
                output=[],
                status=status,
                usage=None,
            )
            serving.response_store[resp.id] = resp

            out = asyncio.run(serving.cancel_responses(resp.id))

            self.assertIs(out, resp, status)
            self.assertEqual(out.status, status)


class StreamingLogprobsRejectionTestCase(CustomTestCase):
    def test_stream_with_logprobs_include_rejected(self):
        import orjson

        serving = make_serving()
        request = ResponsesRequest(
            model="x",
            input="hi",
            store=False,
            stream=True,
            include=["message.output_text.logprobs"],
        )
        result = asyncio.run(serving.create_responses(request))
        self.assertEqual(result.status_code, 400)
        body = orjson.loads(result.body)
        self.assertIn("streaming mode", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
