# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's OpenAIServingResponses
"""Handler for /v1/responses requests"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Optional, Union

import jinja2
import openai.types.responses as openai_responses_types
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_text import Logprob, LogprobTopLogprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part as ResponseReasoningSummaryAddedPart,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    Part as ResponseReasoningSummaryDonePart,
)
from openai_harmony import Message as OpenAIMessage

from sglang.srt.entrypoints.context import (
    ConversationContext,
    HarmonyContext,
    SimpleContext,
    StreamingHarmonyContext,
)
from sglang.srt.entrypoints.harmony_utils import (
    get_developer_message,
    get_stop_tokens_for_assistant_actions,
    get_system_message,
    get_user_message,
    parse_output_message,
    parse_remaining_state,
    parse_response_input,
    render_for_completion,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Function,
    MessageProcessingResult,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.tool_server import MCPToolServer, ToolServer
from sglang.srt.entrypoints.openai.utils import to_openai_style_logprobs
from sglang.srt.function_call.core_types import MalformedToolCallError
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.sampling.sampling_params import (
    set_request_reasoning_end_token_ids,
)
from sglang.srt.utils import random_uuid

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)

_TOOL_RECOVERY_ENV = "SGLANG_RESPONSES_TOOL_RECOVERY"
_TOOL_RECOVERY_TIMEOUT_ENV = "SGLANG_RESPONSES_TOOL_RECOVERY_TIMEOUT_S"
_TOOL_RECOVERY_MAX_TOKENS_ENV = "SGLANG_RESPONSES_TOOL_RECOVERY_MAX_TOKENS"
_TOOL_RECOVERY_CONCURRENCY_ENV = "SGLANG_RESPONSES_TOOL_RECOVERY_CONCURRENCY"
_TOOL_RECOVERY_PING_INTERVAL_ENV = "SGLANG_RESPONSES_TOOL_RECOVERY_PING_INTERVAL_S"
_TOOL_FAILURE_DUMP_DIR_ENV = "SGLANG_RESPONSES_TOOL_FAILURE_DUMP_DIR"
_TOOL_FAILURE_TAIL_CHARS_ENV = "SGLANG_RESPONSES_TOOL_FAILURE_TAIL_CHARS"


@dataclass(frozen=True)
class _RecoveredResponseToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class _ResponsesToolRecoveryResult:
    calls: tuple[_RecoveredResponseToolCall, ...]
    usage: Any
    recovery_rid: str


def _bounded_positive_int_env(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    if value < 1 or value > maximum:
        logger.warning(
            "Ignoring out-of-range %s=%r (expected 1..%d); using %d",
            name,
            raw,
            maximum,
            default,
        )
        return default
    return value


def _response_tool_failure_reason(message: str) -> Optional[str]:
    normalized = (message or "").lower()
    if "arguments that do not match the declared schema" in normalized:
        return "schema_mismatch"
    if "undeclared tool" in normalized:
        return "undeclared_tool"
    if "non-object" in normalized and "tool-call" in normalized:
        return "non_object_arguments"
    if "malformed tool-call arguments" in normalized:
        return "malformed_arguments"
    if "malformed deepseek tool-call parameters" in normalized:
        return "malformed_arguments"
    if "invalid tool-call stream ordering" in normalized:
        return "stream_ordering"
    if "incomplete tool-call stream" in normalized:
        return "incomplete_stream"
    if "malformed" in normalized and "tool" in normalized:
        return "malformed_protocol"
    return None


def _response_failure_error(exc: Exception) -> dict[str, str]:
    message = str(exc)
    reason = _response_tool_failure_reason(message)
    if reason is None:
        return {
            "code": "server_error",
            "message": "Internal server error",
            "sglang_code": "internal_error",
        }
    return {
        "code": "server_error",
        "message": message,
        "sglang_code": reason,
    }


class _MediaInputValidationError(ValueError):
    pass


def _build_output_text_logprobs(meta_info: dict) -> list[Logprob]:
    """Reshape decoded ``meta_info`` logprobs into the Responses logprob type,
    covering every generated token."""
    decoded = to_openai_style_logprobs(
        output_token_logprobs=meta_info.get("output_token_logprobs"),
        output_top_logprobs=meta_info.get("output_top_logprobs"),
    )
    top_lists = decoded.top_logprobs or []
    logprobs: list[Logprob] = []
    for index, (token, logprob) in enumerate(
        zip(decoded.tokens, decoded.token_logprobs)
    ):
        top_entry = top_lists[index] if index < len(top_lists) else None
        top_logprobs = [
            LogprobTopLogprob(
                token=top_token,
                logprob=top_logprob,
                bytes=list(top_token.encode("utf-8")),
            )
            for top_token, top_logprob in (top_entry or {}).items()
        ]
        logprobs.append(
            Logprob(
                token=token,
                logprob=logprob,
                bytes=list(token.encode("utf-8")),
                top_logprobs=top_logprobs,
            )
        )
    return logprobs


def _should_emit_normal_text_as_message(
    text: str, *, any_tool_call_in_progress: bool
) -> bool:
    """Whether ``text`` should open / extend a user-visible message item.

    qwen3-coder separates adjacent tool-call blocks with ``\\n``, which the
    streaming detector cannot tell from real content -- so whitespace arriving
    while a call is open is treated as a separator.
    """
    if not text:
        return False
    if any_tool_call_in_progress and not text.strip():
        return False
    return True


class OpenAIServingResponses(OpenAIServingChat):
    """Handler for /v1/responses requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
        *,
        enable_prompt_tokens_details: bool = False,
        tool_server: Optional[ToolServer] = None,
    ) -> None:
        super().__init__(tokenizer_manager, template_manager)

        # template_manager is already set by parent class; reasoning_parser comes
        # from the parent, which reads the manager's control-plane overlay.
        self.enable_prompt_tokens_details = enable_prompt_tokens_details

        # Parent OpenAIServingChat.__init__ already populated default_sampling_params.
        if not isinstance(self.default_sampling_params, dict):
            self.default_sampling_params = {}

        self.supports_browsing = (
            tool_server.has_tool("browser") if tool_server else False
        )
        self.supports_code_interpreter = (
            tool_server.has_tool("python") if tool_server else False
        )
        self.tool_server = tool_server
        # Get from model config
        self.use_harmony = (
            self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        if self.use_harmony:
            # OpenAI models have two EOS-like tokens: <|return|> and <|call|>.
            # We need to add them to the stop token ids.
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        # Response storage for background and retrieval operations
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # Message storage for conversation continuity
        # Note: In production, this should use a proper storage backend (Redis, database)
        # with TTL/expiration to prevent memory leaks
        self.msg_store: dict[
            str, Union[list[ChatCompletionMessageParam], list[OpenAIMessage]]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}

        recovery_flag = (
            os.getenv(
                _TOOL_RECOVERY_ENV,
                os.getenv("SGLANG_ANTHROPIC_TOOL_RECOVERY", "0"),
            )
            .strip()
            .lower()
        )
        self._tool_recovery_enabled = recovery_flag not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._tool_recovery_timeout_s = _bounded_positive_int_env(
            _TOOL_RECOVERY_TIMEOUT_ENV,
            default=180,
            maximum=3600,
        )
        self._tool_recovery_max_tokens = _bounded_positive_int_env(
            _TOOL_RECOVERY_MAX_TOKENS_ENV,
            default=8192,
            maximum=131072,
        )
        recovery_concurrency = _bounded_positive_int_env(
            _TOOL_RECOVERY_CONCURRENCY_ENV,
            default=2,
            maximum=64,
        )
        self._tool_recovery_ping_interval_s = _bounded_positive_int_env(
            _TOOL_RECOVERY_PING_INTERVAL_ENV,
            default=10,
            maximum=60,
        )
        self._tool_recovery_slots = asyncio.Semaphore(recovery_concurrency)
        failure_dump_dir = os.getenv(_TOOL_FAILURE_DUMP_DIR_ENV)
        self._tool_failure_dump_dir = (
            pathlib.Path(failure_dump_dir) if failure_dump_dir else None
        )
        self._tool_failure_tail_chars = _bounded_positive_int_env(
            _TOOL_FAILURE_TAIL_CHARS_ENV,
            default=8192,
            maximum=65536,
        )
        logger.info(
            "Responses bounded tool-call recovery: %s timeout_s=%d "
            "max_tokens=%d concurrency=%d ping_interval_s=%d failure_dump_dir=%s",
            "enabled" if self._tool_recovery_enabled else "disabled",
            self._tool_recovery_timeout_s,
            self._tool_recovery_max_tokens,
            recovery_concurrency,
            self._tool_recovery_ping_interval_s,
            self._tool_failure_dump_dir or "disabled",
        )

    @staticmethod
    def _has_response_tool(request: ResponsesRequest, *tool_types: str) -> bool:
        return any(tool.type in tool_types for tool in (request.tools or []))

    @staticmethod
    def _validate_response_tool_input(
        request: ResponsesRequest,
        name: str,
        tool_input: dict[str, Any],
        *,
        enforce_schema: bool = False,
    ) -> None:
        declarations = {
            tool.name: tool
            for tool in (request.tools or [])
            if tool.type == "function" and tool.name
        }
        declaration = declarations.get(name)
        if declaration is None:
            raise MalformedToolCallError("Model attempted to call an undeclared tool")

        schema = declaration.parameters or {"type": "object"}
        try:
            Draft202012Validator(schema).validate(tool_input)
        except JSONSchemaValidationError as e:
            if enforce_schema or declaration.strict:
                raise MalformedToolCallError(
                    "Model emitted tool-call arguments that do not match the "
                    "declared schema"
                ) from e
            logger.warning(
                "Publishing schema-invalid arguments for non-strict Responses "
                "tool %r; the client may return a tool error",
                name,
            )
        except (JSONSchemaError, RecursionError) as e:
            raise MalformedToolCallError("Declared tool schema is invalid") from e

    @classmethod
    def _validated_response_recovery_calls(
        cls,
        response: ChatCompletionResponse,
        request: ResponsesRequest,
        recovery_rid: str,
        allowed_tool_names: Optional[set[str]] = None,
    ) -> _ResponsesToolRecoveryResult:
        if not response.choices or not response.choices[0].message.tool_calls:
            raise MalformedToolCallError(
                "Constrained tool-call recovery produced no tool call"
            )
        if response.choices[0].finish_reason not in {"tool_calls", "stop"}:
            raise MalformedToolCallError(
                "Constrained tool-call recovery ended before completing tool calls"
            )

        recovered: list[_RecoveredResponseToolCall] = []
        for tool_call in response.choices[0].message.tool_calls:
            name = tool_call.function.name
            if allowed_tool_names is not None and name not in allowed_tool_names:
                raise MalformedToolCallError(
                    "Constrained tool-call recovery emitted a tool outside its "
                    "allowed intent set"
                )
            raw_arguments = tool_call.function.arguments
            try:
                if isinstance(raw_arguments, dict):
                    tool_input = raw_arguments
                else:
                    tool_input = json.loads(raw_arguments) if raw_arguments else {}
            except (json.JSONDecodeError, TypeError) as e:
                raise MalformedToolCallError(
                    "Constrained tool-call recovery emitted malformed arguments"
                ) from e
            if not isinstance(tool_input, dict):
                raise MalformedToolCallError(
                    "Constrained tool-call recovery emitted non-object arguments"
                )

            cls._validate_response_tool_input(request, name, tool_input)
            recovered.append(
                _RecoveredResponseToolCall(
                    call_id=tool_call.id or f"call_{random_uuid()[:24]}",
                    name=name,
                    arguments=json.dumps(
                        tool_input,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

        return _ResponsesToolRecoveryResult(
            calls=tuple(recovered),
            usage=response.usage,
            recovery_rid=recovery_rid,
        )

    async def _attempt_response_tool_recovery(
        self,
        processed_request: Optional[ChatCompletionRequest],
        request: ResponsesRequest,
        raw_request: Optional[Request],
        reason: str,
        original_rid: Optional[str],
        candidate_tool_names: tuple[str, ...] = (),
    ) -> Optional[_ResponsesToolRecoveryResult]:
        if (
            not self._tool_recovery_enabled
            or processed_request is None
            or not processed_request.tools
            or processed_request.tool_choice == "none"
        ):
            return None
        if any(
            getattr(processed_request, field, None) is not None
            for field in (
                "bootstrap_host",
                "bootstrap_port",
                "bootstrap_room",
                "disagg_prefill_dp_rank",
            )
        ):
            logger.warning(
                "Responses tool recovery skipped for PD-disaggregated request"
            )
            return None

        while True:
            recovery_rid = f"{random_uuid()}_toolrecovery"
            if not original_rid or not recovery_rid.startswith(original_rid):
                break
        declared_names = {
            tool.function.name for tool in (processed_request.tools or [])
        }
        original_named_tool = (
            processed_request.tool_choice.function.name
            if isinstance(processed_request.tool_choice, ToolChoice)
            else None
        )
        observed_declared_names = declared_names.intersection(candidate_tool_names)
        allowed_tool_names = (
            {original_named_tool}
            if original_named_tool is not None
            else observed_declared_names or declared_names
        )

        logger.warning(
            "Responses tool recovery started: reason=%s original_rid=%s "
            "recovery_rid=%s",
            reason,
            original_rid,
            recovery_rid,
        )

        async def _run_recovery():
            async with self._tool_recovery_slots:
                recovery_request = processed_request.model_copy(deep=True)
                recovery_request.rid = recovery_rid
                recovery_request.stream = False
                recovery_request.stream_options = None
                recovery_request.n = 1
                requested_max_tokens = (
                    recovery_request.max_tokens
                    or recovery_request.max_completion_tokens
                    or self._tool_recovery_max_tokens
                )
                recovery_request.max_tokens = min(
                    requested_max_tokens,
                    self._tool_recovery_max_tokens,
                )
                recovery_request.max_completion_tokens = None
                recovery_request.session_id = None
                recovery_request.session_params = None

                if original_named_tool is not None:
                    recovery_request.tools = [
                        tool
                        for tool in (recovery_request.tools or [])
                        if tool.function.name == original_named_tool
                    ]
                elif observed_declared_names:
                    recovery_request.tools = [
                        tool
                        for tool in (recovery_request.tools or [])
                        if tool.function.name in observed_declared_names
                    ]

                if original_named_tool is not None:
                    recovery_request.tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(name=original_named_tool),
                    )
                elif len(observed_declared_names) == 1:
                    candidate_tool_name = next(iter(observed_declared_names))
                    recovery_request.tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(name=candidate_tool_name),
                    )
                elif len(observed_declared_names) > 1:
                    recovery_request.tool_choice = "required"
                elif not isinstance(recovery_request.tool_choice, ToolChoice):
                    recovery_request.tool_choice = (
                        "required"
                        if recovery_request.tool_choice == "required"
                        else "auto"
                    )
                for tool in recovery_request.tools or []:
                    tool.function = tool.function.model_copy(update={"strict": True})

                adapted_request, normalized_request = self._convert_to_internal_request(
                    recovery_request, raw_request
                )
                if any(
                    getattr(adapted_request, field, None) is not None
                    for field in (
                        "bootstrap_host",
                        "bootstrap_port",
                        "bootstrap_room",
                        "disagg_prefill_dp_rank",
                    )
                ):
                    raise ValueError(
                        "Tool recovery is not supported with PD disaggregation"
                    )
                adapted_request.rid = recovery_rid
                adapted_request.received_time = monotonic_time()
                return await self._handle_non_streaming_request(
                    adapted_request,
                    normalized_request,
                    raw_request,
                )

        try:
            response = await asyncio.wait_for(
                _run_recovery(),
                timeout=self._tool_recovery_timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning(
                "Responses tool recovery timed out: reason=%s recovery_rid=%s "
                "timeout_s=%d",
                reason,
                recovery_rid,
                self._tool_recovery_timeout_s,
            )
            abort_request = getattr(self.tokenizer_manager, "abort_request", None)
            if abort_request is not None:
                try:
                    abort_request(recovery_rid)
                except Exception:
                    logger.exception(
                        "Failed to abort timed-out Responses recovery rid=%s",
                        recovery_rid,
                    )
            return None
        except Exception as e:
            logger.exception(
                "Responses tool recovery raised: reason=%s recovery_rid=%s "
                "exception=%s",
                reason,
                recovery_rid,
                type(e).__name__,
            )
            return None

        if not isinstance(response, ChatCompletionResponse):
            logger.warning(
                "Responses tool recovery failed upstream: reason=%s "
                "recovery_rid=%s response_type=%s",
                reason,
                recovery_rid,
                type(response).__name__,
            )
            return None
        try:
            result = self._validated_response_recovery_calls(
                response,
                request,
                recovery_rid,
                allowed_tool_names,
            )
        except MalformedToolCallError as e:
            logger.warning(
                "Responses tool recovery rejected: reason=%s recovery_rid=%s "
                "validation=%s",
                reason,
                recovery_rid,
                e,
            )
            return None

        logger.warning(
            "Responses tool recovery succeeded: reason=%s original_rid=%s "
            "recovery_rid=%s tool_calls=%d",
            reason,
            original_rid,
            recovery_rid,
            len(result.calls),
        )
        return result

    def _write_tool_failure_dump(
        self,
        request_id: str,
        exc: Exception,
        raw_tail: str,
    ) -> None:
        if self._tool_failure_dump_dir is None:
            return
        tail = raw_tail[-self._tool_failure_tail_chars :]
        payload = {
            "captured_at_unix_ns": time.time_ns(),
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "error": _response_failure_error(exc),
            "tail_chars": len(tail),
            "tail_sha256": hashlib.sha256(tail.encode("utf-8")).hexdigest(),
            "raw_pre_parser_tail": tail,
        }
        try:
            self._tool_failure_dump_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            dump_path = self._tool_failure_dump_dir / (
                f"{time.time_ns()}-{random_uuid()}.json"
            )
            descriptor = os.open(
                dump_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as dump_file:
                json.dump(payload, dump_file, ensure_ascii=False)
            logger.warning(
                "Captured Responses parser failure tail: path=%s chars=%d sha256=%s",
                dump_path,
                len(tail),
                payload["tail_sha256"],
            )
        except Exception:
            logger.exception("Failed to capture Responses parser failure tail")

    # error helpers dedicated for v1/responses
    def create_error_response(
        self,
        message: str,
        err_type: str = "invalid_request_error",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        nested_error = {
            "message": message,
            "type": err_type,
            "param": param,
            "code": status_code,
        }
        return ORJSONResponse(content={"error": nested_error}, status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        return json.dumps(
            {
                "error": {
                    "message": message,
                    "type": err_type,
                    "param": None,
                    "code": status_code,
                }
            }
        )

    def _request_id_prefix(self) -> str:
        return "resp_"

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ResponsesResponse, ORJSONResponse]:
        # Validate model
        if not self.tokenizer_manager:
            return self.create_error_response("Model not loaded")

        # FIXME: If the engine is dead, raise an error
        # This is required for the streaming case

        # ``tool_choice="required"`` only works with ``function`` tools.
        if request.tool_choice == "required" and not any(
            tool.type == "function" for tool in (request.tools or [])
        ):
            return self.create_error_response(
                'tool_choice="required" requires at least one tool with '
                'type="function"; other built-in tool types cannot be forced.'
            )

        # harmony emits raw tokens; per-token logprobs aren't wired there.
        if self.use_harmony and request.is_include_output_logprobs():
            return self.create_error_response(
                "logprobs are not supported with gpt-oss models", param="logprobs"
            )
        # streaming skips the logprobs build path; reject so the include doesn't silently no-op.
        if request.stream and request.is_include_output_logprobs():
            return self.create_error_response(
                "logprobs are not supported in streaming mode", param="logprobs"
            )
        # harmony output opens with <|channel|>analysis<|message|>, so a whole-output
        # json_schema forces "{" at the first token and the harmony parse then fails.
        if self.use_harmony and request.has_json_schema_constraint():
            return self.create_error_response(
                "structured output (text.format) is not supported with gpt-oss models",
                param="text",
            )
        if (
            self.use_harmony
            and self._has_response_tool(request, "web_search", "web_search_preview")
            and not self.supports_browsing
        ):
            return self.create_error_response(
                "web_search requires a browser backend. Set EXA_API_KEY on the "
                "SGLang server to enable native Exa-backed web search, or "
                "configure a browser MCP tool server. Create an Exa API key at "
                "https://dashboard.exa.ai/api-keys."
            )

        # Handle the previous response ID
        prev_response_id = request.previous_response_id
        if prev_response_id is not None:
            if not prev_response_id.startswith("resp_"):
                return self._make_invalid_id_error(prev_response_id)
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        try:
            model_name = request.model
            tokenizer = self.tokenizer_manager.tokenizer
            processed_messages: Optional[MessageProcessingResult] = None
            processed_chat_request: Optional[ChatCompletionRequest] = None

            if self.use_harmony:
                messages, request_prompts, engine_prompts = (
                    self._make_request_with_harmony(request, prev_response)
                )
                require_reasoning = self._is_thinking_enabled_for_request(request)
            else:
                (
                    messages,
                    request_prompts,
                    engine_prompts,
                    processed_messages,
                    processed_chat_request,
                ) = await self._make_request(request, prev_response, tokenizer)
                require_reasoning = processed_messages.require_reasoning

        except _MediaInputValidationError as e:
            return self.create_error_response(str(e))
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        request_metadata = RequestResponseMetadata(request_id=request.request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        if (
            self.tool_server is not None
            and isinstance(self.tool_server, MCPToolServer)
            and (request.background or request.stream)
            and request.tools
            and any(
                tool.type in ("web_search", "web_search_preview", "code_interpreter")
                for tool in request.tools
            )
        ):
            return self.create_error_response(
                "MCP tool server is not supported in background mode and streaming mode"
            )

        # Schedule the request and get the result generator
        generators: list[AsyncGenerator[Any, None]] = []
        tool_list = []
        if self.use_harmony:
            if self.supports_browsing:
                tool_list.append("browser")
            if self.supports_code_interpreter:
                tool_list.append("python")
        async with AsyncExitStack() as exit_stack:
            try:
                if self.tool_server is not None:
                    tool_session_ctxs: dict[str, Any] = {
                        tool_name: exit_stack.enter_async_context(
                            self.tool_server.get_tool_session(tool_name)
                        )
                        for tool_name in tool_list
                    }
                    tool_sessions = {}
                    for tool_name in tool_list:
                        tool_sessions[tool_name] = await tool_session_ctxs[tool_name]
                else:
                    assert len(tool_list) == 0
                    tool_sessions = {}
                for i, engine_prompt in enumerate(engine_prompts):
                    # Calculate default max tokens from context length minus prompt length
                    if isinstance(engine_prompt, list):
                        prompt_length = len(engine_prompt)
                    elif isinstance(engine_prompt, str):
                        prompt_length = len(tokenizer.encode(engine_prompt))
                    else:
                        prompt_length = 0

                    context_len = (
                        self.tokenizer_manager.model_config.context_len
                        if hasattr(self.tokenizer_manager.model_config, "context_len")
                        else 4096
                    )
                    # Account for reserved tokens (e.g., EAGLE speculative decoding slots)
                    # that the tokenizer_manager adds during validation
                    num_reserved_tokens = self.tokenizer_manager.num_reserved_tokens
                    default_max_tokens = max(
                        context_len - prompt_length - num_reserved_tokens, 512
                    )  # Ensure minimum 512 tokens
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.default_sampling_params,
                        stop=(
                            processed_messages.stop
                            if processed_messages
                            else request.stop
                        ),
                        tool_call_constraint=(
                            processed_messages.tool_call_constraint
                            if processed_messages
                            else None
                        ),
                    )
                    if processed_messages is not None:
                        set_request_reasoning_end_token_ids(
                            sampling_params,
                            processed_messages.reasoning_end_token_ids,
                        )
                    # _process_messages set skip_special_tokens on a chat_request
                    # we then discard, so re-apply it to the engine sampling dict.
                    if processed_messages is not None and (
                        not processed_messages.skip_special_tokens
                    ):
                        sampling_params["skip_special_tokens"] = False

                    context: ConversationContext
                    if self.use_harmony:
                        if request.stream:
                            context = StreamingHarmonyContext(messages, tool_sessions)
                        else:
                            context = HarmonyContext(messages, tool_sessions)
                    else:
                        context = SimpleContext()

                    # Create GenerateReqInput for SGLang
                    if isinstance(engine_prompt, str):
                        prompt_kwargs = {"text": engine_prompt}
                    else:
                        prompt_kwargs = {"input_ids": engine_prompt}

                    logprob_kwargs = (
                        {
                            "return_logprob": True,
                            "logprob_start_len": -1,
                            "top_logprobs_num": request.top_logprobs or 0,
                            "return_text_in_logprobs": True,
                        }
                        if request.is_include_output_logprobs()
                        else {}
                    )

                    adapted_request = GenerateReqInput(
                        **prompt_kwargs,
                        **logprob_kwargs,
                        image_data=(
                            processed_messages.image_data
                            if processed_messages
                            else None
                        ),
                        video_data=(
                            processed_messages.video_data
                            if processed_messages
                            else None
                        ),
                        audio_data=(
                            processed_messages.audio_data
                            if processed_messages
                            else None
                        ),
                        modalities=(
                            processed_messages.modalities
                            if processed_messages
                            else None
                        ),
                        sampling_params=sampling_params,
                        stream=request.stream,
                        rid=request.request_id,
                        session_id=request.session_id,
                        extra_key=request.extra_key,
                        cache_salt=request.cache_salt,
                        # background+stream streams on this connection, so don't detach.
                        background=request.background and not request.stream,
                        require_reasoning=require_reasoning,
                    )

                    generator = self._generate_with_builtin_tools(
                        request.request_id,
                        request_prompts[i],
                        adapted_request,
                        sampling_params,
                        context,
                        raw_request=raw_request,
                        priority=request.priority,
                    )
                    generators.append(generator)
            except ValueError as e:
                return self.create_error_response(str(e))

            assert len(generators) == 1
            (result_generator,) = generators

            # Store the input messages
            if request.store:
                self.msg_store[request.request_id] = messages

            if request.background and not request.stream:
                created_time = int(time.time())
                response = ResponsesResponse.from_request(
                    request,
                    sampling_params,
                    model_name=model_name,
                    created_time=created_time,
                    output=[],
                    status="queued",
                    usage=None,
                )
                async with self.response_store_lock:
                    self.response_store[response.id] = response

                # Run the request in the background
                task = asyncio.create_task(
                    self._run_background_request(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                        require_reasoning=require_reasoning,
                    ),
                    name=f"create_{response.id}",
                )

                # For cleanup
                self.background_tasks[response.id] = task
                task.add_done_callback(
                    lambda _: self.background_tasks.pop(response.id, None)
                )
                return response

            if request.stream:
                if self.use_harmony:
                    return self.responses_stream_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        require_reasoning=require_reasoning,
                    )
                return self.responses_stream_generator_non_harmony(
                    request,
                    sampling_params,
                    result_generator,
                    model_name,
                    tokenizer,
                    request_metadata,
                    require_reasoning=require_reasoning,
                    processed_request=processed_chat_request,
                    raw_request=raw_request,
                )
            try:
                result: Union[ORJSONResponse, ResponsesResponse] = (
                    await self.responses_full_generator(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        require_reasoning=require_reasoning,
                    )
                )
                return result
            except Exception as e:
                return self.create_error_response(str(e))
        return self.create_error_response("Unknown error")

    async def _make_request(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
        tokenizer: Any,
    ):
        messages = self._construct_input_messages(request, prev_response)

        chat_tools = self._response_tools_to_chat_tools(request)
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=messages,
            stream=request.stream,
            tools=chat_tools or None,
            tool_choice=(
                self._chat_tool_choice(request.effective_tool_choice())
                if chat_tools
                else "none"
            ),
            parallel_tool_calls=(
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            stop=request.stop,
            max_tokens=request.max_output_tokens,
            reasoning_effort=(request.reasoning.effort if request.reasoning else None),
            chat_template_kwargs=request.chat_template_kwargs,
        )

        media_error = self._validate_media_content(chat_request)
        if media_error:
            raise _MediaInputValidationError(media_error)

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed_messages = self._process_messages(chat_request, is_multimodal)
        # ``_process_messages`` merges server defaults into the temporary Chat
        # request before rendering. Response parsing happens later from the
        # original request, so carry over the exact template kwargs that selected
        # the wire-format delimiters.
        request.chat_template_kwargs = (
            dict(chat_request.chat_template_kwargs)
            if chat_request.chat_template_kwargs is not None
            else None
        )

        engine_prompt = self._select_engine_prompt(processed_messages, is_multimodal)
        request_prompts = [engine_prompt]
        engine_prompts = [engine_prompt]

        return (
            messages,
            request_prompts,
            engine_prompts,
            processed_messages,
            chat_request,
        )

    def _make_request_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ):
        if request.tool_choice != "auto":
            raise NotImplementedError(
                "Only 'auto' tool_choice is supported in response API"
            )
        messages = self._construct_input_messages_with_harmony(request, prev_response)
        prompt_token_ids = render_for_completion(messages)
        engine_prompt = prompt_token_ids
        return messages, [prompt_token_ids], [engine_prompt]

    async def responses_full_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *,
        require_reasoning: bool,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if created_time is None:
            created_time = int(time.time())

        try:
            async for _ in result_generator:
                pass
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        status = "completed"
        if self.use_harmony:
            assert isinstance(context, HarmonyContext)
            output = self._make_response_output_items_with_harmony(context)
            # num_reasoning_tokens isn't wired through HarmonyContext yet; stays 0.
            num_prompt_tokens = context.num_prompt_tokens
            num_generated_tokens = context.num_output_tokens
            num_cached_tokens = context.num_cached_tokens
            num_reasoning_tokens = context.num_reasoning_tokens
            status = self._status_from_finish_reason(context.finish_reason)
        else:
            assert isinstance(context, SimpleContext)
            final_res = context.last_output
            assert final_res is not None

            num_reasoning_tokens = 0
            meta_info = None
            if isinstance(final_res, dict) and isinstance(
                final_res.get("meta_info"), dict
            ):
                meta_info = final_res["meta_info"]
            elif hasattr(final_res, "meta_info"):
                meta_info = final_res.meta_info

            output_logprobs = (
                _build_output_text_logprobs(meta_info)
                if request.is_include_output_logprobs() and isinstance(meta_info, dict)
                else None
            )
            output = self._make_response_output_items(
                request,
                final_res["text"],
                tokenizer,
                output_logprobs=output_logprobs,
                require_reasoning=require_reasoning,
                finish_reason=(meta_info or {}).get("finish_reason"),
            )

            if meta_info is not None:
                num_prompt_tokens = meta_info.get("prompt_tokens", 0)
                num_generated_tokens = meta_info.get("completion_tokens", 0)
                num_cached_tokens = meta_info.get("cached_tokens", 0)
                num_reasoning_tokens = meta_info.get("reasoning_tokens", 0)
                status = self._status_from_finish_reason(meta_info.get("finish_reason"))
            elif isinstance(final_res, dict) and (
                final_res.get("prompt_token_ids") is not None
                or final_res.get("output_ids") is not None
            ):
                prompt_token_ids = final_res.get("prompt_token_ids") or []
                output_token_ids = final_res.get("output_ids") or []
                num_prompt_tokens = len(prompt_token_ids)
                num_generated_tokens = len(output_token_ids)
                num_cached_tokens = final_res.get("num_cached_tokens", 0)
            elif hasattr(final_res, "prompt_token_ids") and hasattr(
                final_res, "outputs"
            ):
                # Fallback calculation if meta_info not available
                num_prompt_tokens = (
                    len(final_res.prompt_token_ids) if final_res.prompt_token_ids else 0
                )
                num_generated_tokens = (
                    len(final_res.outputs[0].token_ids)
                    if final_res.outputs and final_res.outputs[0].token_ids
                    else 0
                )
                num_cached_tokens = getattr(final_res, "num_cached_tokens", 0)
            else:
                # Final fallback
                num_prompt_tokens = 0
                num_generated_tokens = 0
                num_cached_tokens = 0
                num_reasoning_tokens = 0

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            reasoning_tokens=num_reasoning_tokens,
        )
        if self.enable_prompt_tokens_details and num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=num_cached_tokens
            )
        request_metadata.final_usage_info = usage

        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=output,
            status=status,
            usage=usage,
        )

        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response

        return response

    @staticmethod
    def _wants_reasoning_summary(request: ResponsesRequest) -> bool:
        return request.reasoning is not None and request.reasoning.summary is not None

    @staticmethod
    def _status_from_finish_reason(finish_reason: Any) -> str:
        """Only a length-capped generation is ``incomplete``; anything that got
        here otherwise finished normally."""
        reason = None
        if isinstance(finish_reason, dict):
            reason = finish_reason.get("type")
        elif isinstance(finish_reason, str):
            reason = finish_reason
        return "incomplete" if reason == "length" else "completed"

    def _is_thinking_enabled_for_request(self, request: ResponsesRequest) -> bool:
        if not self.reasoning_parser:
            return False
        # an explicit toggle wins; the key differs by family (enable_thinking vs thinking).
        ctk = request.chat_template_kwargs or {}
        thinking_toggles = (ctk.get("enable_thinking"), ctk.get("thinking"))
        if any(toggle is False for toggle in thinking_toggles):
            return False
        if any(toggle is True for toggle in thinking_toggles):
            return True
        effort = request.reasoning.effort if request.reasoning is not None else None
        if self.reasoning_parser == "hunyuan":
            return effort not in (None, "none", "no_think")
        if self.template_manager.force_reasoning:
            return True
        config = self.template_manager.reasoning_config
        if config is None:
            # Parser-only models (DeepSeek-R1, …) carry the thinking default in
            # the detector itself.
            detector = getattr(self, "_reasoning_detector", None)
            mode = getattr(detector, "reasoning_default", None) if detector else None
            if mode is None or mode == "always":
                return mode == "always"
            if mode == "mistral":
                return effort is not None and effort != "none"
            if mode in ("thinking", "enable_thinking"):
                return effort != "none"
            if mode in ("explicit_thinking", "explicit_enable_thinking"):
                return False
            return False
        if config.special_case == "always":
            return True
        if config.special_case == "mistral":
            return effort is not None and effort != "none"
        if config.toggle_param is None or config.default_enabled is None:
            return False
        if effort == "none":
            return False
        return bool(config.default_enabled)

    def _make_response_output_items(
        self,
        request: ResponsesRequest,
        final_output: Any,
        tokenizer: Any,
        output_logprobs: Optional[list] = None,
        *,
        require_reasoning: bool,
        finish_reason: Optional[dict[str, Any]] = None,
    ):
        chat_tools = self._response_tools_to_chat_tools(request)
        if self.reasoning_parser:
            reasoning_parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                # A template that prefills <think> forces the parser open even
                # when the request itself did not ask for reasoning, same as chat.
                force_reasoning=(
                    self.template_manager.force_reasoning or require_reasoning
                ),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
                tool_call_parser_active=bool(
                    chat_tools
                    and self.tool_call_parser
                    and request.tool_choice != "none"
                ),
            )
            reasoning_content, content = reasoning_parser.parse_non_stream(final_output)
        else:
            reasoning_content = None
            content = final_output

        output_items = []
        if reasoning_content:
            # Mirror the single parsed blob into ``summary`` when the caller opts
            # in via ``reasoning.summary``; full trace stays in ``content``.
            wants_summary = self._wants_reasoning_summary(request)
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                type="reasoning",
                summary=(
                    [
                        ResponseReasoningSummary(
                            type="summary_text", text=reasoning_content
                        )
                    ]
                    if wants_summary
                    else []
                ),
                content=[
                    ResponseReasoningTextContent(
                        type="reasoning_text", text=reasoning_content
                    ),
                ],
                status=None,
            )
            output_items.append(reasoning_item)

        is_required = request.tool_choice == "required"
        tool_call_items: list[ResponseFunctionToolCall] = []
        parsed_via_native = False
        detector_owns_format = False
        if (
            content
            and chat_tools
            and self.tool_call_parser
            and request.tool_choice != "none"
        ):
            parser = FunctionCallParser(
                chat_tools,
                self.tool_call_parser,
                tokenizer=self.tokenizer_manager.tokenizer,
            )
            detector_owns_format = (
                parser.detector.supports_structural_tag()
                or parser.detector.parses_required_natively()
            )
            should_try_native = not is_required or detector_owns_format
            if should_try_native and parser.has_tool_call(content):
                try:
                    content, call_info_list = parser.parse_non_stream(content)
                    for call_info in call_info_list:
                        tool_call_items.append(
                            ResponseFunctionToolCall(
                                arguments=call_info.parameters or "",
                                call_id=f"call_{random_uuid()[:24]}",
                                type="function_call",
                                name=call_info.name,
                                id=f"fc_{random_uuid()[:8]}",
                                status="completed",
                            )
                        )
                    parsed_via_native = bool(call_info_list)
                except MalformedToolCallError:
                    raise
                except Exception as e:
                    logger.error("Tool call parsing error: %s", e)

        if (
            content
            and chat_tools
            and is_required
            and not parsed_via_native
            and not detector_owns_format
        ):
            try:
                tool_call_data = orjson.loads(content)
                if isinstance(tool_call_data, dict):
                    tool_call_data = [tool_call_data]
                if isinstance(tool_call_data, list):
                    for tool in tool_call_data:
                        if not isinstance(tool, dict) or "name" not in tool:
                            continue
                        arguments = json.dumps(
                            tool.get("parameters", {}), ensure_ascii=False
                        )
                        tool_call_items.append(
                            ResponseFunctionToolCall(
                                arguments=arguments,
                                call_id=f"call_{random_uuid()[:24]}",
                                type="function_call",
                                name=tool["name"],
                                id=f"fc_{random_uuid()[:8]}",
                                status="completed",
                            )
                        )
                    content = ""
            except Exception as e:
                logger.error("Required tool JSON parse error: %s", e)

        if content:
            output_text = ResponseOutputText(
                text=content,
                annotations=[],  # TODO
                type="output_text",
                # logprobs cover all generated tokens, not just the stripped content.
                logprobs=output_logprobs,
            )
            message = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
                **(
                    {
                        "phase": (
                            "final_answer"
                            if not tool_call_items
                            and (finish_reason or {}).get("type") == "stop"
                            else "commentary"
                        )
                    }
                    if self.chat_encoding_spec == "dsv41"
                    else {}
                ),
            )
            output_items.append(message)
        output_items.extend(tool_call_items)
        return output_items

    def _make_response_output_items_with_harmony(
        self,
        context: HarmonyContext,
    ):
        output_items = []
        num_init_messages = context.num_init_messages
        for msg in context.messages[num_init_messages:]:
            output_items.extend(parse_output_message(msg))
        # Handle the generation stopped in the middle (if any).
        last_items = parse_remaining_state(context.parser)
        if last_items:
            output_items.extend(last_items)
        return output_items

    @staticmethod
    def _chat_tool_choice(tool_choice: Any) -> Any:
        """Nest an ``effective_tool_choice()`` result the way chat expects:
        ``{"type":"function","name":X}`` -> ``{...,"function":{"name":X}}``."""
        if not isinstance(tool_choice, dict):
            return tool_choice
        return {"type": "function", "function": {"name": tool_choice["name"]}}

    @staticmethod
    def _response_tools_to_chat_tools(request: ResponsesRequest) -> list[Tool]:
        # Only ``function`` tools flow to chat; built-ins go through harmony.
        chat_tools = []
        for tool in request.tools:
            if tool.type != "function":
                continue
            chat_tools.append(
                Tool(
                    type="function",
                    function=Function(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                        strict=tool.strict,
                    ),
                )
            )
        return chat_tools

    @staticmethod
    def _normalize_response_content_part_for_chat(content_part: Any) -> Any:
        # Default detail=\"auto\" and lift flat min/max_dynamic_patch onto
        # image_url so the image preprocessor sees them.
        if hasattr(content_part, "model_dump"):
            content_part = content_part.model_dump(exclude_none=True)
        if not isinstance(content_part, dict):
            return content_part

        part_type = content_part.get("type")
        if part_type in ("input_text", "output_text"):
            return {"type": "text", "text": content_part.get("text", "")}

        if part_type == "input_image":
            image_url = content_part.get("image_url")
            if isinstance(image_url, dict):
                image_url_obj = image_url.copy()
            else:
                image_url_obj = {"url": image_url}
            if not image_url_obj.get("detail"):
                image_url_obj["detail"] = content_part.get("detail") or "auto"
            for key in ("min_dynamic_patch", "max_dynamic_patch"):
                if key in content_part and key not in image_url_obj:
                    image_url_obj[key] = content_part[key]
            return {"type": "image_url", "image_url": image_url_obj}

        if part_type == "text":
            return content_part

        if part_type == "image_url":
            image_url = content_part.get("image_url")
            if isinstance(image_url, str):
                image_url = {
                    "url": image_url,
                    "detail": content_part.get("detail", "auto"),
                }
            elif isinstance(image_url, dict):
                image_url = image_url.copy()
                if not image_url.get("detail"):
                    image_url["detail"] = content_part.get("detail") or "auto"
            return {**content_part, "image_url": image_url}

        return content_part

    @classmethod
    def _normalize_response_message_for_chat(cls, message: Any) -> Any:
        """Convert one Responses-API input item to a chat-completions message."""
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        if not isinstance(message, dict):
            return message

        # Most chat templates only recognize system/user/assistant/tool;
        # collapse ``developer`` to ``system`` at the boundary.
        if message.get("role") == "developer":
            message = {**message, "role": "system"}

        msg_type = message.get("type")
        if msg_type == "function_call":
            # Coerce ``arguments`` to a valid JSON-object string so the chat
            # template's unconditional ``orjson.loads`` survives truncated or
            # dict-shaped echoes.
            raw = message.get("arguments")
            if isinstance(raw, str):
                try:
                    parsed = orjson.loads(raw) if raw else None
                except orjson.JSONDecodeError:
                    parsed = None
                if not isinstance(parsed, dict):
                    raw = "{}"
            elif isinstance(raw, dict):
                raw = orjson.dumps(raw).decode("utf-8")
            else:
                raw = "{}"
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": message.get("call_id") or message.get("id"),
                        "type": "function",
                        "function": {
                            "name": message.get("name"),
                            "arguments": raw,
                        },
                    }
                ],
            }
        if msg_type == "function_call_output":
            # ``output`` may be a string or an array of content parts (OpenAI
            # allows both); the chat tool message needs a string, so flatten.
            out = message.get("output", "")
            if isinstance(out, list):
                out = "".join(p.get("text", "") for p in out if isinstance(p, dict))
            return {
                "role": "tool",
                "tool_call_id": message.get("call_id"),
                "content": out,
            }
        # Reasoning items render as {role: assistant, reasoning_content};
        # empty ones drop instead of injecting an empty assistant block.
        if msg_type == "reasoning":
            # Prefer ``summary``; fall back to ``content`` only when summary
            # is empty, since clients often populate both with the same text.
            def _collect(parts):
                out: list[str] = []
                for entry in parts or []:
                    if isinstance(entry, dict):
                        text = entry.get("text")
                        if text:
                            out.append(text)
                return out

            text_parts = _collect(message.get("summary"))
            if not text_parts:
                text_parts = _collect(message.get("content"))
            if not text_parts:
                return None
            return {
                "role": "assistant",
                "reasoning_content": "\n".join(text_parts),
            }
        if msg_type not in (None, "message"):
            raise ValueError(f"Unsupported Responses API input item type: {msg_type!r}")

        content = message.get("content")
        if not isinstance(content, list):
            return {
                k: v
                for k, v in message.items()
                if v is not None and k not in ("id", "status", "type")
            }

        return {
            k: v
            for k, v in {
                **message,
                "content": [
                    cls._normalize_response_content_part_for_chat(part)
                    for part in content
                ],
            }.items()
            if v is not None and k not in ("id", "status", "type")
        }

    @staticmethod
    def _output_message_text(output_item: Any) -> Optional[str]:
        """Return assistant text from a ``message`` output item (joining
        ``output_text`` parts with newlines), or None for non-message items."""
        if isinstance(output_item, ResponseReasoningItem):
            return None
        if hasattr(output_item, "model_dump"):
            output_item = output_item.model_dump(exclude_none=True)
        if not isinstance(output_item, dict):
            return None
        if output_item.get("type") != "message":
            return None

        text_parts = []
        for content in output_item.get("content") or []:
            if isinstance(content, ResponseOutputText):
                text_parts.append(content.text)
                continue
            if hasattr(content, "model_dump"):
                content = content.model_dump(exclude_none=True)
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if text is not None:
                    text_parts.append(text)

        return "\n".join(text_parts) if text_parts else None

    @staticmethod
    def _merge_consecutive_assistant_messages(
        messages: list,
    ) -> list:
        """Collapse runs of consecutive ``assistant`` dicts into one entry,
        joining ``content`` and concatenating ``tool_calls`` and
        ``reasoning_content`` so a logical turn renders as a single block."""
        merged: list = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and merged
                and isinstance(merged[-1], dict)
                and merged[-1].get("role") == "assistant"
            ):
                prev = merged[-1] = dict(merged[-1])
                # Lift mixed str/list content to list parts so non-text parts
                # (e.g. image_url) survive when the two sides differ in shape.
                new_content = msg.get("content")
                if new_content is not None and new_content != "":
                    prev_content = prev.get("content")
                    if prev_content is None or prev_content == "":
                        prev["content"] = new_content
                    elif isinstance(prev_content, str) and isinstance(new_content, str):
                        sep = "\n\n" if prev_content and new_content else ""
                        prev["content"] = prev_content + sep + new_content
                    else:

                        def _as_parts(c):
                            if isinstance(c, list):
                                return list(c)
                            if isinstance(c, str) and c:
                                return [{"type": "text", "text": c}]
                            return []

                        prev["content"] = _as_parts(prev_content) + _as_parts(
                            new_content
                        )
                new_calls = msg.get("tool_calls")
                if new_calls:
                    prev_calls = prev.get("tool_calls") or []
                    prev["tool_calls"] = prev_calls + list(new_calls)
                new_reasoning = msg.get("reasoning_content")
                if new_reasoning:
                    prev_reasoning = prev.get("reasoning_content")
                    prev["reasoning_content"] = (
                        f"{prev_reasoning}\n{new_reasoning}"
                        if prev_reasoning
                        else new_reasoning
                    )
                continue
            merged.append(msg)
        return merged

    def _construct_input_messages(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse] = None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if request.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": request.instructions,
                }
            )

        # Prepend the conversation history
        if prev_response is not None:
            # Add the previous messages
            prev_msg = self.msg_store[prev_response.id]
            messages.extend(prev_msg)

            for output_item in prev_response.output:
                assistant_text = self._output_message_text(output_item)
                if assistant_text is None:
                    continue
                messages.append({"role": "assistant", "content": assistant_text})

        # Append the new input
        # Responses API supports simple text inputs without chat format
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for input_item in request.input:
                normalized = self._normalize_response_message_for_chat(input_item)
                if normalized is not None:
                    messages.append(normalized)  # type: ignore

        # One Responses-API assistant turn maps to multiple input items
        # (message + function_call(s)); collapse them into one chat message
        # so chat templates render a single assistant block per turn.
        messages = self._merge_consecutive_assistant_messages(messages)

        # Most chat templates expect a single leading ``system`` message.
        # V4.1 can encode inline system messages: only coalesce its initial
        # instructions so appended developer/system updates do not rewrite
        # the conversation prefix. Keep developer -> system normalization
        # above; the native encoder does not treat developer as a system role.
        preserve_inline_system = self.chat_encoding_spec == "dsv41"
        system_chunks: list[str] = []
        other_msgs: list = []
        for m in messages:
            if (
                isinstance(m, dict)
                and m.get("role") == "system"
                and not (preserve_inline_system and other_msgs)
            ):
                content = m.get("content")
                if isinstance(content, str):
                    system_chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                system_chunks.append(text)
            else:
                other_msgs.append(m)
        if system_chunks:
            return [
                {"role": "system", "content": "\n\n".join(system_chunks)}
            ] + other_msgs
        return other_msgs

    def _construct_input_messages_with_harmony(
        self,
        request: ResponsesRequest,
        prev_response: Optional[ResponsesResponse],
    ) -> list[OpenAIMessage]:
        messages: list[OpenAIMessage] = []
        if prev_response is None:
            # New conversation.
            reasoning_effort = request.reasoning.effort if request.reasoning else None
            tool_types = [tool.type for tool in request.tools]
            enable_browser = (
                any(t in tool_types for t in ("web_search", "web_search_preview"))
                and self.tool_server is not None
            )
            enable_code_interpreter = (
                "code_interpreter" in tool_types and self.tool_server is not None
            )
            sys_msg = get_system_message(
                reasoning_effort=reasoning_effort,
                browser_description=(
                    self.tool_server.get_tool_description("browser")
                    if self.tool_server and enable_browser
                    else None
                ),
                python_description=(
                    self.tool_server.get_tool_description("python")
                    if self.tool_server and enable_code_interpreter
                    else None
                ),
            )
            messages.append(sys_msg)
            dev_msg = get_developer_message(request.instructions, request.tools)
            messages.append(dev_msg)
        else:
            # Continue the previous conversation.
            # FIXME: Currently, request params like reasoning and
            # instructions are ignored.
            prev_msgs = self.msg_store[prev_response.id]
            # Remove the previous chain-of-thoughts if there is a new "final"
            # message.
            if (
                len(prev_msgs) > 0
                and hasattr(prev_msgs[-1], "channel")
                and prev_msgs[-1].channel == "final"
            ):  # type: ignore[union-attr]
                prev_final_msg_idx = -1
                for i in range(len(prev_msgs) - 2, -1, -1):
                    if (
                        hasattr(prev_msgs[i], "channel")
                        and prev_msgs[i].channel == "final"
                    ):  # type: ignore[union-attr]
                        prev_final_msg_idx = i
                        break
                recent_turn_msgs = prev_msgs[prev_final_msg_idx + 1 :]
                del prev_msgs[prev_final_msg_idx + 1 :]
                for msg in recent_turn_msgs:
                    if hasattr(msg, "channel") and msg.channel != "analysis":  # type: ignore[union-attr]
                        prev_msgs.append(msg)
            messages.extend(prev_msgs)
        # Append the new input.
        # Responses API supports simple text inputs without chat format.
        if isinstance(request.input, str):
            messages.append(get_user_message(request.input))
        else:
            if prev_response is not None:
                prev_outputs = list(prev_response.output)
            else:
                prev_outputs = []
            for response_msg in request.input:
                messages.append(parse_response_input(response_msg, prev_outputs))
                if isinstance(response_msg, ResponseFunctionToolCall):
                    prev_outputs.append(response_msg)
        return messages

    async def _run_background_request(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        context: ConversationContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *,
        require_reasoning: bool,
    ):
        try:
            # Update the status to "in_progress"
            async with self.response_store_lock:
                stored_response = self.response_store.get(request.request_id)
                assert stored_response is not None
                stored_response.status = "in_progress"

            response = await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
                created_time,
                require_reasoning=require_reasoning,
            )
        except Exception as e:
            logger.exception("Background request failed for %s", request.request_id)
            response = self.create_error_response(str(e))

        if isinstance(response, ORJSONResponse):
            # If the request has failed, update the status to "failed"
            response_id = request.request_id
            async with self.response_store_lock:
                stored_response = self.response_store.get(response_id)
                assert stored_response is not None
                if stored_response.status not in ("completed", "cancelled"):
                    stored_response.status = "failed"

    async def retrieve_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)
        return response

    async def cancel_responses(
        self,
        response_id: str,
    ) -> Union[ResponsesResponse, ORJSONResponse]:
        if not response_id.startswith("resp_"):
            return self._make_invalid_id_error(response_id)

        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._make_not_found_error(response_id)

            prev_status = response.status
            if prev_status not in ("queued", "in_progress"):
                # already terminal; a second cancel is a no-op, return as-is.
                return response

            # Update the status to "cancelled"
            response.status = "cancelled"

        # The response_id is the same as the rid used when submitting the request
        self.tokenizer_manager.abort_request(rid=response_id)

        if task := self.background_tasks.get(response_id):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.exception("Background task for %s was cancelled", response_id)
        return response

    def _make_invalid_id_error(self, response_id: str):
        return self.create_error_response(
            message=(
                f"Invalid 'response_id': '{response_id}'. "
                "Expected an ID that begins with 'resp'."
            ),
            err_type="invalid_request_error",
            param="response_id",
        )

    def _make_not_found_error(self, response_id: str):
        return self.create_error_response(
            message=f"Response with id '{response_id}' not found.",
            err_type="invalid_request_error",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )

    async def responses_stream_generator(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[StreamingHarmonyContext],
        context: StreamingHarmonyContext,
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *,
        require_reasoning: bool,
    ) -> AsyncGenerator[str, None]:
        # TODO:
        # 1. Handle disconnect

        created_time = created_time or int(time.time())

        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            # Set sequence_number if the event has this attribute
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            # Get event type from the event's type field if it exists
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"
            )

        current_content_index = 0
        current_output_index = 0
        current_item_id = f"item_{random_uuid()}"
        sent_output_item_added = False

        initial_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )

        async for ctx in result_generator:
            # Only process context objects that implement the `is_expecting_start()` method,
            # which indicates they support per-turn streaming (e.g., StreamingHarmonyContext).
            # Contexts without this method are skipped, as they do not represent a new turn
            # or are not compatible with per-turn handling in the /v1/responses endpoint.
            if not hasattr(ctx, "is_expecting_start"):
                continue

            if ctx.is_expecting_start():
                current_output_index += 1
                sent_output_item_added = False

                if len(ctx.parser.messages) > 0:
                    previous_item = ctx.parser.messages[-1]
                    if previous_item.recipient is not None:
                        # Deal with tool call here
                        pass
                    elif previous_item.channel == "analysis":
                        reasoning_item = ResponseReasoningItem(
                            id=f"rs_{random_uuid()}",
                            type="reasoning",
                            summary=[],
                            content=[
                                ResponseReasoningTextContent(
                                    text=previous_item.content[0].text,
                                    type="reasoning_text",
                                ),
                            ],
                            status="completed",
                        )
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDoneEvent(
                                type="response.reasoning_text.done",
                                item_id=current_item_id,
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=reasoning_item,
                            )
                        )
                    elif previous_item.channel == "final":
                        text_content = openai_responses_types.ResponseOutputText(
                            type="output_text",
                            text=previous_item.content[0].text,
                            annotations=[],
                        )
                        yield _send_event(
                            openai_responses_types.ResponseTextDoneEvent(
                                type="response.output_text.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                text=previous_item.content[0].text,
                                logprobs=[],
                                item_id=current_item_id,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartDoneEvent(
                                type="response.content_part.done",
                                sequence_number=-1,
                                item_id=current_item_id,
                                output_index=current_output_index,
                                content_index=current_content_index,
                                part=text_content,
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[text_content],
                                    status="completed",
                                ),
                            )
                        )

            if ctx.parser.last_content_delta:
                if (
                    ctx.parser.current_channel == "final"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseOutputMessage(
                                    id=current_item_id,
                                    type="message",
                                    role="assistant",
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    yield _send_event(
                        openai_responses_types.ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=-1,
                            content_index=current_content_index,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            delta=ctx.parser.last_content_delta,
                            # TODO, use logprobs from ctx.last_request_output
                            logprobs=[],
                        )
                    )
                elif (
                    ctx.parser.current_channel == "analysis"
                    and ctx.parser.current_recipient is None
                ):
                    if not sent_output_item_added:
                        sent_output_item_added = True
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item=openai_responses_types.ResponseReasoningItem(
                                    type="reasoning",
                                    id=current_item_id,
                                    summary=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        yield _send_event(
                            openai_responses_types.ResponseContentPartAddedEvent(
                                type="response.content_part.added",
                                sequence_number=-1,
                                output_index=current_output_index,
                                item_id=current_item_id,
                                content_index=current_content_index,
                                # TODO: migrate this to
                                # ResponseReasoningTextContent for now
                                part=openai_responses_types.ResponseOutputText(
                                    type="output_text",
                                    text="",
                                    annotations=[],
                                    logprobs=None,
                                ),
                            )
                        )
                    # TODO: migrate to OpenAI types once updated.
                    yield _send_event(
                        openai_responses_types.ResponseReasoningTextDeltaEvent(
                            type="response.reasoning_text.delta",
                            item_id=current_item_id,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            delta=ctx.parser.last_content_delta,
                            sequence_number=-1,
                        )
                    )

            if ctx.is_assistant_action_turn() and len(ctx.parser.messages) > 0:
                previous_item = ctx.parser.messages[-1]
                if (
                    self.supports_browsing
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("browser.")
                ):
                    function_name = previous_item.recipient[len("browser.") :]
                    action = None
                    parsed_args = orjson.loads(previous_item.content[0].text)
                    if function_name == "search":
                        action = openai_responses_types.response_function_web_search.ActionSearch(
                            type="search",
                            query=parsed_args["query"],
                        )
                    elif function_name == "open":
                        action = openai_responses_types.response_function_web_search.ActionOpenPage(
                            type="open_page",
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    elif function_name == "find":
                        action = openai_responses_types.response_function_web_search.ActionFind(
                            type="find",
                            pattern=parsed_args["pattern"],
                            # TODO: translate to url
                            url=f"cursor:{parsed_args.get('cursor', '')}",
                        )
                    else:
                        raise ValueError(f"Unknown function name: {function_name}")

                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.response_function_web_search.ResponseFunctionWebSearch(
                                # TODO: generate a unique id for web search call
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallInProgressEvent(
                            type="response.web_search_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallSearchingEvent(
                            type="response.web_search_call.searching",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )

                    # enqueue
                    yield _send_event(
                        openai_responses_types.ResponseWebSearchCallCompletedEvent(
                            type="response.web_search_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseFunctionWebSearch(
                                type="web_search_call",
                                id=current_item_id,
                                action=action,
                                status="completed",
                            ),
                        )
                    )

                if (
                    self.supports_code_interpreter
                    and previous_item.recipient is not None
                    and previous_item.recipient.startswith("python")
                ):
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code="",
                                container_id="auto",
                                outputs=[],
                                status="in_progress",
                            ),
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInProgressEvent(
                            type="response.code_interpreter_call.in_progress",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    # TODO: do we need to add delta event here?
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCodeDoneEvent(
                            type="response.code_interpreter_call_code.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            code=previous_item.content[0].text,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallInterpretingEvent(
                            type="response.code_interpreter_call.interpreting",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseCodeInterpreterCallCompletedEvent(
                            type="response.code_interpreter_call.completed",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                        )
                    )
                    yield _send_event(
                        openai_responses_types.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=openai_responses_types.ResponseCodeInterpreterToolCallParam(
                                type="code_interpreter_call",
                                id=current_item_id,
                                code=previous_item.content[0].text,
                                container_id="auto",
                                # TODO: add outputs here
                                outputs=[],
                                status="completed",
                            ),
                        )
                    )

        async def empty_async_generator():
            for _ in ():
                yield

        final_response = await self.responses_full_generator(
            request,
            sampling_params,
            empty_async_generator(),
            context,
            model_name,
            tokenizer,
            request_metadata,
            created_time=created_time,
            require_reasoning=require_reasoning,
        )
        response_dict = final_response.model_dump()
        # OpenAI SDK's Tool union may not know extended types; drop echo.
        response_dict["tools"] = []

        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=response_dict,
            )
        )

    async def responses_stream_generator_non_harmony(
        self,
        request: ResponsesRequest,
        sampling_params: Any,
        result_generator: AsyncIterator[Any],
        model_name: str,
        tokenizer: Any,
        request_metadata: RequestResponseMetadata,
        created_time: Optional[int] = None,
        *,
        require_reasoning: bool,
        processed_request: Optional[ChatCompletionRequest] = None,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a /v1/responses response as typed OpenAI SSE events for
        non-harmony models. Each engine chunk is run through the reasoning
        and function-call parsers; leftover text becomes
        ``response.output_text.delta``.
        """

        created_time = created_time or int(time.time())
        sequence_number = 0

        def _send_event(event):
            nonlocal sequence_number
            if hasattr(event, "sequence_number"):
                event.sequence_number = sequence_number
            sequence_number += 1
            event_type = getattr(event, "type", "unknown")
            return (
                f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"
            )

        # The streaming Response* event models echo ``tools`` through a
        # narrower OpenAI SDK Tool union; strip it to avoid pydantic
        # validation failures on extended tool types.
        def _sanitize_response_dict(d: dict) -> dict:
            d["tools"] = []
            return d

        initial_response = _sanitize_response_dict(
            ResponsesResponse.from_request(
                request,
                sampling_params,
                model_name=model_name,
                created_time=created_time,
                output=[],
                status="in_progress",
                usage=None,
            ).model_dump()
        )
        yield _send_event(
            openai_responses_types.ResponseCreatedEvent(
                type="response.created",
                sequence_number=-1,
                response=initial_response,
            )
        )
        yield _send_event(
            openai_responses_types.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=-1,
                response=initial_response,
            )
        )

        chat_tools = self._response_tools_to_chat_tools(request)
        is_required = request.tool_choice == "required"
        tool_parser: Optional[Union[FunctionCallParser, JsonArrayParser]] = None
        if chat_tools and request.tool_choice != "none":
            native_supports_structural_tag = False
            if self.tool_call_parser:
                probe = FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                native_supports_structural_tag = (
                    probe.detector.supports_structural_tag()
                )
            if is_required and not native_supports_structural_tag:
                tool_parser = JsonArrayParser()
            elif self.tool_call_parser:
                tool_parser = FunctionCallParser(
                    chat_tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
        reasoning_parser_obj: Optional[ReasoningParser] = None
        if self.reasoning_parser:
            reasoning_parser_obj = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=True,
                # A template that prefills <think> forces the parser open even
                # when the request itself did not ask for reasoning, same as chat.
                force_reasoning=(
                    self.template_manager.force_reasoning or require_reasoning
                ),
                request=request,
                tokenizer=self.tokenizer_manager.tokenizer,
                tool_call_parser_active=isinstance(tool_parser, FunctionCallParser),
            )

        current_output_index = -1
        reasoning_state = {
            "open": False,
            "item_id": "",
            "output_index": -1,
            "text": "",
        }
        message_state = {
            "open": False,
            "item_id": "",
            "output_index": -1,
            "text": "",
        }
        tool_call_states: dict[int, dict[str, Any]] = {}
        # Items closed during the stream, in wire order. Feeds the final
        # ``response.completed`` snapshot and the stored response.
        emitted_items: list = []
        tool_transaction_active = False
        tool_transaction_frames: list[str] = []
        tool_transaction_sequence_checkpoint = 0
        tool_transaction_output_index_checkpoint = -1
        tool_transaction_items_checkpoint = 0
        raw_model_tail = ""
        recovery_usage = None

        def _queue_event(event) -> Optional[str]:
            frame = _send_event(event)
            if tool_transaction_active:
                tool_transaction_frames.append(frame)
                return None
            return frame

        def _begin_tool_transaction() -> None:
            nonlocal tool_transaction_active
            nonlocal tool_transaction_sequence_checkpoint
            nonlocal tool_transaction_output_index_checkpoint
            nonlocal tool_transaction_items_checkpoint
            if tool_transaction_active:
                return
            tool_transaction_active = True
            tool_transaction_sequence_checkpoint = sequence_number
            tool_transaction_output_index_checkpoint = current_output_index
            tool_transaction_items_checkpoint = len(emitted_items)

        def _rollback_tool_transaction() -> None:
            nonlocal sequence_number, current_output_index
            nonlocal tool_transaction_active
            if not tool_transaction_active:
                return
            sequence_number = tool_transaction_sequence_checkpoint
            current_output_index = tool_transaction_output_index_checkpoint
            del emitted_items[tool_transaction_items_checkpoint:]
            tool_transaction_frames.clear()
            tool_call_states.clear()
            message_state.update(
                open=False,
                item_id="",
                output_index=-1,
                text="",
            )
            tool_transaction_active = False

        def _commit_tool_transaction() -> list[str]:
            nonlocal tool_transaction_active
            frames = list(tool_transaction_frames)
            tool_transaction_frames.clear()
            tool_transaction_active = False
            return frames

        def _validate_tool_transaction() -> None:
            if not tool_call_states:
                raise MalformedToolCallError(
                    "Model emitted incomplete tool-call stream"
                )
            for state in tool_call_states.values():
                if not state.get("name"):
                    raise MalformedToolCallError(
                        "Model emitted incomplete tool-call stream"
                    )
                try:
                    arguments = json.loads(state["arguments"] or "{}")
                except (json.JSONDecodeError, TypeError) as e:
                    raise MalformedToolCallError(
                        "Model emitted malformed tool-call arguments"
                    ) from e
                if not isinstance(arguments, dict):
                    raise MalformedToolCallError(
                        "Model emitted non-object tool-call arguments"
                    )
                self._validate_response_tool_input(
                    request,
                    state["name"],
                    arguments,
                )

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        total_tokens_meta = 0
        reasoning_tokens_meta = 0
        finish_reason: Optional[dict[str, Any]] = None
        flushed = False
        stream_offset = 0
        incremental = self.tokenizer_manager.server_args.incremental_streaming_output

        def _open_reasoning_item() -> str:
            nonlocal current_output_index
            current_output_index += 1
            item_id = f"rs_{random_uuid()}"
            reasoning_state.update(
                open=True, item_id=item_id, output_index=current_output_index, text=""
            )
            return item_id

        wants_summary = self._wants_reasoning_summary(request)

        def _close_reasoning_item():
            if not reasoning_state["open"]:
                return []
            text = reasoning_state["text"]
            completed_item = ResponseReasoningItem(
                id=reasoning_state["item_id"],
                type="reasoning",
                summary=(
                    [ResponseReasoningSummary(type="summary_text", text=text)]
                    if wants_summary
                    else []
                ),
                content=[
                    ResponseReasoningTextContent(type="reasoning_text", text=text),
                ],
                status="completed",
            )
            events: list = []
            if wants_summary:
                events.append(
                    _queue_event(
                        openai_responses_types.ResponseReasoningSummaryTextDoneEvent(
                            type="response.reasoning_summary_text.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            summary_index=0,
                            text=text,
                        )
                    )
                )
                events.append(
                    _queue_event(
                        openai_responses_types.ResponseReasoningSummaryPartDoneEvent(
                            type="response.reasoning_summary_part.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            summary_index=0,
                            part=ResponseReasoningSummaryDonePart(
                                type="summary_text", text=text
                            ),
                        )
                    )
                )
            else:
                events.append(
                    _queue_event(
                        openai_responses_types.ResponseReasoningTextDoneEvent(
                            type="response.reasoning_text.done",
                            item_id=reasoning_state["item_id"],
                            sequence_number=-1,
                            output_index=reasoning_state["output_index"],
                            content_index=0,
                            text=text,
                        )
                    )
                )
            events += [
                _queue_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=reasoning_state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            reasoning_state["open"] = False
            return events

        def _open_message_item() -> str:
            nonlocal current_output_index
            current_output_index += 1
            item_id = f"msg_{random_uuid()}"
            message_state.update(
                open=True, item_id=item_id, output_index=current_output_index, text=""
            )
            return item_id

        def _close_message_item(*, phase="commentary"):
            if not message_state["open"]:
                return []
            text = message_state["text"]
            text_content = openai_responses_types.ResponseOutputText(
                type="output_text", text=text, annotations=[], logprobs=None
            )
            completed_item = ResponseOutputMessage(
                id=message_state["item_id"],
                type="message",
                role="assistant",
                content=[text_content],
                status="completed",
                **({"phase": phase} if self.chat_encoding_spec == "dsv41" else {}),
            )
            events = [
                _queue_event(
                    openai_responses_types.ResponseTextDoneEvent(
                        type="response.output_text.done",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        content_index=0,
                        text=text,
                        logprobs=[],
                        item_id=message_state["item_id"],
                    )
                ),
                _queue_event(
                    openai_responses_types.ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        sequence_number=-1,
                        item_id=message_state["item_id"],
                        output_index=message_state["output_index"],
                        content_index=0,
                        part=text_content,
                    )
                ),
                _queue_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            message_state["open"] = False
            return events

        def _close_tool_call_state(tool_index: int):
            state = tool_call_states.get(tool_index)
            if state is None or state.get("done"):
                return []
            arguments = state["arguments"]
            completed_item = ResponseFunctionToolCall(
                arguments=arguments,
                call_id=state["call_id"],
                name=state["name"] or "",
                type="function_call",
                id=state["item_id"],
                status="completed",
            )
            events = [
                _queue_event(
                    openai_responses_types.ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=-1,
                        item_id=state["item_id"],
                        output_index=state["output_index"],
                        arguments=arguments,
                        name=state["name"] or "",
                    )
                ),
                _queue_event(
                    openai_responses_types.ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=-1,
                        output_index=state["output_index"],
                        item=completed_item,
                    )
                ),
            ]
            emitted_items.append(completed_item)
            state["done"] = True
            return events

        def _yieldable(events):
            return (event for event in events if event is not None)

        def _emit_tool_calls(calls):
            nonlocal current_output_index
            if calls:
                if reasoning_state["open"]:
                    yield from _yieldable(_close_reasoning_item())
                if message_state["open"]:
                    yield from _yieldable(_close_message_item())
                _begin_tool_transaction()

            for call in calls:
                tool_index = call.tool_index
                state = tool_call_states.get(tool_index)
                if state is None or state.get("done"):
                    for other_index in list(tool_call_states):
                        if other_index != tool_index:
                            yield from _yieldable(_close_tool_call_state(other_index))
                    current_output_index += 1
                    state = {
                        "item_id": f"fc_{random_uuid()[:8]}",
                        "call_id": f"call_{random_uuid()[:24]}",
                        "output_index": current_output_index,
                        "name": call.name or "",
                        "arguments": "",
                        "added": False,
                        "done": False,
                    }
                    tool_call_states[tool_index] = state
                if not state["added"]:
                    state["added"] = True
                    frame = _queue_event(
                        openai_responses_types.ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=state["output_index"],
                            item=ResponseFunctionToolCall(
                                arguments="",
                                call_id=state["call_id"],
                                name=state["name"],
                                type="function_call",
                                id=state["item_id"],
                                status="in_progress",
                            ),
                        )
                    )
                    if frame is not None:
                        yield frame
                if call.parameters:
                    state["arguments"] += call.parameters
                    frame = _queue_event(
                        openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                            type="response.function_call_arguments.delta",
                            sequence_number=-1,
                            item_id=state["item_id"],
                            output_index=state["output_index"],
                            delta=call.parameters,
                        )
                    )
                    if frame is not None:
                        yield frame

        def _emit_normal_text(normal_text: str):
            if not normal_text or not _should_emit_normal_text_as_message(
                normal_text,
                any_tool_call_in_progress=any(
                    not state.get("done") for state in tool_call_states.values()
                ),
            ):
                return
            if reasoning_state["open"]:
                yield from _yieldable(_close_reasoning_item())
            for tool_index in list(tool_call_states):
                yield from _yieldable(_close_tool_call_state(tool_index))
            if not message_state["open"]:
                item_id = _open_message_item()
                frame = _queue_event(
                    openai_responses_types.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        item=ResponseOutputMessage(
                            id=item_id,
                            type="message",
                            role="assistant",
                            content=[],
                            status="in_progress",
                            # V4.1 has no native phase token. Keep the turn working
                            # while streaming; completion resolves whether this
                            # message precedes tools or is the terminal answer.
                            **(
                                {"phase": "commentary"}
                                if self.chat_encoding_spec == "dsv41"
                                else {}
                            ),
                        ),
                    )
                )
                if frame is not None:
                    yield frame
                frame = _queue_event(
                    openai_responses_types.ResponseContentPartAddedEvent(
                        type="response.content_part.added",
                        sequence_number=-1,
                        output_index=message_state["output_index"],
                        item_id=message_state["item_id"],
                        content_index=0,
                        part=openai_responses_types.ResponseOutputText(
                            type="output_text",
                            text="",
                            annotations=[],
                            logprobs=None,
                        ),
                    )
                )
                if frame is not None:
                    yield frame
            message_state["text"] += normal_text
            frame = _queue_event(
                openai_responses_types.ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=-1,
                    content_index=0,
                    output_index=message_state["output_index"],
                    item_id=message_state["item_id"],
                    delta=normal_text,
                    logprobs=[],
                )
            )
            if frame is not None:
                yield frame

        def _stage_recovered_calls(result: _ResponsesToolRecoveryResult):
            nonlocal current_output_index
            _begin_tool_transaction()
            for tool_index, call in enumerate(result.calls):
                current_output_index += 1
                state = {
                    "item_id": f"fc_{random_uuid()[:8]}",
                    "call_id": call.call_id,
                    "output_index": current_output_index,
                    "name": call.name,
                    "arguments": call.arguments,
                    "added": True,
                    "done": False,
                }
                tool_call_states[tool_index] = state
                _queue_event(
                    openai_responses_types.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=-1,
                        output_index=state["output_index"],
                        item=ResponseFunctionToolCall(
                            arguments="",
                            call_id=state["call_id"],
                            name=state["name"],
                            type="function_call",
                            id=state["item_id"],
                            status="in_progress",
                        ),
                    )
                )
                if call.arguments:
                    _queue_event(
                        openai_responses_types.ResponseFunctionCallArgumentsDeltaEvent(
                            type="response.function_call_arguments.delta",
                            sequence_number=-1,
                            item_id=state["item_id"],
                            output_index=state["output_index"],
                            delta=call.arguments,
                        )
                    )
                _close_tool_call_state(tool_index)
            _validate_tool_transaction()
            return _commit_tool_transaction()

        async def _close_result_stream() -> None:
            close = getattr(result_generator, "aclose", None)
            if close is None:
                return
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to close failed Responses generation stream")

        try:
            async for ctx in result_generator:
                if isinstance(ctx, dict):
                    chunk = ctx
                else:
                    chunk = getattr(ctx, "last_output", None)
                if not isinstance(chunk, dict):
                    continue
                meta = chunk.get("meta_info") or {}
                prompt_tokens = meta.get("prompt_tokens", prompt_tokens)
                completion_tokens = meta.get("completion_tokens", completion_tokens)
                cached_tokens = meta.get("cached_tokens", cached_tokens)
                total_tokens_meta = meta.get("total_tokens", total_tokens_meta)
                reasoning_tokens_meta = meta.get(
                    "reasoning_tokens", reasoning_tokens_meta
                )
                finish_reason = meta.get("finish_reason") or finish_reason

                text = chunk.get("text", "") or ""
                if incremental:
                    delta = text
                else:
                    delta = text[stream_offset:]
                    stream_offset = len(text)
                raw_model_tail = (raw_model_tail + delta)[
                    -self._tool_failure_tail_chars :
                ]
                if not delta and finish_reason is None:
                    continue
                flush = (
                    not flushed
                    and finish_reason is not None
                    and finish_reason.get("type") != "abort"
                )
                flushed = flushed or flush

                if reasoning_parser_obj is not None:
                    reasoning_chunk, delta = reasoning_parser_obj.parse_stream_chunk(
                        delta
                    )
                    if flush:
                        end_reasoning, end_normal = (
                            reasoning_parser_obj.parse_stream_end()
                        )
                        if end_reasoning:
                            reasoning_chunk = (reasoning_chunk or "") + end_reasoning
                        if end_normal:
                            delta = (delta or "") + end_normal
                else:
                    reasoning_chunk = None

                if reasoning_chunk:
                    if tool_transaction_active:
                        raise MalformedToolCallError(
                            "Invalid tool-call stream ordering"
                        )
                    if message_state["open"]:
                        for frame in _yieldable(_close_message_item()):
                            yield frame
                    if not reasoning_state["open"]:
                        item_id = _open_reasoning_item()
                        yield _send_event(
                            openai_responses_types.ResponseOutputItemAddedEvent(
                                type="response.output_item.added",
                                sequence_number=-1,
                                output_index=reasoning_state["output_index"],
                                item=ResponseReasoningItem(
                                    id=item_id,
                                    type="reasoning",
                                    summary=[],
                                    content=[],
                                    status="in_progress",
                                ),
                            )
                        )
                        if wants_summary:
                            yield _send_event(
                                openai_responses_types.ResponseReasoningSummaryPartAddedEvent(
                                    type="response.reasoning_summary_part.added",
                                    item_id=item_id,
                                    output_index=reasoning_state["output_index"],
                                    summary_index=0,
                                    part=ResponseReasoningSummaryAddedPart(
                                        type="summary_text", text=""
                                    ),
                                    sequence_number=-1,
                                )
                            )
                    reasoning_state["text"] += reasoning_chunk
                    if wants_summary:
                        yield _send_event(
                            openai_responses_types.ResponseReasoningSummaryTextDeltaEvent(
                                type="response.reasoning_summary_text.delta",
                                item_id=reasoning_state["item_id"],
                                output_index=reasoning_state["output_index"],
                                summary_index=0,
                                delta=reasoning_chunk,
                                sequence_number=-1,
                            )
                        )
                    else:
                        yield _send_event(
                            openai_responses_types.ResponseReasoningTextDeltaEvent(
                                type="response.reasoning_text.delta",
                                item_id=reasoning_state["item_id"],
                                output_index=reasoning_state["output_index"],
                                content_index=0,
                                delta=reasoning_chunk,
                                sequence_number=-1,
                            )
                        )

                if not delta and not flush:
                    continue

                if isinstance(tool_parser, JsonArrayParser):
                    parsed = tool_parser.parse_streaming_increment(delta, chat_tools)
                    normal_text, tool_calls = parsed.normal_text or "", parsed.calls
                elif tool_parser is not None:
                    normal_text, tool_calls = tool_parser.parse_stream_chunk(delta)
                    if flush:
                        end_text, end_calls = tool_parser.parse_stream_end()
                        normal_text = (normal_text or "") + end_text
                        tool_calls = list(tool_calls) + end_calls
                else:
                    normal_text, tool_calls = delta, []

                def _is_continuing(call):
                    state = tool_call_states.get(call.tool_index)
                    return state is not None and not state.get("done")

                continuing = [call for call in tool_calls if _is_continuing(call)]
                opening = [call for call in tool_calls if not _is_continuing(call)]

                for frame in _emit_tool_calls(continuing):
                    yield frame
                for frame in _emit_normal_text(normal_text):
                    yield frame
                for frame in _emit_tool_calls(opening):
                    yield frame

                if flush and tool_transaction_active:
                    for frame in _yieldable(_close_message_item()):
                        yield frame
                    for tool_index in list(tool_call_states):
                        for frame in _yieldable(_close_tool_call_state(tool_index)):
                            yield frame
                    _validate_tool_transaction()
                    for frame in _commit_tool_transaction():
                        yield frame

            if tool_transaction_active:
                raise MalformedToolCallError(
                    "Model emitted incomplete tool-call stream"
                )
        except Exception as exc:
            logger.exception("Error while streaming /v1/responses")
            self._write_tool_failure_dump(request.request_id, exc, raw_model_tail)
            failure_error = _response_failure_error(exc)
            recovery_reason = _response_tool_failure_reason(str(exc))
            candidate_tool_names = tuple(
                dict.fromkeys(
                    state["name"]
                    for state in tool_call_states.values()
                    if state.get("name")
                )
            )
            await _close_result_stream()
            _rollback_tool_transaction()
            # A malformed marker can fail before the parser has emitted a tool
            # name, while a reasoning or text prefix is still open on the wire.
            # Finalize that already-published prefix before recovery adds a later
            # output item, preserving output-index and done-event ordering.
            for frame in _yieldable(_close_reasoning_item()):
                yield frame
            for frame in _yieldable(_close_message_item()):
                yield frame

            recovered = None
            if recovery_reason is not None:
                recovery_task = asyncio.create_task(
                    self._attempt_response_tool_recovery(
                        processed_request,
                        request,
                        raw_request,
                        recovery_reason,
                        request.request_id,
                        candidate_tool_names,
                    )
                )
                try:
                    while not recovery_task.done():
                        done, _ = await asyncio.wait(
                            {recovery_task},
                            timeout=self._tool_recovery_ping_interval_s,
                        )
                        if not done:
                            yield ": responses-tool-recovery\n\n"
                    recovered = recovery_task.result()
                except asyncio.CancelledError:
                    if not recovery_task.done():
                        recovery_task.cancel()
                        try:
                            await recovery_task
                        except asyncio.CancelledError:
                            pass
                    raise
                except Exception:
                    logger.exception(
                        "Responses tool recovery callback failed: reason=%s",
                        recovery_reason,
                    )

            if recovered is not None:
                for frame in _stage_recovered_calls(recovered):
                    yield frame
                recovery_usage = recovered.usage
                finish_reason = {"type": "tool_calls"}
            else:
                tool_call_states.clear()
                failed = _sanitize_response_dict(
                    ResponsesResponse.from_request(
                        request,
                        sampling_params,
                        model_name=model_name,
                        created_time=created_time,
                        output=list(emitted_items),
                        status="failed",
                        usage=None,
                    ).model_dump()
                )
                failed["error"] = failure_error
                yield _send_event(
                    openai_responses_types.ResponseFailedEvent(
                        type="response.failed",
                        sequence_number=-1,
                        response=failed,
                    )
                )
                return

        for ev in _close_reasoning_item():
            if ev is not None:
                yield ev
        for ev in _close_message_item(
            phase=(
                "final_answer"
                if not tool_call_states
                and (finish_reason or {}).get("type") == "stop"
                else "commentary"
            )
        ):
            if ev is not None:
                yield ev
        for tool_index in list(tool_call_states):
            for ev in _close_tool_call_state(tool_index):
                if ev is not None:
                    yield ev

        final_output_items = list(emitted_items)

        if recovery_usage is not None:
            original_total_tokens = total_tokens_meta or (
                prompt_tokens + completion_tokens
            )
            recovered_prompt_tokens = getattr(recovery_usage, "prompt_tokens", 0) or 0
            recovered_completion_tokens = (
                getattr(recovery_usage, "completion_tokens", 0) or 0
            )
            recovered_total_tokens = getattr(recovery_usage, "total_tokens", 0) or (
                recovered_prompt_tokens + recovered_completion_tokens
            )
            recovered_reasoning_tokens = (
                getattr(recovery_usage, "reasoning_tokens", 0) or 0
            )
            recovered_prompt_details = getattr(
                recovery_usage, "prompt_tokens_details", None
            )
            recovered_cached_tokens = (
                getattr(recovered_prompt_details, "cached_tokens", 0) or 0
            )
            prompt_tokens += recovered_prompt_tokens
            completion_tokens += recovered_completion_tokens
            total_tokens_meta = original_total_tokens + recovered_total_tokens
            reasoning_tokens_meta += recovered_reasoning_tokens
            cached_tokens += recovered_cached_tokens

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens_meta or (prompt_tokens + completion_tokens),
            reasoning_tokens=reasoning_tokens_meta,
        )
        if self.enable_prompt_tokens_details and cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=cached_tokens
            )
        request_metadata.final_usage_info = usage

        final_response = ResponsesResponse.from_request(
            request,
            sampling_params,
            model_name=model_name,
            created_time=created_time,
            output=final_output_items,
            status=self._status_from_finish_reason(finish_reason),
            usage=usage,
        )
        if request.store:
            async with self.response_store_lock:
                stored = self.response_store.get(final_response.id)
                if stored is None or stored.status != "cancelled":
                    self.response_store[final_response.id] = final_response

        response_dict = _sanitize_response_dict(final_response.model_dump())

        yield _send_event(
            openai_responses_types.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=response_dict,
            )
        )

    async def _generate_with_builtin_tools(
        self,
        request_id: str,
        request_prompt: Any,
        adapted_request: GenerateReqInput,
        sampling_params: Any,
        context: ConversationContext,
        raw_request: Optional[Request] = None,
        priority: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Generate with builtin tool support for harmony-based models."""
        orig_priority = priority or 0

        while True:
            # Generate using SGLang's tokenizer manager
            generator = self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            )

            try:
                async for res in generator:
                    context.append_output(res)
                    # NOTE(woosuk): The stop condition is handled by the engine.
                    yield context
            finally:
                # Parser failure or client disconnect can close this outer
                # generator while the tokenizer stream is suspended after a
                # yield. Close it deterministically so the scheduler receives
                # AbortReq instead of decoding into deleted TokenizerManager state.
                close = getattr(generator, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "Failed to close child Responses generation stream"
                        )

            if not context.need_builtin_tool_call():
                # The model did not ask for a tool call, so we're done.
                break

            # Call the tool and update the context with the result.
            tool_output = await context.call_tool()
            context.append_output(tool_output)

            # Prepare for the next generation turn
            # Render the updated conversation for the next completion
            prompt_token_ids = context.render_for_completion()

            # Update the adapted request with new prompt
            adapted_request = GenerateReqInput(
                input_ids=prompt_token_ids,
                sampling_params=sampling_params,
                stream=adapted_request.stream,
                rid=request_id,
                session_id=adapted_request.session_id,
                extra_key=adapted_request.extra_key,
                cache_salt=adapted_request.cache_salt,
                return_logprob=adapted_request.return_logprob,
                logprob_start_len=adapted_request.logprob_start_len,
                top_logprobs_num=adapted_request.top_logprobs_num,
                return_text_in_logprobs=adapted_request.return_text_in_logprobs,
                return_hidden_states=adapted_request.return_hidden_states,
                background=adapted_request.background,
                require_reasoning=adapted_request.require_reasoning,
            )

            # Update sampling params with reduced max_tokens
            if hasattr(sampling_params, "max_new_tokens") or isinstance(
                sampling_params, dict
            ):
                context_len = getattr(
                    self.tokenizer_manager.model_config, "context_len", 4096
                )
                num_reserved_tokens = self.tokenizer_manager.num_reserved_tokens
                remaining_tokens = (
                    context_len - len(prompt_token_ids) - num_reserved_tokens
                )

                if isinstance(sampling_params, dict):
                    sampling_params["max_new_tokens"] = max(remaining_tokens, 1)
                else:
                    sampling_params.max_new_tokens = max(remaining_tokens, 1)

            # Slightly reduce priority for subsequent tool calls
            priority = orig_priority - 1
