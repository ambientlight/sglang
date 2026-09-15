"""Handler for Anthropic Messages API requests.

Converts Anthropic requests to OpenAI ChatCompletion format, delegates to
OpenAIServingChat for processing, and converts responses back to Anthropic format.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import pathlib
import uuid
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Optional,
    Union,
)

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, ValidationError
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessage,
    AnthropicMessageEndDelta,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ErrorEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    is_server_tool,
)
from sglang.srt.entrypoints.openai.chat_encoding import (
    InlineSystemStrategy,
    custom_encoder_inline_system_strategy,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.core_types import MalformedToolCallError
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.parser.template_detection import detect_inline_system_support

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)

# Map OpenAI finish reasons to Anthropic stop reasons. Only the four
# values in ``AnthropicMessagesResponse.stop_reason``'s Literal are valid
# on the wire; ``content_filter`` and ``abort`` have no perfect mapping
# so they fall through to the ``end_turn`` default with a WARNING at the
# call site so operators don't lose the safety/abort signal in logs.
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "request_timeout_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
}

_THINKING_KEY_FILE_ENV = "SGLANG_ANTHROPIC_THINKING_KEY_FILE"
_THINKING_SIGNATURE_PREFIX = "sglang_thinking_v1."
_THINKING_SIGNATURE_AAD = b"sglang-anthropic-thinking-v1"
_MAX_SEALED_THINKING_BYTES = 8 * 1024 * 1024
_TOOL_RECOVERY_ENV = "SGLANG_ANTHROPIC_TOOL_RECOVERY"
_TOOL_RECOVERY_TIMEOUT_ENV = "SGLANG_ANTHROPIC_TOOL_RECOVERY_TIMEOUT_S"
_TOOL_RECOVERY_MAX_TOKENS_ENV = "SGLANG_ANTHROPIC_TOOL_RECOVERY_MAX_TOKENS"
_TOOL_RECOVERY_CONCURRENCY_ENV = "SGLANG_ANTHROPIC_TOOL_RECOVERY_CONCURRENCY"
_TOOL_RECOVERY_PING_INTERVAL_ENV = "SGLANG_ANTHROPIC_TOOL_RECOVERY_PING_INTERVAL_S"


class _ToolRecoveryReason(str, Enum):
    MALFORMED_PROTOCOL = "malformed_protocol"
    MALFORMED_ARGUMENTS = "malformed_arguments"
    NON_OBJECT_ARGUMENTS = "non_object_arguments"
    SCHEMA_MISMATCH = "schema_mismatch"
    UNDECLARED_TOOL = "undeclared_tool"
    STREAM_ORDERING = "stream_ordering"
    INCOMPLETE_STREAM = "incomplete_stream"


def _classify_tool_recovery_reason(message: str) -> Optional[_ToolRecoveryReason]:
    """Map outward failure text to a stable, non-payload diagnostic code."""

    normalized = (message or "").lower()
    if "arguments that do not match the declared schema" in normalized:
        return _ToolRecoveryReason.SCHEMA_MISMATCH
    if "undeclared tool" in normalized:
        return _ToolRecoveryReason.UNDECLARED_TOOL
    if "non-object" in normalized and (
        "tool-call" in normalized or "deepseek tool-call" in normalized
    ):
        return _ToolRecoveryReason.NON_OBJECT_ARGUMENTS
    if "malformed tool-call arguments" in normalized:
        return _ToolRecoveryReason.MALFORMED_ARGUMENTS
    if (
        "invalid tool-call stream ordering" in normalized
        or "conflicting tool-call stream index" in normalized
    ):
        return _ToolRecoveryReason.STREAM_ORDERING
    if "incomplete tool-call stream" in normalized:
        return _ToolRecoveryReason.INCOMPLETE_STREAM
    if "malformed" in normalized and "tool" in normalized:
        return _ToolRecoveryReason.MALFORMED_PROTOCOL
    return None


def _bounded_positive_int_env(
    name: str,
    default: int,
    maximum: int,
) -> int:
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


@dataclass
class _PendingAnthropicToolCall:
    """One tool segment held until the upstream parser completes the turn.

    Anthropic clients incrementally parse ``input_json_delta``.  Once a
    ``tool_use`` block is on the wire it cannot be retracted if the model later
    emits malformed protocol.  The transaction starts at the first tool call
    and also holds any later text so mixed tool/text output can be validated and
    published in its original block order.
    """

    tool_index: int
    id: str
    name: str
    argument_parts: list[str]


@dataclass
class _PendingAnthropicTextSegment:
    """Ordinary response text ordered between pending tool calls."""

    parts: list[str]


_PendingAnthropicSegment = Union[
    _PendingAnthropicToolCall,
    _PendingAnthropicTextSegment,
]


@dataclass(frozen=True)
class _RecoveredAnthropicToolCall:
    id: str
    name: str
    tool_input: dict[str, Any]
    argument_json: str


@dataclass(frozen=True)
class _AnthropicToolRecoveryResult:
    calls: tuple[_RecoveredAnthropicToolCall, ...]
    usage: Any
    model: str
    recovery_rid: str
    response: ChatCompletionResponse


def _validate_anthropic_tool_input(
    request: AnthropicMessagesRequest,
    name: str,
    tool_input: dict[str, Any],
    *,
    enforce_schema: bool = False,
) -> None:
    """Validate a model-produced call before publishing ``tool_use``.

    Anthropic guarantees schema conformance only for tools declared with
    ``strict: true``.  A syntactically valid object from a non-strict tool must
    reach the client so its normal ``tool_result`` validation/error loop can
    ask the model to correct the call. Raw parser grammar is validated in the
    model-specific detector; this API-layer check deliberately treats
    schema-valid strings as opaque so legitimate XML or DSML-like tool input
    remains supported.

    Unit-level translation tests sometimes omit request tools entirely and feed
    synthetic OpenAI chunks directly. Preserve that internal compatibility.
    """
    if request.tools is None:
        return

    declarations = {}
    for tool in request.tools:
        if not is_server_tool(tool):
            declarations[tool.name] = tool

    declaration = declarations.get(name)
    if declaration is None:
        raise MalformedToolCallError("Model attempted to call an undeclared tool")

    try:
        Draft202012Validator(declaration.input_schema).validate(tool_input)
    except JSONSchemaValidationError as e:
        if enforce_schema or declaration.strict is True:
            raise MalformedToolCallError(
                "Model emitted tool-call arguments that do not match the declared schema"
            ) from e
        logger.warning(
            "Publishing schema-invalid arguments for non-strict Anthropic tool %r; "
            "the client may return an is_error tool_result",
            name,
        )
    except (JSONSchemaError, RecursionError) as e:
        # Request validation normally catches this before generation. Keep the
        # publication boundary safe when called directly or by a custom route.
        raise MalformedToolCallError("Declared tool schema is invalid") from e


class _ThinkingSignatureCodec:
    """Seal hidden reasoning into an authenticated, stateless continuation token.

    Anthropic clients treat ``thinking.signature`` as opaque and return it
    unchanged on later turns. A local backend still needs the hidden text to
    reconstruct the model prompt, so ``display=omitted`` cannot be implemented
    by merely dropping reasoning. This codec encrypts a compressed,
    model-bound payload with AES-256-GCM. The key lives outside the image and
    must remain stable across container restarts.

    Tokens deliberately carry an SGLang prefix. They are valid only on the
    SGLang deployment holding the matching key; they never masquerade as an
    Anthropic-issued signature.
    """

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("Anthropic thinking-signature key must be 32 bytes")
        self._key_id = hashlib.sha256(key).hexdigest()[:16]
        self._cipher = AESGCM(key)

    @property
    def key_id(self) -> str:
        return self._key_id

    @classmethod
    def from_environment(cls) -> Optional[_ThinkingSignatureCodec]:
        key_path_value = os.environ.get(_THINKING_KEY_FILE_ENV)
        if not key_path_value:
            return None

        key_path = pathlib.Path(key_path_value)
        raw = key_path.read_bytes()
        key = cls._decode_key_file(raw)
        mode = key_path.stat().st_mode & 0o777
        if mode & 0o077:
            raise RuntimeError(
                f"{_THINKING_KEY_FILE_ENV} must not be group/world accessible "
                f"(mode is {mode:04o})"
            )
        return cls(key)

    @staticmethod
    def _decode_key_file(raw: bytes) -> bytes:
        if len(raw) == 32:
            return raw

        encoded = raw.strip()
        if len(encoded) == 64:
            try:
                decoded = bytes.fromhex(encoded.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                decoded = b""
            if len(decoded) == 32:
                return decoded

        try:
            decoded = base64.b64decode(encoded, altchars=b"-_", validate=True)
        except (ValueError, TypeError):
            decoded = b""
        if len(decoded) == 32:
            return decoded
        raise ValueError(
            "Anthropic thinking-signature key file must contain 32 raw bytes, "
            "64 hexadecimal characters, or base64 for 32 bytes"
        )

    @staticmethod
    def is_local_signature(signature: str) -> bool:
        return signature.startswith(_THINKING_SIGNATURE_PREFIX)

    def seal(self, thinking: str, model: str) -> str:
        payload = json.dumps(
            {"model": model, "thinking": thinking, "version": 1},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > _MAX_SEALED_THINKING_BYTES:
            raise ValueError("Thinking content is too large to seal")
        compressed = zlib.compress(payload, level=6)
        nonce = os.urandom(12)
        aad = _THINKING_SIGNATURE_AAD + b":" + self._key_id.encode("ascii")
        ciphertext = self._cipher.encrypt(nonce, compressed, aad)
        token = base64.urlsafe_b64encode(nonce + ciphertext).rstrip(b"=")
        return f"{_THINKING_SIGNATURE_PREFIX}{self._key_id}.{token.decode('ascii')}"

    def unseal(self, signature: str, model: str) -> str:
        try:
            remainder = signature.removeprefix(_THINKING_SIGNATURE_PREFIX)
            key_id, encoded = remainder.split(".", 1)
            if key_id != self._key_id:
                raise ValueError
            if len(encoded) > 2 * _MAX_SEALED_THINKING_BYTES:
                raise ValueError
            padding = "=" * (-len(encoded) % 4)
            sealed = base64.b64decode(
                encoded + padding,
                altchars=b"-_",
                validate=True,
            )
            if len(sealed) < 12 + 16:
                raise ValueError
            nonce, ciphertext = sealed[:12], sealed[12:]
            aad = _THINKING_SIGNATURE_AAD + b":" + key_id.encode("ascii")
            compressed = self._cipher.decrypt(nonce, ciphertext, aad)
            payload = zlib.decompress(compressed)
            if len(payload) > _MAX_SEALED_THINKING_BYTES:
                raise ValueError
            value = json.loads(payload)
            if (
                not isinstance(value, dict)
                or value.get("version") != 1
                or value.get("model") != model
                or not isinstance(value.get("thinking"), str)
            ):
                raise ValueError
            return value["thinking"]
        except (
            InvalidTag,
            UnicodeDecodeError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            zlib.error,
        ) as error:
            raise ValueError("Invalid SGLang thinking signature") from error


def _cached_prompt_tokens(usage) -> int:
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    return getattr(prompt_tokens_details, "cached_tokens", 0) or 0


def _anthropic_input_tokens(usage) -> int:
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    cached = _cached_prompt_tokens(usage)
    if cached > prompt:
        # Upstream telemetry bug: cached cannot exceed the prompt it caches.
        # Clamping silently here would hide the discrepancy from billing
        # dashboards, so make it visible at WARNING level.
        logger.warning(
            "Cached tokens (%d) exceed prompt tokens (%d); clamping "
            "input_tokens to 0. This usually indicates an upstream "
            "telemetry bug.",
            cached,
            prompt,
        )
    return max(prompt - cached, 0)


def _anthropic_usage_from_openai(
    usage,
    *,
    include_input: bool,
    include_output: bool,
    force_zero_output: bool = False,
) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage(
            input_tokens=0 if include_input else None,
            output_tokens=0 if include_output else None,
        )

    usage_fields: dict[str, int] = {}
    cached_tokens = _cached_prompt_tokens(usage)
    if include_input:
        usage_fields["input_tokens"] = _anthropic_input_tokens(usage)
        if cached_tokens:
            usage_fields["cache_read_input_tokens"] = cached_tokens
    if include_output:
        usage_fields["output_tokens"] = (
            0 if force_zero_output else (getattr(usage, "completion_tokens", 0) or 0)
        )
    return AnthropicUsage(**usage_fields)


def _extract_system_text(
    content: Union[str, list[AnthropicContentBlock]],
) -> Optional[str]:
    """Flatten a system message's content to a trimmed string, or ``None``."""
    if isinstance(content, str):
        return content.strip() or None
    texts = []
    for block in content:
        if isinstance(block, BaseModel) and getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
        else:
            continue
        text = (text or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts) if texts else None


@dataclass(frozen=True)
class _LatestReminderMessage:
    """Internal marker emitted while normalizing Anthropic messages for DSV4."""

    content: str


def _validate_inline_system_placement(
    messages: list[AnthropicMessage],
) -> None:
    """Validate Anthropic's placement rules for inline system-message runs.

    Consecutive system messages form one section. A section must immediately
    follow a user turn (including a user turn containing tool results), and it
    must either end the request or be immediately followed by an assistant
    turn. Anthropic also permits placement after an assistant turn ending in a
    server-tool result, but those server-result block variants are not yet part
    of SGLang's Anthropic protocol union and are rejected during request
    parsing before this function runs.
    """

    index = 0
    while index < len(messages):
        if messages[index].role != "system":
            index += 1
            continue

        run_start = index
        while index < len(messages) and messages[index].role == "system":
            index += 1

        if run_start == 0 or messages[run_start - 1].role != "user":
            raise ValueError(
                "A mid-conversation system message must immediately follow a "
                "user turn (placement after assistant server-tool results is "
                "not yet supported)"
            )
        if index < len(messages) and messages[index].role != "assistant":
            raise ValueError(
                "A mid-conversation system message must end messages or be "
                "immediately followed by an assistant turn"
            )


def _text_only_inline_system_content(
    content: Union[str, list[AnthropicContentBlock]],
) -> str:
    """Return DSV4 reminder text without silently discarding content blocks."""

    if isinstance(content, str):
        text = content
    else:
        texts: list[str] = []
        for block in content:
            block_type = (
                getattr(block, "type", None)
                if isinstance(block, BaseModel)
                else block.get("type")
            )
            if block_type != "text":
                raise ValueError(
                    "DeepSeek-V4 inline system messages support text content only; "
                    f"got content block type {block_type!r}"
                )
            block_text = (
                getattr(block, "text", "")
                if isinstance(block, BaseModel)
                else block.get("text", "")
            )
            texts.append(block_text)
        text = "\n".join(texts)

    if not text.strip():
        raise ValueError(
            "DeepSeek-V4 inline system messages must contain non-empty text"
        )
    return text


def _normalize_dsv4_inline_system_messages(
    messages: list[AnthropicMessage],
) -> list[Union[AnthropicMessage, _LatestReminderMessage]]:
    """Map valid Anthropic system sections to DSV4 ``latest_reminder``.

    DeepSeek-V4's reference encoder places ``latest_reminder`` before the user
    turn it qualifies. Normalize at the Anthropic-message boundary so a user
    containing text plus one or more tool results remains a single logical
    group when the existing converter later expands it to OpenAI messages.
    """

    normalized: list[Union[AnthropicMessage, _LatestReminderMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != "user":
            normalized.append(message)
            index += 1
            continue

        user_end = index + 1
        while user_end < len(messages) and messages[user_end].role == "user":
            user_end += 1

        system_start = user_end
        system_end = system_start
        while system_end < len(messages) and messages[system_end].role == "system":
            system_end += 1

        if system_end == system_start:
            normalized.extend(messages[index:user_end])
            index = user_end
            continue

        reminder_parts = [
            _text_only_inline_system_content(messages[i].content)
            for i in range(system_start, system_end)
        ]
        reminder = _LatestReminderMessage(content="\n".join(reminder_parts))
        normalized.append(reminder)
        normalized.extend(messages[index:user_end])
        index = system_end

    return normalized


def _wrap_sse_event(data: str, event_type: str) -> str:
    """Format an Anthropic SSE event with event type and data lines."""
    return f"event: {event_type}\ndata: {data}\n\n"


def _scrub_error_message(message: str, status_code: int) -> str:
    """Return a safe outward-facing error message.

    5xx is always generic — never echo upstream ``str(e)`` payloads, which
    may contain stack frames, file paths, or PII. 4xx keeps the original
    message (truncated and with obvious traceback lines stripped) so
    callers see the real validation failure.
    """
    if status_code >= 500:
        return "Internal server error"
    if not message:
        return "Request failed"
    safe_lines = [
        ln
        for ln in message.splitlines()
        if not ln.startswith("Traceback") and 'File "/' not in ln
    ]
    cleaned = "\n".join(safe_lines).strip()
    if len(cleaned) > 500:
        cleaned = cleaned[:500] + "…"
    return cleaned or "Request failed"


def _openai_error_message(response: Any) -> Optional[str]:
    """Extract an OpenAI error message without exposing it to success paths."""

    direct_message = getattr(response, "message", None)
    if isinstance(direct_message, str):
        return direct_message
    body = getattr(response, "body", b"") or b""
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    error_payload = payload.get("error", payload)
    if isinstance(error_payload, dict):
        message = error_payload.get("message")
        return message if isinstance(message, str) else None
    return error_payload if isinstance(error_payload, str) else None


class AnthropicServing:
    """Handler for Anthropic Messages API requests.

    Acts as a translation layer between Anthropic's Messages API and SGLang's
    OpenAI-compatible chat completion infrastructure.
    """

    def __init__(self, openai_serving_chat: OpenAIServingChat):
        self.openai_serving_chat = openai_serving_chat
        recovery_flag = os.getenv(_TOOL_RECOVERY_ENV, "0").strip().lower()
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
        logger.info(
            "Anthropic bounded tool-call recovery: %s timeout_s=%d "
            "max_tokens=%d concurrency=%d ping_interval_s=%d",
            "enabled" if self._tool_recovery_enabled else "disabled",
            self._tool_recovery_timeout_s,
            self._tool_recovery_max_tokens,
            recovery_concurrency,
            self._tool_recovery_ping_interval_s,
        )
        self._thinking_signature_codec = _ThinkingSignatureCodec.from_environment()
        if self._thinking_signature_codec is not None:
            logger.info(
                "Anthropic omitted-thinking signatures enabled (key id %s)",
                self._thinking_signature_codec.key_id,
            )
        custom_strategy = custom_encoder_inline_system_strategy(
            getattr(openai_serving_chat, "chat_encoding_spec", None)
        )
        if custom_strategy is not None:
            self._inline_system_strategy = custom_strategy
        elif detect_inline_system_support(self._chat_template()):
            self._inline_system_strategy = InlineSystemStrategy.PASSTHROUGH
        else:
            self._inline_system_strategy = InlineSystemStrategy.MERGE

        self._merge_inline_system = (
            self._inline_system_strategy == InlineSystemStrategy.MERGE
        )
        logger.info(
            "Anthropic inline system-message strategy: %s",
            self._inline_system_strategy.value,
        )

    def _chat_template(self) -> Optional[str]:
        tokenizer_manager = getattr(self.openai_serving_chat, "tokenizer_manager", None)
        if tokenizer_manager is None:
            return None
        tokenizer = getattr(tokenizer_manager, "tokenizer", None)
        if tokenizer is None:
            return None
        return getattr(tokenizer, "chat_template", None)

    @staticmethod
    def _validated_recovery_calls(
        response: ChatCompletionResponse,
        anthropic_request: AnthropicMessagesRequest,
        recovery_rid: str,
        allowed_tool_names: Optional[set[str]] = None,
    ) -> _AnthropicToolRecoveryResult:
        """Extract an all-or-nothing, wire-valid tool suffix from a retry.

        Original ``strict`` semantics remain authoritative: a non-strict
        schema mismatch may reach the client, while strict mismatches fail.
        """

        if not response.choices or not response.choices[0].message.tool_calls:
            raise MalformedToolCallError(
                "Constrained tool-call recovery produced no tool call"
            )
        if response.choices[0].finish_reason not in {"tool_calls", "stop"}:
            raise MalformedToolCallError(
                "Constrained tool-call recovery ended before completing tool calls"
            )

        recovered: list[_RecoveredAnthropicToolCall] = []
        for tool_call in response.choices[0].message.tool_calls:
            if (
                allowed_tool_names is not None
                and tool_call.function.name not in allowed_tool_names
            ):
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

            _validate_anthropic_tool_input(
                anthropic_request,
                tool_call.function.name,
                tool_input,
            )
            recovered.append(
                _RecoveredAnthropicToolCall(
                    id=tool_call.id or f"toolu_{uuid.uuid4().hex}",
                    name=tool_call.function.name,
                    tool_input=tool_input,
                    argument_json=json.dumps(
                        tool_input,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

        return _AnthropicToolRecoveryResult(
            calls=tuple(recovered),
            usage=response.usage,
            model=response.model,
            recovery_rid=recovery_rid,
            response=response,
        )

    def _response_from_recovery(
        self,
        result: _AnthropicToolRecoveryResult,
        anthropic_request: AnthropicMessagesRequest,
    ) -> AnthropicMessagesResponse:
        # No part of the failed non-streaming attempt reached the client, so
        # preserve the validated retry's thinking/signature/text exactly as a
        # normal successful response. Streaming recovery intentionally emits
        # only ``result.calls`` because its original prefix is already on wire.
        return self._convert_response(result.response, anthropic_request)

    async def _attempt_tool_call_recovery(
        self,
        processed_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
        reason: _ToolRecoveryReason,
        original_rid: Optional[str],
        candidate_tool_names: tuple[str, ...] = (),
    ) -> Optional[_AnthropicToolRecoveryResult]:
        """Run one isolated, grammar-constrained tool-only regeneration.

        The caller owns the one-attempt bound. This method always assigns a
        fresh RID, disables SGLang session mutation, forces native tool grammar
        plus parameter schemas, and accepts only a wire-valid tool suffix.
        The streaming caller intentionally discards retry reasoning/text; the
        non-streaming caller can preserve the complete validated retry.
        """

        if not self._tool_recovery_enabled or not processed_request.tools:
            return None
        if processed_request.tool_choice == "none":
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
                "Anthropic tool recovery skipped for PD-disaggregated request; "
                "a fresh coordinated bootstrap room is required"
            )
            return None

        # Scheduler aborts intentionally match RID prefixes so one parent can
        # cancel a generated batch. The original RID may be header-controlled,
        # so a fixed recovery prefix is unsafe (for example, an original RID
        # of ``toolrecovery_`` would also abort ``toolrecovery_<uuid>``). Put
        # entropy first and reject the rare prefix collision explicitly.
        while True:
            recovery_rid = f"{uuid.uuid4().hex}_toolrecovery"
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
            "Anthropic tool recovery started: reason=%s original_rid=%s "
            "recovery_rid=%s",
            reason.value,
            original_rid,
            recovery_rid,
        )

        async def _run_recovery():
            async with self._tool_recovery_slots:
                # Long Claude Code requests can contain hundreds of tools and
                # a near-full context. Keep the deep copy behind the admission
                # gate so a failure burst cannot multiply that memory by every
                # queued retry while only a few decodes are allowed to run.
                recovery_request = processed_request.model_copy(deep=True)
                recovery_request.rid = recovery_rid
                recovery_request.stream = False
                recovery_request.stream_options = None
                recovery_request.n = 1
                recovery_request.max_tokens = min(
                    recovery_request.max_tokens,
                    self._tool_recovery_max_tokens,
                )
                recovery_request.session_id = None
                recovery_request.session_params = None

                if original_named_tool is not None:
                    # The caller's named choice is authoritative even if
                    # malformed output mentioned a different declared tool.
                    recovery_request.tools = [
                        tool
                        for tool in (recovery_request.tools or [])
                        if tool.function.name == original_named_tool
                    ]
                elif observed_declared_names:
                    # Keep recovery inside the exact intent set already
                    # observed in the unpublished V10 transaction. In auto
                    # mode, required-any over the full catalog could invent a
                    # different side-effecting call.
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
                    # A name already parsed into V10's unpublished transaction
                    # is strong evidence of intent and can safely narrow the
                    # regeneration.
                    candidate_tool_name = next(iter(observed_declared_names))
                    recovery_request.tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(name=candidate_tool_name),
                    )
                elif len(observed_declared_names) > 1:
                    recovery_request.tool_choice = "required"
                elif not isinstance(recovery_request.tool_choice, ToolChoice):
                    # Preserve auto mode when intent is unknown. Marking the
                    # tools strict below gives auto mode a triggered structural
                    # grammar, but still permits an ordinary final response.
                    # That response is discarded and recovery fails closed; we
                    # never invent an arbitrary side-effecting tool call.
                    recovery_request.tool_choice = (
                        "required"
                        if recovery_request.tool_choice == "required"
                        else "auto"
                    )
                for tool in recovery_request.tools or []:
                    tool.function = tool.function.model_copy(update={"strict": True})

                adapted_request, normalized_request = (
                    self.openai_serving_chat._convert_to_internal_request(
                        recovery_request,
                        raw_request,
                    )
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
                # Header overrides are allowed on the original request, but a
                # retry must never reuse x-override-rid and collide with deleted
                # state.
                adapted_request.rid = recovery_rid
                adapted_request.received_time = monotonic_time()
                return await self.openai_serving_chat._handle_non_streaming_request(
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
                "Anthropic tool recovery timed out: reason=%s recovery_rid=%s "
                "timeout_s=%d",
                reason.value,
                recovery_rid,
                self._tool_recovery_timeout_s,
            )
            abort_request = getattr(
                getattr(self.openai_serving_chat, "tokenizer_manager", None),
                "abort_request",
                None,
            )
            if abort_request is not None:
                try:
                    abort_request(recovery_rid)
                except Exception:
                    logger.exception(
                        "Failed to abort timed-out Anthropic recovery rid=%s",
                        recovery_rid,
                    )
            return None
        except Exception as e:
            logger.exception(
                "Anthropic tool recovery raised: reason=%s recovery_rid=%s "
                "exception=%s",
                reason.value,
                recovery_rid,
                type(e).__name__,
            )
            return None

        if not isinstance(response, ChatCompletionResponse):
            logger.warning(
                "Anthropic tool recovery failed upstream: reason=%s "
                "recovery_rid=%s response_type=%s",
                reason.value,
                recovery_rid,
                type(response).__name__,
            )
            return None

        try:
            result = self._validated_recovery_calls(
                response,
                anthropic_request,
                recovery_rid,
                allowed_tool_names,
            )
        except MalformedToolCallError as e:
            logger.warning(
                "Anthropic tool recovery rejected: reason=%s recovery_rid=%s "
                "validation=%s",
                reason.value,
                recovery_rid,
                e,
            )
            return None

        logger.warning(
            "Anthropic tool recovery succeeded: reason=%s original_rid=%s "
            "recovery_rid=%s tool_calls=%d",
            reason.value,
            original_rid,
            recovery_rid,
            len(result.calls),
        )
        return result

    async def handle_messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[JSONResponse, StreamingResponse]:
        """Main entry point for /v1/messages endpoint."""
        if self._request_requires_thinking_codec(request):
            if self._thinking_signature_codec is None:
                logger.error(
                    "Anthropic omitted-thinking continuity requested but %s "
                    "is not configured",
                    _THINKING_KEY_FILE_ENV,
                )
                return self._error_response(
                    status_code=500,
                    error_type="api_error",
                    message="Internal server error",
                )
        try:
            chat_request = self._convert_to_chat_completion_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting Anthropic request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        if request.stream:
            return await self._handle_streaming(chat_request, request, raw_request)
        else:
            return await self._handle_non_streaming(chat_request, request, raw_request)

    @staticmethod
    def _request_requires_thinking_codec(
        request: AnthropicMessagesRequest,
    ) -> bool:
        if request.thinking is not None and request.thinking.display == "omitted":
            return True
        return AnthropicServing._request_contains_local_thinking_signature(request)

    @staticmethod
    def _request_contains_local_thinking_signature(
        request: AnthropicMessagesRequest,
    ) -> bool:
        return any(
            block.type == "thinking"
            and bool(block.signature)
            and _ThinkingSignatureCodec.is_local_signature(block.signature)
            for message in request.messages
            if isinstance(message.content, list)
            for block in message.content
        )

    def _convert_to_chat_completion_request(
        self, anthropic_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """Convert an Anthropic Messages request to an OpenAI ChatCompletion request."""
        openai_messages = []

        _validate_inline_system_placement(anthropic_request.messages)
        if self._inline_system_strategy == InlineSystemStrategy.LATEST_REMINDER:
            conversion_messages = _normalize_dsv4_inline_system_messages(
                anthropic_request.messages
            )
        else:
            conversion_messages = anthropic_request.messages

        def _convert_anthropic_image_source_to_openai_part(
            source: Any,
        ) -> Optional[dict]:
            # Source may arrive as a Pydantic model (typed ImageBlock.source)
            # or as a raw dict when parsed from a nested tool_result payload.
            if isinstance(source, BaseModel):
                source = source.model_dump(exclude_none=True)
            if not isinstance(source, dict):
                return None

            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                if not data:
                    return None
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{data}",
                    },
                }

            url = source.get("url")
            if url:
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                }

            return None

        def _text_from_search_result(item: dict[str, Any]) -> str:
            search_parts = []
            title = item.get("title")
            if title:
                search_parts.append(f"Title: {title}")

            source = item.get("source")
            if isinstance(source, dict):
                source_text = source.get("url") or source.get("text")
                if source_text:
                    search_parts.append(f"Source: {source_text}")
            elif source:
                search_parts.append(f"Source: {source}")

            content = item.get("content")
            content_parts = []
            if isinstance(content, str):
                content_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and part.get("text"):
                        content_parts.append(part["text"])
            if content_parts:
                search_parts.append("Content: " + "\n".join(content_parts))

            return "\n".join(search_parts)

        def _convert_tool_result_content(
            content: Any,
        ) -> tuple[list[Union[str, list[dict]]], str]:
            if isinstance(content, list):
                tool_content_parts = []
                tool_text_parts = []

                for raw_item in content:
                    # Items may be typed Pydantic blocks (after request
                    # validation) or raw dicts (from legacy callers). Coerce
                    # to dict so the existing key-based logic still works.
                    if isinstance(raw_item, BaseModel):
                        item = raw_item.model_dump(exclude_none=True)
                    elif isinstance(raw_item, dict):
                        item = raw_item
                    else:
                        continue

                    item_type = item.get("type")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            tool_text_parts.append(text)
                            tool_content_parts.append({"type": "text", "text": text})
                    elif item_type == "image":
                        image_part = _convert_anthropic_image_source_to_openai_part(
                            item.get("source")
                        )
                        if image_part is not None:
                            tool_content_parts.append(image_part)
                    elif item_type == "tool_reference":
                        # Anthropic uses `tool_name`; the SGLang chat template
                        # matches on `name`. Translate at the boundary.
                        ref_name = item.get("tool_name") or item.get("name")
                        if ref_name:
                            tool_content_parts.append(
                                {"type": "tool_reference", "name": ref_name}
                            )
                    elif item_type == "search_result":
                        search_text = _text_from_search_result(item)
                        if search_text:
                            tool_text_parts.append(search_text)
                            tool_content_parts.append(
                                {"type": "text", "text": search_text}
                            )

                tool_text = "\n".join(tool_text_parts)
                # GLM templates expand references only at the start of a tool
                # message, so isolate reference runs without changing part order.
                tool_content_groups: list[list[dict]] = []
                for part in tool_content_parts:
                    is_reference = part["type"] == "tool_reference"
                    if (
                        not tool_content_groups
                        or (tool_content_groups[-1][0]["type"] == "tool_reference")
                        != is_reference
                    ):
                        tool_content_groups.append([])
                    tool_content_groups[-1].append(part)

                tool_contents: list[Union[str, list[dict]]] = []
                for group in tool_content_groups:
                    if len(group) == 1 and group[0]["type"] == "text":
                        tool_contents.append(group[0]["text"])
                    else:
                        tool_contents.append(group)
                return tool_contents or [""], tool_text

            tool_text = str(content) if content else ""
            return [tool_text], tool_text

        def _convert_assistant_thinking_blocks(
            blocks: list[AnthropicContentBlock],
        ) -> tuple[Optional[str], Optional[str]]:
            """Reconstruct prior-turn thinking as ``(reasoning_content, text)``.

            At most one is set: encoders that frame the reasoning channel take
            it as ``reasoning_content``, everything else gets it re-wrapped and
            spliced into content.

            ``redacted_thinking`` carries encrypted bytes that no local
            parser can interpret, so we raise rather than silently drop it.
            On non-reasoning models (no detector configured) the rewrap is
            best-effort: we log a warning and drop the thinking text so a
            history echo doesn't 400 the whole request — the prior thinking
            is opaque context the model didn't need anyway.
            """
            if any(block.type == "redacted_thinking" for block in blocks):
                raise ValueError("Anthropic redacted_thinking history is not supported")

            thinking_parts: list[str] = []
            for block in blocks:
                if block.type != "thinking":
                    continue
                if block.signature and _ThinkingSignatureCodec.is_local_signature(
                    block.signature
                ):
                    if self._thinking_signature_codec is None:
                        raise RuntimeError(
                            "SGLang thinking-signature key is not configured"
                        )
                    thinking_parts.append(
                        self._thinking_signature_codec.unseal(
                            block.signature,
                            anthropic_request.model,
                        )
                    )
                elif block.thinking:
                    # Visible/summarized thinking, including history produced by
                    # pre-sealing SGLang versions, remains directly replayable.
                    thinking_parts.append(block.thinking)
                elif block.signature:
                    raise ValueError(
                        "Opaque thinking signature was not issued by this "
                        "SGLang endpoint"
                    )
            if not thinking_parts:
                return None, None

            reasoning_text = "\n".join(thinking_parts)
            if self.openai_serving_chat.supports_native_reasoning_history():
                return reasoning_text, None

            try:
                return None, self.openai_serving_chat.wrap_reasoning_history(
                    reasoning_text
                )
            except ValueError as e:
                logger.warning(
                    "Dropping prior-turn thinking history (%d blocks): %s",
                    len(thinking_parts),
                    e,
                )
                return None, None

        system_parts: list[str] = []
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                if anthropic_request.system.strip():
                    system_parts.append(anthropic_request.system)
            else:
                for block in anthropic_request.system:
                    if block.type == "text" and block.text:
                        system_parts.append(block.text)

        if self._merge_inline_system:
            for msg in anthropic_request.messages:
                if msg.role != "system":
                    continue
                text = _extract_system_text(msg.content)
                if text:
                    system_parts.append(text)

        if system_parts:
            openai_messages.append(
                {"role": "system", "content": "\n".join(system_parts)}
            )

        def _emit_user_message(parts: list[dict]) -> None:
            """Append accumulated parts as a user message, then clear them.

            Used to flush content collected BEFORE a tool_result so the
            wire order stays user(pre) → tool → user(post). Without this
            flush, text/image parts that appeared before a tool_result
            block would be moved AFTER the tool message at end of loop.
            """
            if not parts:
                return
            if len(parts) == 1 and parts[0]["type"] == "text":
                openai_messages.append({"role": "user", "content": parts[0]["text"]})
            else:
                openai_messages.append({"role": "user", "content": list(parts)})
            parts.clear()

        # Convert messages
        for msg in conversion_messages:
            if isinstance(msg, _LatestReminderMessage):
                openai_messages.append(
                    {"role": "latest_reminder", "content": msg.content}
                )
                continue
            if msg.role == "system" and self._merge_inline_system:
                continue
            if isinstance(msg.content, str):
                openai_messages.append({"role": msg.role, "content": msg.content})
                continue

            # Complex content with blocks
            openai_msg = {"role": msg.role}
            content_parts: list[dict] = []
            tool_calls: list[dict] = []

            if msg.role == "assistant":
                reasoning_content, reasoning_history = (
                    _convert_assistant_thinking_blocks(msg.content)
                )
                if reasoning_content is not None:
                    openai_msg["reasoning_content"] = reasoning_content
                if reasoning_history is not None:
                    content_parts.append({"type": "text", "text": reasoning_history})

            for block in msg.content:
                # ``thinking``/``redacted_thinking`` blocks are surfaced via
                # the reasoning-history reconstruction above; skip them here
                # to avoid double-injecting their text into the prompt.
                if block.type in ("thinking", "redacted_thinking"):
                    continue

                # ``is not None`` (not truthy) so an empty-string text block
                # still produces a placeholder text part — without it, an
                # assistant turn whose only content is "" vanishes and
                # subsequent user→user pairs trip strict chat templates.
                if block.type == "text" and block.text is not None:
                    content_parts.append({"type": "text", "text": block.text})

                elif block.type == "image" and block.source:
                    image_part = _convert_anthropic_image_source_to_openai_part(
                        block.source
                    )
                    if image_part is not None:
                        content_parts.append(image_part)

                elif block.type == "search_result":
                    search_text = _text_from_search_result(block.model_dump())
                    if search_text:
                        content_parts.append({"type": "text", "text": search_text})

                elif block.type == "tool_use":
                    tool_call = {
                        "id": block.id or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                    tool_calls.append(tool_call)

                elif block.type == "tool_result":
                    tool_contents, tool_text = _convert_tool_result_content(
                        block.content
                    )

                    # Use tool_use_id (per spec) with fallback to id
                    tool_call_id = block.tool_use_id or block.id or ""

                    # Tool results from user become separate tool messages.
                    # Flush any pending text/image first so the wire order
                    # is preserved (a tool_result that arrived AFTER a text
                    # block must come AFTER that text in OpenAI form too).
                    if msg.role == "user":
                        _emit_user_message(content_parts)
                        for tool_content in tool_contents:
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": tool_content,
                                    "is_error": block.is_error,
                                }
                            )
                    else:
                        content_parts.append(
                            {
                                "type": "text",
                                "text": f"Tool result: {tool_text}",
                            }
                        )

            # Attach tool calls to assistant messages
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls

            # Attach content
            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    openai_msg["content"] = content_parts[0]["text"]
                else:
                    openai_msg["content"] = content_parts
                openai_messages.append(openai_msg)
            elif tool_calls:
                openai_messages.append(openai_msg)
            elif msg.role == "user":
                # User turn that was entirely tool_results — the tool
                # messages were already emitted above, nothing left.
                continue
            else:
                # Assistant turn with no content and no tool_calls: emit
                # an empty-string placeholder so strict templates still
                # see a valid role-alternation sequence.
                openai_msg["content"] = ""
                openai_messages.append(openai_msg)

        # Build ChatCompletionRequest
        request_data = {
            "messages": openai_messages,
            "model": anthropic_request.model,
            "max_tokens": anthropic_request.max_tokens,
            "stream": anthropic_request.stream or False,
        }

        if anthropic_request.temperature is not None:
            request_data["temperature"] = anthropic_request.temperature
        if anthropic_request.top_p is not None:
            request_data["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            request_data["top_k"] = anthropic_request.top_k
        if anthropic_request.stop_sequences is not None:
            request_data["stop"] = anthropic_request.stop_sequences

        # Enable usage in stream so we can report it
        if anthropic_request.stream:
            request_data["stream_options"] = StreamOptions(
                include_usage=True,
                continuous_usage_stats=True,
            )

        chat_request = ChatCompletionRequest(**request_data)

        if anthropic_request.thinking is not None:
            # The protocol layer already enforces SDK shape:
            #   enabled  -> budget_tokens required (>=1024), display optional
            #   disabled -> neither budget_tokens nor display allowed
            #   adaptive -> budget_tokens forbidden, display optional
            # So by the time we get here ``budget_tokens`` can only be
            # set on ``enabled``. The local backend has no equivalent
            # hard-cap knob, so we log a WARNING instead of rejecting —
            # the Anthropic SDK would have accepted the request and we
            # mirror that. Operators see the unenforced budget in logs.
            if anthropic_request.thinking.budget_tokens is not None:
                logger.warning(
                    "Anthropic thinking.budget_tokens=%d is accepted for "
                    "SDK compatibility but the local backend has no "
                    "equivalent hard-cap knob — the budget is not enforced",
                    anthropic_request.thinking.budget_tokens,
                )
            # Claude 4.7's ``adaptive`` is treated identically to ``enabled``
            # because the local backend has no auto-throttle equivalent.
            # Anything other than ``disabled`` enables reasoning.
            enabled = anthropic_request.thinking.type != "disabled"
            self.openai_serving_chat.apply_reasoning_enabled(chat_request, enabled)

        # Claude 4.7 ``output_config``: map ``effort`` onto the OpenAI
        # ``reasoning_effort`` knob. ``xhigh`` collapses to ``max`` because
        # the OpenAI Literal does not include the Anthropic-only ``xhigh``.
        # ``task_budget`` is a soft hint forwarded as a custom param so the
        # model can see it without it becoming a hard cap (``max_tokens``
        # is still the hard cap).
        if anthropic_request.output_config is not None:
            oc = anthropic_request.output_config
            if oc.effort is not None:
                chat_request.reasoning_effort = (
                    "max" if oc.effort == "xhigh" else oc.effort
                )
            if oc.task_budget is not None:
                # Custom params are silently ignored by backends that
                # don't recognise them; logging it makes the propagation
                # visible.
                logger.info(
                    "Anthropic output_config.task_budget hint: %d %s",
                    oc.task_budget.total,
                    oc.task_budget.type,
                )

        # ``betas`` is the Anthropic SDK's opt-in feature list (e.g.
        # ``["thinking-2025-08-04"]``). The local backend has no
        # equivalent beta system; accept-and-log so requests don't 400.
        if anthropic_request.betas:
            logger.info(
                "Anthropic request opted into betas %s — no-op locally",
                anthropic_request.betas,
            )

        # Convert tools. Deferred tools stay in the list with defer_loading=True;
        # the chat template hides them from the initial <tools> block and renders
        # them on demand when a tool_reference block names them.
        if anthropic_request.tools:
            converted_tools = []
            for tool in anthropic_request.tools:
                if is_server_tool(tool):
                    # Anthropic server-side tools (web_search_*, computer_*,
                    # bash_*, text_editor_*) have no client-side input_schema
                    # because Anthropic provides the implementation. We can't
                    # forward them to the OpenAI tools array (which requires a
                    # schema), so skip with a visible log.
                    logger.info(
                        "Skipping built-in Anthropic server tool %r (type=%r): "
                        "no native support in the OpenAI-compatible backend",
                        tool.name,
                        tool.type,
                    )
                    continue

                # Custom tools always carry an input_schema (enforced at
                # Pydantic parse time); validate that it is legal JSON Schema
                # before any prompt is rendered or generation is submitted.
                try:
                    Draft202012Validator.check_schema(tool.input_schema)
                except JSONSchemaError as e:
                    raise ValueError(
                        f"Anthropic tool {tool.name!r} has an invalid input_schema"
                    ) from e
                function_data = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                }
                if "strict" in tool.model_fields_set:
                    # Preserve absence as absence: explicitly serializing
                    # strict=false changes the DSV4 reference prompt even
                    # though the Anthropic caller never requested strict use.
                    function_data["strict"] = tool.strict
                converted_tools.append(
                    Tool(
                        type="function",
                        defer_loading=tool.defer_loading,
                        function=function_data,
                    )
                )

            if converted_tools:
                chat_request.tools = converted_tools

        # Convert tool choice. ``any``/``tool`` express a hard requirement
        # ("the model MUST call a tool"); if every requested tool was a
        # server-side Anthropic built-in that we just skipped, there is
        # no tool the model could call. Silently downgrading to "no tool"
        # would deceive the caller, so raise an explicit 400.
        if anthropic_request.tool_choice is not None:
            tc_type = anthropic_request.tool_choice.type
            if tc_type == "none":
                chat_request.tool_choice = "none"
            elif chat_request.tools:
                if tc_type == "auto":
                    chat_request.tool_choice = "auto"
                elif tc_type == "any":
                    chat_request.tool_choice = "required"
                elif tc_type == "tool":
                    tool_name = anthropic_request.tool_choice.name
                    # ``Tool.function`` is a ``Function`` Pydantic model, not
                    # a dict — access by attribute. A dict ``.get`` would
                    # AttributeError and surface as a 500 instead of the
                    # intended 400 / happy path.
                    if not any(
                        t.function.name == tool_name for t in chat_request.tools
                    ):
                        raise ValueError(
                            f"tool_choice references tool {tool_name!r} but it "
                            f"is not in the forwarded tools list "
                            f"(server-side Anthropic tools cannot be selected)"
                        )
                    chat_request.tool_choice = ToolChoice(
                        type="function",
                        function=ToolChoiceFuncName(name=tool_name),
                    )
            elif tc_type in ("any", "tool"):
                raise ValueError(
                    f"tool_choice={tc_type!r} requires at least one custom "
                    f"tool; all supplied tools were server-side Anthropic "
                    f"built-ins which the OpenAI-compatible backend cannot "
                    f"invoke"
                )
        elif chat_request.tools:
            chat_request.tool_choice = "auto"

        return chat_request

    async def _handle_non_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle non-streaming Anthropic request by delegating to OpenAI handler."""
        # ``monotonic_time`` is ``time.perf_counter`` under the hood; the
        # downstream stats layer subtracts other ``perf_counter`` samples
        # from this, so they must come from the same clock.
        received_time = monotonic_time()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            # Convert to internal request
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.received_time = received_time

            # Get response from OpenAI handler
            response = await self.openai_serving_chat._handle_non_streaming_request(
                adapted_request, processed_request, raw_request
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error processing Anthropic request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
            )

        # Check for error responses from OpenAI handler
        if not isinstance(response, ChatCompletionResponse):
            error_message = _openai_error_message(response)
            recovery_reason = _classify_tool_recovery_reason(error_message or "")
            if recovery_reason is not None:
                recovered = await self._attempt_tool_call_recovery(
                    processed_request,
                    anthropic_request,
                    raw_request,
                    recovery_reason,
                    getattr(adapted_request, "rid", None),
                )
                if recovered is not None:
                    anthropic_response = self._response_from_recovery(
                        recovered,
                        anthropic_request,
                    )
                    return JSONResponse(
                        content=anthropic_response.model_dump(exclude_none=True)
                    )
            # It's an error response (ORJSONResponse).
            return self._convert_openai_error_response(response)

        # Convert to Anthropic response. Model-produced tool calls are validated
        # at the same atomic boundary as streaming responses; malformed input
        # must not be rewritten to an indistinguishable empty tool call.
        try:
            anthropic_response = self._convert_response(response, anthropic_request)
        except MalformedToolCallError as e:
            logger.warning("Rejecting non-streaming Anthropic tool call: %s", e)
            candidate_tool_names = tuple(
                dict.fromkeys(
                    tool_call.function.name
                    for choice in response.choices
                    for tool_call in (choice.message.tool_calls or [])
                )
            )
            recovered = await self._attempt_tool_call_recovery(
                processed_request,
                anthropic_request,
                raw_request,
                _classify_tool_recovery_reason(str(e))
                or _ToolRecoveryReason.MALFORMED_ARGUMENTS,
                getattr(adapted_request, "rid", None),
                candidate_tool_names,
            )
            if recovered is not None:
                anthropic_response = self._response_from_recovery(
                    recovered,
                    anthropic_request,
                )
                return JSONResponse(
                    content=anthropic_response.model_dump(exclude_none=True)
                )
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message=str(e),
            )
        return JSONResponse(content=anthropic_response.model_dump(exclude_none=True))

    async def _handle_streaming(
        self,
        chat_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, JSONResponse]:
        """Handle streaming Anthropic request."""
        received_time = monotonic_time()

        # Validate
        error_msg = self.openai_serving_chat._validate_request(chat_request)
        if error_msg:
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=error_msg,
            )

        try:
            adapted_request, processed_request = (
                self.openai_serving_chat._convert_to_internal_request(
                    chat_request, raw_request
                )
            )
            adapted_request.received_time = received_time
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting streaming request: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
            )

        return StreamingResponse(
            self._generate_anthropic_stream(
                adapted_request,
                processed_request,
                anthropic_request,
                raw_request,
            ),
            media_type="text/event-stream",
            background=self.openai_serving_chat.tokenizer_manager.create_abort_task(
                adapted_request
            ),
        )

    async def _generate_anthropic_stream(
        self,
        adapted_request,
        processed_request: ChatCompletionRequest,
        anthropic_request: AnthropicMessagesRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Own both adapter layers for the full lifetime of the HTTP stream."""
        openai_stream = self.openai_serving_chat._generate_chat_stream(
            adapted_request, processed_request, raw_request
        )

        async def _recover_tool_suffix(
            reason: _ToolRecoveryReason,
            candidate_tool_names: tuple[str, ...],
        ) -> Optional[_AnthropicToolRecoveryResult]:
            return await self._attempt_tool_call_recovery(
                processed_request,
                anthropic_request,
                raw_request,
                reason,
                getattr(adapted_request, "rid", None),
                candidate_tool_names,
            )

        translation_stream = self._translate_openai_stream(
            openai_stream,
            anthropic_request,
            tool_recovery_callback=_recover_tool_suffix,
        )
        try:
            async for frame in translation_stream:
                yield frame
        finally:
            # Client disconnects close this outer generator directly.  Close
            # both nested generators here as well as on their explicit error
            # paths so ownership never depends on async-generator GC.
            await translation_stream.aclose()
            close = getattr(openai_stream, "aclose", None)
            if close is not None:
                try:
                    await close()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Failed to close child OpenAI generation stream")

    async def _translate_openai_stream(
        self,
        openai_stream,
        anthropic_request: AnthropicMessagesRequest,
        tool_recovery_callback: Optional[
            Callable[
                [_ToolRecoveryReason, tuple[str, ...]],
                Awaitable[Optional[_AnthropicToolRecoveryResult]],
            ]
        ] = None,
    ) -> AsyncGenerator[str, None]:
        """Convert OpenAI chat stream to Anthropic event stream."""
        upstream_closed = False

        async def _close_openai_stream() -> None:
            """Deterministically release the child OpenAI generation stream.

            This adapter frequently terminates after translating an upstream
            error into an Anthropic SSE error.  Returning while the child async
            generator is suspended on its error chunk delays its ``finally``
            cleanup until garbage collection, which can leave scheduler work
            decoding after the Messages response has already ended.
            """
            nonlocal upstream_closed
            if upstream_closed:
                return
            upstream_closed = True
            close = getattr(openai_stream, "aclose", None)
            if close is None:
                return
            try:
                await close()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to close child OpenAI generation stream")

        content_block_index = 0
        content_block_open = False
        content_block_type: Optional[str] = None
        captured_thinking_parts: list[str] = []
        finish_reason: Optional[str] = None
        final_usage: Optional[AnthropicUsage] = None
        message_started = False
        had_content_delta = False
        pending_tool_calls: dict[int, _PendingAnthropicToolCall] = {}
        pending_segments: list[_PendingAnthropicSegment] = []
        active_tool_call_index: Optional[int] = None
        next_tool_call_index = 0
        recovery_attempted = False
        recovery_succeeded = False
        message_id = f"msg_{uuid.uuid4().hex}"
        model = anthropic_request.model
        omit_thinking = bool(
            anthropic_request.thinking is not None
            and anthropic_request.thinking.display == "omitted"
        )

        def _message_start_event(usage) -> MessageStartEvent:
            return MessageStartEvent(
                message=AnthropicMessagesResponse(
                    id=message_id,
                    content=[],
                    model=model,
                    usage=_anthropic_usage_from_openai(
                        usage,
                        include_input=True,
                        include_output=True,
                        force_zero_output=True,
                    ),
                ),
            )

        def _emit(event: AnthropicStreamEvent) -> str:
            return _wrap_sse_event(
                event.model_dump_json(exclude_none=True),
                event.type,
            )

        def _close_content_block_events() -> list[AnthropicStreamEvent]:
            nonlocal content_block_index, content_block_open
            nonlocal content_block_type, captured_thinking_parts

            events: list[AnthropicStreamEvent] = []
            if not content_block_open:
                return events

            if content_block_type == "thinking" and omit_thinking:
                if self._thinking_signature_codec is None:
                    raise RuntimeError(
                        "SGLang thinking-signature key is not configured"
                    )
                signature = self._thinking_signature_codec.seal(
                    "".join(captured_thinking_parts),
                    model,
                )
                events.append(
                    ContentBlockDeltaEvent(
                        index=content_block_index,
                        delta=SignatureDelta(
                            signature=signature,
                        ),
                    )
                )

            events.append(ContentBlockStopEvent(index=content_block_index))
            content_block_open = False
            content_block_type = None
            content_block_index += 1
            captured_thinking_parts = []
            return events

        def _ensure_content_block_events(
            block_type: str,
            content_block: AnthropicContentBlock,
            force_new: bool = False,
        ) -> list[AnthropicStreamEvent]:
            """Open a content_block, closing the prior one if needed.

            ``force_new=True`` closes an existing block even when its type
            matches — required when a stream emits two consecutive
            ``tool_use`` blocks: each tool needs its own
            ``content_block_start``/``stop`` pair and its own
            ``content_block_index``, otherwise the second tool's
            ``input_json_delta`` chunks would append to the first tool's
            JSON arguments and corrupt both tool calls.
            """
            nonlocal content_block_open, content_block_type

            events: list[AnthropicStreamEvent] = []
            if content_block_open and (force_new or content_block_type != block_type):
                events.extend(_close_content_block_events())
            if not content_block_open:
                events.append(
                    ContentBlockStartEvent(
                        index=content_block_index,
                        content_block=content_block,
                    )
                )
                content_block_open = True
                content_block_type = block_type
            return events

        def _ensure_message_started(usage) -> list[str]:
            """Emit message_start exactly once. Returns SSE frames to yield."""
            nonlocal message_started
            if message_started:
                return []
            message_started = True
            return [_emit(_message_start_event(usage))]

        def _build_error_event(error_type: str, message: str) -> ErrorEvent:
            return ErrorEvent(
                error=AnthropicError(type=error_type, message=message),
            )

        def _flush_on_error(error_type: str, message: str) -> list[str]:
            """Build a self-contained terminal SSE sequence on error.

            Guarantees that whatever events we emit on the failure path
            leave the wire in a valid state: message_start (if not yet
            sent), close any open content block, then ErrorEvent and
            MessageStopEvent. Strict SDK clients reject streams whose
            content_block_start has no matching content_block_stop, so
            the close step is mandatory even on the error path.
            """
            frames: list[str] = []
            frames.extend(_ensure_message_started(None))
            for event in _close_content_block_events():
                frames.append(_emit(event))
            frames.append(_emit(_build_error_event(error_type, message)))
            frames.append(_emit(MessageStopEvent()))
            return frames

        async def _recover_tool_suffix(
            reason: _ToolRecoveryReason,
        ) -> AsyncGenerator[str, None]:
            """Replace the quarantined suffix with one validated retry result."""

            nonlocal recovery_attempted, recovery_succeeded
            nonlocal had_content_delta, final_usage
            recovery_succeeded = False
            if recovery_attempted or tool_recovery_callback is None:
                return
            recovery_attempted = True

            candidate_tool_names = tuple(
                dict.fromkeys(pending.name for pending in pending_tool_calls.values())
            )

            # V10 guarantees no buffered tool block or post-tool text has been
            # published. Abort the failed decode before submitting the fresh
            # RID so late tokenizer output cannot collide with recovery state.
            await _close_openai_stream()
            pending_tool_calls.clear()
            pending_segments.clear()

            callback_task = asyncio.create_task(
                tool_recovery_callback(reason, candidate_tool_names)
            )
            try:
                while not callback_task.done():
                    done, _ = await asyncio.wait(
                        {callback_task},
                        timeout=self._tool_recovery_ping_interval_s,
                    )
                    if not done:
                        # Anthropic permits ping events at any point in an SSE
                        # response after message_start. Before message_start,
                        # use a standards-compliant SSE comment heartbeat so we
                        # preserve event ordering and can still report accurate
                        # prompt usage when recovery returns.
                        if message_started:
                            yield _emit(PingEvent())
                        else:
                            yield ": anthropic-tool-recovery\n\n"
                recovered = callback_task.result()
            except asyncio.CancelledError:
                if not callback_task.done():
                    callback_task.cancel()
                    try:
                        await callback_task
                    except asyncio.CancelledError:
                        pass
                raise
            except Exception:
                logger.exception(
                    "Anthropic tool recovery callback failed: reason=%s",
                    reason.value,
                )
                return
            finally:
                # ``aclose()`` injects GeneratorExit while this generator may
                # be suspended at a ping yield. Cancellation cleanup therefore
                # belongs in ``finally``, not only the CancelledError branch,
                # or a client disconnect can orphan the retry decode.
                if not callback_task.done():
                    callback_task.cancel()
                    await asyncio.gather(callback_task, return_exceptions=True)
            if recovered is None:
                return

            for frame in _ensure_message_started(recovered.usage):
                yield frame
            for event in _close_content_block_events():
                yield _emit(event)

            for call in recovered.calls:
                for event in _ensure_content_block_events(
                    "tool_use",
                    ToolUseBlock(
                        id=call.id,
                        name=call.name,
                        input={},
                    ),
                    force_new=True,
                ):
                    yield _emit(event)
                yield _emit(
                    ContentBlockDeltaEvent(
                        index=content_block_index,
                        delta=InputJsonDelta(partial_json=call.argument_json),
                    )
                )
                for event in _close_content_block_events():
                    yield _emit(event)
                had_content_delta = True

            recovered_usage = _anthropic_usage_from_openai(
                recovered.usage,
                include_input=False,
                include_output=True,
            )
            original_output_tokens = (
                final_usage.output_tokens
                if final_usage is not None and final_usage.output_tokens is not None
                else 0
            )
            recovered_output_tokens = recovered_usage.output_tokens or 0
            final_usage = AnthropicUsage(
                output_tokens=original_output_tokens + recovered_output_tokens
            )
            yield _emit(
                MessageDeltaEvent(
                    delta=AnthropicMessageEndDelta(stop_reason="tool_use"),
                    usage=final_usage,
                )
            )
            yield _emit(MessageStopEvent())
            recovery_succeeded = True

        def _pending_tool_has_following_text(
            pending: _PendingAnthropicToolCall,
        ) -> bool:
            """Return whether text has logically closed this tool segment.

            Argument chunks for parallel tools may be interleaved, so a later
            tool segment alone does not close an earlier tool.  Once ordinary
            text follows a tool, however, Anthropic cannot reopen that tool's
            content block for more ``input_json_delta`` events.
            """

            seen_pending = False
            for segment in pending_segments:
                if segment is pending:
                    seen_pending = True
                elif seen_pending and isinstance(segment, _PendingAnthropicTextSegment):
                    return True
            return False

        def _append_pending_text(text: str) -> None:
            if pending_segments and isinstance(
                pending_segments[-1], _PendingAnthropicTextSegment
            ):
                pending_segments[-1].parts.append(text)
            else:
                pending_segments.append(_PendingAnthropicTextSegment(parts=[text]))

        def _pending_segment_kinds() -> list[str]:
            return [
                "text" if isinstance(segment, _PendingAnthropicTextSegment) else "tool"
                for segment in pending_segments
            ]

        def _commit_pending_content_events() -> list[AnthropicStreamEvent]:
            """Validate and atomically publish ordered tool/text segments.

            The OpenAI-side DeepSeek parser emits a tool name before it knows
            whether the complete invoke is valid.  Buffering through the final
            ``tool_calls`` finish reason prevents a later parser rejection from
            leaving Claude/Anthropic clients with an empty or partial tool block.
            Text observed after the first tool is part of the same transaction
            so inter-call text remains ordered without escaping before every
            tool has passed validation.
            """
            nonlocal had_content_delta

            for pending in pending_tool_calls.values():
                arguments = "".join(pending.argument_parts)
                parsed_arguments: dict[str, Any] = {}
                if arguments:
                    try:
                        parsed_arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError) as e:
                        raise MalformedToolCallError(
                            "Model emitted malformed tool-call arguments"
                        ) from e
                    if not isinstance(parsed_arguments, dict):
                        raise MalformedToolCallError(
                            "Model emitted non-object tool-call arguments"
                        )
                _validate_anthropic_tool_input(
                    anthropic_request,
                    pending.name,
                    parsed_arguments,
                )

            events: list[AnthropicStreamEvent] = []
            for segment in pending_segments:
                if isinstance(segment, _PendingAnthropicTextSegment):
                    events.extend(
                        _ensure_content_block_events(
                            "text",
                            TextBlock(text=""),
                        )
                    )
                    for part in segment.parts:
                        events.append(
                            ContentBlockDeltaEvent(
                                index=content_block_index,
                                delta=TextDelta(text=part),
                            )
                        )
                    had_content_delta = True
                    continue

                pending = segment
                events.extend(
                    _ensure_content_block_events(
                        "tool_use",
                        ToolUseBlock(
                            id=pending.id,
                            name=pending.name,
                            input={},
                        ),
                        force_new=True,
                    )
                )
                for part in pending.argument_parts:
                    if part:
                        events.append(
                            ContentBlockDeltaEvent(
                                index=content_block_index,
                                delta=InputJsonDelta(partial_json=part),
                            )
                        )
                events.extend(_close_content_block_events())
                had_content_delta = True

            pending_tool_calls.clear()
            pending_segments.clear()
            return events

        def _parse_upstream_error(data_str: str) -> Optional[tuple[str, str]]:
            """Detect an OpenAI handler streaming-error envelope.

            ``OpenAIServingChat.create_streaming_error_response`` emits
            ``data: {"error": {"object":"error","message":"...",
            "type":"BadRequestError","code":400}}``; the regular
            ChatCompletionStreamResponse validator rejects it. Pull the
            type/message out so the Anthropic client sees the real
            failure instead of a generic 'Stream processing error'.
            """
            try:
                payload = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(payload, dict):
                return None
            err = payload.get("error")
            if not isinstance(err, dict):
                return None
            upstream_message = err.get("message") or "Upstream error"
            code = err.get("code")
            error_type = (
                ERROR_TYPE_MAP.get(code, "api_error")
                if isinstance(code, int)
                else "api_error"
            )
            return error_type, str(upstream_message)

        # Pre-first-chunk errors from the OpenAI generator (e.g. tokenization
        # failure that raises ValueError before any chunk is yielded) would
        # otherwise abort the StreamingResponse with no envelope at all and
        # the client would see a half-open SSE / TCP close. Catch them here
        # and emit a clean Anthropic error sequence instead.
        try:
            stream_iter = openai_stream.__aiter__()
        except Exception as e:
            logger.exception("Failed to open OpenAI stream: %s", e)
            await _close_openai_stream()
            for frame in _flush_on_error("api_error", "Internal server error"):
                yield frame
            return

        while True:
            try:
                sse_line = await stream_iter.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                await _close_openai_stream()
                raise
            except MalformedToolCallError as e:
                logger.warning("Malformed DeepSeek tool-call output: %s", e)
                recovery_reason = (
                    _classify_tool_recovery_reason(str(e))
                    or _ToolRecoveryReason.MALFORMED_PROTOCOL
                )
                async for frame in _recover_tool_suffix(recovery_reason):
                    yield frame
                if recovery_succeeded:
                    return
                await _close_openai_stream()
                for frame in _flush_on_error("api_error", str(e)):
                    yield frame
                return
            except ValueError as e:
                # _generate_chat_stream re-raises ValueError when its own
                # ``stream_started`` flag is still False — surface as a
                # proper Anthropic error event rather than aborting the
                # StreamingResponse generator.
                logger.warning("OpenAI stream raised before first chunk: %s", e)
                await _close_openai_stream()
                for frame in _flush_on_error(
                    "invalid_request_error", str(e) or "Request failed"
                ):
                    yield frame
                return
            except Exception as e:
                logger.exception("OpenAI stream raised mid-flight: %s", e)
                await _close_openai_stream()
                for frame in _flush_on_error("api_error", "Internal server error"):
                    yield frame
                return

            if not sse_line.startswith("data: "):
                continue

            data_str = sse_line[6:].strip()

            if data_str == "[DONE]":
                await _close_openai_stream()
                for frame in _ensure_message_started(None):
                    yield frame

                if pending_tool_calls:
                    logger.warning(
                        "OpenAI stream ended before buffered tool calls received "
                        "a terminal finish reason"
                    )
                    async for frame in _recover_tool_suffix(
                        _ToolRecoveryReason.INCOMPLETE_STREAM
                    ):
                        yield frame
                    if recovery_succeeded:
                        return
                    for frame in _flush_on_error(
                        "api_error", "Incomplete tool-call stream"
                    ):
                        yield frame
                    return

                # No content AND no finish_reason: the backend dropped the
                # stream silently. Surface as api_error so clients see the
                # failure instead of a fake empty success. If finish_reason
                # IS set we trust the backend's signal — a legitimate empty
                # completion (max_tokens=1 stop, content filter, etc.)
                # deserves a normal message_delta/message_stop pair, not
                # an error that triggers SDK retry loops.
                if not had_content_delta and finish_reason is None:
                    logger.warning(
                        "Stream produced no content and no finish_reason "
                        "before [DONE]; emitting api_error event"
                    )
                    yield _emit(
                        _build_error_event("api_error", "Backend produced no content")
                    )
                    yield _emit(MessageStopEvent())
                    return

                # Close any open content block
                for event in _close_content_block_events():
                    yield _emit(event)

                # Emit message_delta with stop_reason and usage
                effective_finish = finish_reason or "stop"
                if effective_finish not in STOP_REASON_MAP:
                    logger.warning(
                        "Unmapped streaming finish_reason %r; defaulting to end_turn",
                        effective_finish,
                    )
                stop_reason = STOP_REASON_MAP.get(effective_finish, "end_turn")
                yield _emit(
                    MessageDeltaEvent(
                        delta=AnthropicMessageEndDelta(stop_reason=stop_reason),
                        usage=final_usage or AnthropicUsage(output_tokens=0),
                    )
                )

                yield _emit(MessageStopEvent())
                return

            # Parse the OpenAI chunk
            try:
                chunk = ChatCompletionStreamResponse.model_validate_json(data_str)
            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
                # First check whether this is the OpenAI handler's
                # streaming error envelope (validator rejects it because
                # it lacks id/choices/created/model). Forwarding the real
                # type/message keeps the failure debuggable instead of
                # collapsing every backend error into "Stream processing
                # error".
                upstream = _parse_upstream_error(data_str)
                if upstream is not None:
                    error_type, error_message = upstream
                    logger.warning(
                        "Forwarding upstream stream error (%s): %s",
                        error_type,
                        error_message,
                    )
                    recovery_reason = _classify_tool_recovery_reason(error_message)
                    if recovery_reason is not None:
                        async for frame in _recover_tool_suffix(recovery_reason):
                            yield frame
                        if recovery_succeeded:
                            return
                    await _close_openai_stream()
                    for frame in _flush_on_error(error_type, error_message):
                        yield frame
                    return

                logger.warning(
                    "Failed to parse Anthropic stream chunk (%s): %s",
                    type(e).__name__,
                    data_str[:200],
                )
                await _close_openai_stream()
                for frame in _flush_on_error("api_error", "Stream processing error"):
                    yield frame
                return

            if chunk.usage is not None:
                final_usage = _anthropic_usage_from_openai(
                    chunk.usage,
                    include_input=False,
                    include_output=True,
                )

            # Usage-only chunk (empty choices with usage info)
            if not chunk.choices and chunk.usage:
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            # Capture finish_reason on this chunk but DO NOT short-circuit:
            # some OpenAI-compatible backends pack the final content token
            # (or last tool-args fragment) into the same chunk as
            # finish_reason. Skipping delta processing would silently drop
            # that payload — sometimes the whole completion if it was a
            # one-token reply. Fall through to the delta handlers below.
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

            delta = choice.delta

            # Defer message_start until the first chunk carrying real prompt
            # usage or content. OpenAI streams emit a role-only chunk before
            # usage is available; emitting message_start there would ship
            # input_tokens=0 to the client.
            has_delta_payload = bool(
                delta.reasoning_content
                or delta.tool_calls
                or (delta.content is not None and delta.content != "")
                or chunk.usage
            )
            # The finish_reason chunk should also flip message_started so a
            # zero-content completion (the path that previously fired the
            # 'Backend produced no content' error) emits the standard
            # message_start before [DONE] closes the stream.
            if (
                has_delta_payload or choice.finish_reason is not None
            ) and not message_started:
                yield _emit(_message_start_event(chunk.usage))
                message_started = True

            if (
                not has_delta_payload
                and delta.role == "assistant"
                and (delta.content is None or delta.content == "")
            ):
                continue

            # Reasoning is valid only before tool-call protocol begins. If a
            # corrupted stream reopens reasoning after a tool fragment, keep
            # that entire suffix quarantined and replace it through the same
            # bounded recovery path instead of publishing reordered content.
            if delta.reasoning_content and pending_tool_calls:
                logger.warning(
                    "Reasoning resumed after tool-call stream began; "
                    "failing Anthropic stream closed: rid=%s "
                    "pending_segments=%s",
                    chunk.id,
                    _pending_segment_kinds(),
                )
                async for frame in _recover_tool_suffix(
                    _ToolRecoveryReason.STREAM_ORDERING
                ):
                    yield frame
                if recovery_succeeded:
                    return
                await _close_openai_stream()
                for frame in _flush_on_error(
                    "api_error", "Invalid tool-call stream ordering"
                ):
                    yield frame
                return

            # Handle reasoning content deltas
            if delta.reasoning_content:
                for event in _ensure_content_block_events(
                    "thinking",
                    ThinkingBlock(thinking=""),
                ):
                    yield _emit(event)

                if omit_thinking:
                    captured_thinking_parts.append(delta.reasoning_content)
                else:
                    yield _emit(
                        ContentBlockDeltaEvent(
                            index=content_block_index,
                            delta=ThinkingDelta(thinking=delta.reasoning_content),
                        )
                    )
                had_content_delta = True

            # Handle tool call deltas
            if delta.tool_calls:
                tool_stream_error: Optional[str] = None
                tool_stream_index: Optional[int] = None
                tool_stream_has_name = False
                for tc in delta.tool_calls:
                    tc_id = tc.id
                    tc_func = tc.function
                    raw_arguments = tc_func.arguments if tc_func else None
                    if isinstance(raw_arguments, dict):
                        argument_part: Optional[str] = json.dumps(
                            raw_arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    elif raw_arguments is None:
                        argument_part = None
                    else:
                        argument_part = str(raw_arguments)

                    if tc_func and tc_func.name:
                        if tc.index is None:
                            tool_index = next_tool_call_index
                            next_tool_call_index += 1
                        else:
                            tool_index = tc.index
                            next_tool_call_index = max(
                                next_tool_call_index, tool_index + 1
                            )
                        pending = pending_tool_calls.get(tool_index)
                        if pending is None:
                            pending = _PendingAnthropicToolCall(
                                tool_index=tool_index,
                                id=tc_id or f"toolu_{uuid.uuid4().hex}",
                                name=tc_func.name,
                                argument_parts=[],
                            )
                            pending_tool_calls[tool_index] = pending
                            pending_segments.append(pending)
                        elif pending.name != tc_func.name:
                            tool_stream_error = "Conflicting tool-call stream index"
                            tool_stream_index = tool_index
                            tool_stream_has_name = True
                            break
                        elif _pending_tool_has_following_text(pending):
                            tool_stream_error = "Invalid tool-call stream ordering"
                            tool_stream_index = tool_index
                            tool_stream_has_name = True
                            break
                        elif tc_id:
                            pending.id = tc_id
                        active_tool_call_index = tool_index
                        if argument_part:
                            pending.argument_parts.append(argument_part)

                    elif argument_part:
                        tool_index = (
                            tc.index if tc.index is not None else active_tool_call_index
                        )
                        pending = (
                            pending_tool_calls.get(tool_index)
                            if tool_index is not None
                            else None
                        )
                        if pending is None or _pending_tool_has_following_text(pending):
                            tool_stream_error = "Invalid tool-call stream ordering"
                            tool_stream_index = tool_index
                            tool_stream_has_name = False
                            break
                        pending.argument_parts.append(argument_part)

                if tool_stream_error is not None:
                    logger.warning(
                        "%s; failing Anthropic stream closed: rid=%s "
                        "incoming_tool_index=%r incoming_has_name=%s "
                        "active_tool_index=%r pending_segments=%s",
                        tool_stream_error,
                        chunk.id,
                        tool_stream_index,
                        tool_stream_has_name,
                        active_tool_call_index,
                        _pending_segment_kinds(),
                    )
                    async for frame in _recover_tool_suffix(
                        _ToolRecoveryReason.STREAM_ORDERING
                    ):
                        yield frame
                    if recovery_succeeded:
                        return
                    await _close_openai_stream()
                    for frame in _flush_on_error("api_error", tool_stream_error):
                        yield frame
                    return

            # Handle text content deltas
            if delta.content is not None and delta.content != "":
                if pending_tool_calls:
                    _append_pending_text(delta.content)
                else:
                    for event in _ensure_content_block_events(
                        "text",
                        TextBlock(text=""),
                    ):
                        yield _emit(event)

                    yield _emit(
                        ContentBlockDeltaEvent(
                            index=content_block_index,
                            delta=TextDelta(text=delta.content),
                        )
                    )
                    had_content_delta = True

            if choice.finish_reason is not None and pending_tool_calls:
                if choice.finish_reason not in {"stop", "tool_calls"}:
                    logger.warning(
                        "Tool-call stream terminated with finish_reason=%r",
                        choice.finish_reason,
                    )
                    async for frame in _recover_tool_suffix(
                        _ToolRecoveryReason.INCOMPLETE_STREAM
                    ):
                        yield frame
                    if recovery_succeeded:
                        return
                    await _close_openai_stream()
                    for frame in _flush_on_error(
                        "api_error", "Incomplete tool-call stream"
                    ):
                        yield frame
                    return
                try:
                    pending_events = _commit_pending_content_events()
                except MalformedToolCallError as e:
                    logger.warning("Rejecting buffered Anthropic tool call: %s", e)
                    recovery_reason = (
                        _classify_tool_recovery_reason(str(e))
                        or _ToolRecoveryReason.MALFORMED_ARGUMENTS
                    )
                    async for frame in _recover_tool_suffix(recovery_reason):
                        yield frame
                    if recovery_succeeded:
                        return
                    await _close_openai_stream()
                    for frame in _flush_on_error("api_error", str(e)):
                        yield frame
                    return
                for event in pending_events:
                    yield _emit(event)
                # Match OpenAI's normal conversion: a natural stop after a
                # parsed tool call is surfaced as Anthropic ``tool_use``.
                if finish_reason == "stop":
                    finish_reason = "tool_calls"

        if pending_tool_calls:
            async for frame in _recover_tool_suffix(
                _ToolRecoveryReason.INCOMPLETE_STREAM
            ):
                yield frame
            if recovery_succeeded:
                return
        await _close_openai_stream()
        logger.warning("OpenAI stream ended without a terminal [DONE] event")
        for frame in _flush_on_error("api_error", "Incomplete upstream stream"):
            yield frame

    def _convert_response(
        self,
        response: ChatCompletionResponse,
        anthropic_request: Optional[AnthropicMessagesRequest] = None,
    ) -> AnthropicMessagesResponse:
        """Convert an OpenAI ChatCompletionResponse to an Anthropic Messages response."""
        if not response.choices:
            return AnthropicMessagesResponse(
                content=[TextBlock(text="")],
                model=response.model,
                stop_reason="end_turn",
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            )

        choice = response.choices[0]
        content: list[AnthropicContentBlock] = []

        omit_thinking = bool(
            anthropic_request is not None
            and anthropic_request.thinking is not None
            and anthropic_request.thinking.display == "omitted"
        )

        # Visible mode returns the backend's reasoning directly. Omitted mode
        # returns an empty block plus an authenticated token that lets a later
        # request reconstruct the hidden reasoning without server-side state.
        if choice.message.reasoning_content:
            if omit_thinking:
                if self._thinking_signature_codec is None:
                    raise RuntimeError(
                        "SGLang thinking-signature key is not configured"
                    )
                content.append(
                    ThinkingBlock(
                        thinking="",
                        signature=self._thinking_signature_codec.seal(
                            choice.message.reasoning_content,
                            anthropic_request.model,
                        ),
                    )
                )
            else:
                content.append(ThinkingBlock(thinking=choice.message.reasoning_content))

        # Add text content
        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))

        # Add tool calls
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                raw_args = tool_call.function.arguments
                try:
                    tool_input = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError) as e:
                    raise MalformedToolCallError(
                        "Model emitted malformed tool-call arguments"
                    ) from e
                if not isinstance(tool_input, dict):
                    raise MalformedToolCallError(
                        "Model emitted non-object tool-call arguments"
                    )
                if anthropic_request is not None:
                    _validate_anthropic_tool_input(
                        anthropic_request,
                        tool_call.function.name,
                        tool_input,
                    )

                content.append(
                    ToolUseBlock(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_input,
                    )
                )

        # Map stop reason
        finish_reason = choice.finish_reason or "stop"
        if choice.message.tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        if finish_reason not in STOP_REASON_MAP:
            logger.warning(
                "Unmapped OpenAI finish_reason %r; defaulting to end_turn",
                finish_reason,
            )
        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

        # Anthropic requires ``content`` to contain at least one block.
        # Empty string completions (max_tokens=1 stop, content filter, etc.)
        # would otherwise ship ``content=[]`` and break strict SDK parsers.
        if not content:
            content.append(TextBlock(text=""))

        return AnthropicMessagesResponse(
            id=f"msg_{uuid.uuid4().hex}",
            content=content,
            model=response.model,
            stop_reason=stop_reason,
            usage=_anthropic_usage_from_openai(
                response.usage,
                include_input=True,
                include_output=True,
            ),
        )

    def _convert_openai_error_response(self, response) -> JSONResponse:
        """Forward an upstream OpenAI-handler error as an Anthropic error.

        The original error message is preserved for 4xx (after light
        sanitization) so callers see the real validation failure. For 5xx
        we always return a generic ``"Internal server error"`` to avoid
        leaking ``str(e)`` payloads that the OpenAI handler builds from
        raw exceptions (paths, tracebacks, prompt fragments, etc.).
        """
        status_code = getattr(response, "status_code", 500)
        body = getattr(response, "body", b"") or b""
        error_type = ERROR_TYPE_MAP.get(status_code, "api_error")

        upstream_message: Optional[str] = None
        try:
            payload = json.loads(body.decode("utf-8")) if body else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Non-JSON body (HTML gateway error, plain text, ...). Use a
            # bounded slice of the raw body so the client still has a
            # useful hint instead of a generic placeholder.
            try:
                upstream_message = body.decode("utf-8", errors="replace")[:500]
            except Exception:
                upstream_message = None
        else:
            if isinstance(payload, dict):
                error_payload = payload.get("error", payload)
                if isinstance(error_payload, dict):
                    upstream_message = error_payload.get("message") or payload.get(
                        "message"
                    )
                    # Honor the upstream error.type only for 4xx; 5xx is
                    # normalized below.
                    if status_code < 500:
                        upstream_type = error_payload.get("type")
                        if isinstance(upstream_type, str) and upstream_type:
                            error_type = upstream_type
                elif isinstance(error_payload, str):
                    upstream_message = error_payload
                elif isinstance(payload.get("message"), str):
                    upstream_message = payload["message"]

        message = _scrub_error_message(upstream_message or "", status_code)
        return self._error_response(
            status_code=status_code,
            error_type=error_type,
            message=message,
        )

    def _error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
        exception_name: Optional[str] = None,
    ) -> JSONResponse:
        """Create an Anthropic-format error response.

        ``error.type`` is restricted to Anthropic's documented enum so strict
        SDK clients (anthropic-sdk-python / -typescript) keep parsing the
        response into their typed error classes. ``exception_name`` — when
        provided — is logged at WARNING level so operators can still grep
        server-side, but it never reaches the wire.
        """
        if exception_name:
            logger.warning(
                "Anthropic error response %s (exception=%s): %s",
                error_type,
                exception_name,
                message,
            )
        error_resp = AnthropicErrorResponse(
            error=AnthropicError(type=error_type, message=message)
        )
        return JSONResponse(
            status_code=status_code,
            content=error_resp.model_dump(),
        )

    async def handle_count_tokens(
        self,
        request: AnthropicCountTokensRequest,
        raw_request: Request,
    ) -> JSONResponse:
        """Handle /v1/messages/count_tokens endpoint.

        Converts the request to a ChatCompletionRequest, applies the chat
        template via the OpenAI handler to tokenize, and returns the count.
        """
        try:
            # Build a minimal AnthropicMessagesRequest so we can reuse conversion
            messages_request = AnthropicMessagesRequest(
                model=request.model,
                messages=request.messages,
                max_tokens=1,  # dummy, not used for counting
                system=request.system,
                thinking=request.thinking,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            if (
                self._request_contains_local_thinking_signature(messages_request)
                and self._thinking_signature_codec is None
            ):
                logger.error(
                    "Anthropic thinking-signature history requested but %s "
                    "is not configured",
                    _THINKING_KEY_FILE_ENV,
                )
                return self._error_response(
                    status_code=500,
                    error_type="api_error",
                    message="Internal server error",
                )
            chat_request = self._convert_to_chat_completion_request(messages_request)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error converting count_tokens request: %s", e)
            return self._error_response(
                status_code=400,
                error_type="invalid_request_error",
                message=str(e),
            )

        try:
            is_multimodal = (
                self.openai_serving_chat.tokenizer_manager.model_config.is_multimodal
            )
            processed = self.openai_serving_chat._process_messages(
                chat_request, is_multimodal
            )

            if isinstance(processed.prompt_ids, list):
                input_tokens = len(processed.prompt_ids)
            else:
                # prompt_ids is a string (multimodal case) — tokenize it
                tokenizer = self.openai_serving_chat.tokenizer_manager.tokenizer
                input_tokens = len(tokenizer.encode(processed.prompt_ids))

            return JSONResponse(
                content=AnthropicCountTokensResponse(
                    input_tokens=input_tokens
                ).model_dump()
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Error counting tokens: %s", e)
            return self._error_response(
                status_code=500,
                error_type="api_error",
                message="Internal server error",
                exception_name=type(e).__name__,
            )
