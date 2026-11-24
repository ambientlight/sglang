"""One bounded local retry of an unpublished malformed Responses function call."""

import asyncio
import logging
import os
from uuid import uuid4

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.parser.dsv41_tool_transaction import validate_call

logger = logging.getLogger(__name__)


def _limit(name, default, maximum):
    try:
        value = int(os.environ.get("SGLANG_RESPONSES_TOOL_RECOVERY_" + name, default))
        if 1 <= value <= maximum:
            return value
    except ValueError:
        pass
    logger.warning("Invalid Responses recovery %s, using %d", name, default)
    return default


class ResponsesToolRecovery:
    def __init__(self):
        self.timeout = _limit("TIMEOUT_S", 180, 3600)
        self.max_tokens = _limit("MAX_TOKENS", 8192, 131072)
        self.ping_interval = _limit("PING_INTERVAL_S", 10, 60)
        self.slots = asyncio.Semaphore(_limit("CONCURRENCY", 2, 64))

    async def retry(self, handler, original, raw_request, candidate_names, rid):
        if (
            original is None
            or not original.tools
            or original.tool_choice == "none"
            or handler.is_disaggregated
        ):
            return None
        recovery_id = uuid4().hex + "_toolrecovery"
        while rid and recovery_id.startswith(rid):
            recovery_id = uuid4().hex + "_toolrecovery"
        submitted = False
        try:
            async with asyncio.timeout(self.timeout), self.slots:
                request = original.model_copy(deep=True)
                request.rid = recovery_id
                request.stream, request.stream_options = False, None
                request.session_id = request.session_params = None
                limit = request.effective_max_tokens
                request.max_tokens = (
                    self.max_tokens if limit is None else min(limit, self.max_tokens)
                )
                request.max_completion_tokens = None
                declared = {
                    tool.function.name: tool.function for tool in original.tools
                }
                selected = set(candidate_names).intersection(declared)
                if isinstance(request.tool_choice, ToolChoice):
                    selected = {request.tool_choice.function.name}
                if selected:
                    request.tools = [
                        tool for tool in request.tools if tool.function.name in selected
                    ]
                    request.tool_choice = (
                        ToolChoice(
                            type="function",
                            function=ToolChoiceFuncName(name=next(iter(selected))),
                        )
                        if len(selected) == 1
                        else "required"
                    )
                for tool in request.tools:
                    tool.function.strict = True
                adapted, normalized = handler._convert_to_internal_request(
                    request, raw_request
                )
                adapted.rid = recovery_id
                adapted.received_time = monotonic_time()
                submitted = True
                response = await handler._handle_non_streaming_request(
                    adapted, normalized, raw_request
                )
                if (
                    not isinstance(response, ChatCompletionResponse)
                    or not response.choices
                ):
                    return None
                choice = response.choices[0]
                if (
                    choice.finish_reason not in ("stop", "tool_calls")
                    or not choice.message.tool_calls
                ):
                    return None
                calls = []
                for i, call in enumerate(choice.message.tool_calls):
                    name, args = call.function.name, call.function.arguments
                    if selected and name not in selected:
                        return None
                    validate_call(declared, name, args)
                    calls.append(ToolCallItem(tool_index=i, name=name, parameters=args))
                return calls, response.usage
        except (TimeoutError, asyncio.CancelledError) as exc:
            if submitted:
                handler.tokenizer_manager.abort_request(recovery_id)
            if isinstance(exc, asyncio.CancelledError):
                raise
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Responses tool recovery failed (%s)", type(exc).__name__)
        return None
