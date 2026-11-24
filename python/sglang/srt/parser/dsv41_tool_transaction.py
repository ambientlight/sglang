"""Validate an entire local DeepSeek tool suffix before exposing any call."""

import json

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from sglang.srt.function_call.core_types import MalformedToolCallError
from sglang.srt.parser.deepseek_recipe_output import RecipeOutputDelta, whole_tool_calls


class ToolTransaction:
    def __init__(self, parser, tools):
        self.parser = parser
        self.tools = {tool.name: tool for tool in tools}
        self.staged = []
        self.error = None
        self.closed = False

    @property
    def num_consumed_tokens(self):
        return self.parser.num_consumed_tokens

    def feed(self, token_ids):
        if self.error is not None:
            # Consume stream output through the original finish before retrying.
            self.parser.num_consumed_tokens += len(token_ids)
            return RecipeOutputDelta()
        try:
            piece = self.parser.feed(token_ids)
        except ValueError:
            self.error = MalformedToolCallError(
                "Model emitted malformed tool-call protocol"
            )
            return RecipeOutputDelta()
        if piece.tool_calls or self.staged:
            self.staged.append(piece)
            return RecipeOutputDelta()
        return piece

    def finish(self, token_ids, finish_reason, **kwargs):
        if self.closed:
            return RecipeOutputDelta()
        self.closed = True
        if self.error is not None:
            raise self.error
        try:
            final = self.parser.finish(token_ids, finish_reason, **kwargs)
        except ValueError as exc:
            raise MalformedToolCallError(
                "Model emitted malformed tool-call protocol"
            ) from exc
        if not self.staged and not final.tool_calls:
            return final
        pieces, self.staged = self.staged + [final], []
        merged = RecipeOutputDelta(
            reasoning="".join(piece.reasoning for piece in pieces),
            content="".join(piece.content for piece in pieces),
            tool_calls=tuple(call for piece in pieces for call in piece.tool_calls),
        )
        if finish_reason.get("type") not in ("stop", "tool_calls"):
            raise MalformedToolCallError("Incomplete tool-call stream")
        calls = whole_tool_calls(merged)
        try:
            for call in calls:
                validate_call(self.tools, call.name, call.arguments)
        except MalformedToolCallError as exc:
            exc.candidate_names = tuple(
                call.name for call in calls if call.name in self.tools
            )
            raise
        return merged


def validate_call(tools, name, arguments):
    tool = tools.get(name)
    if tool is None:
        raise MalformedToolCallError("Model attempted to call an undeclared tool")
    try:
        arguments = json.loads(arguments)
    except (TypeError, ValueError) as exc:
        raise MalformedToolCallError(
            "Model emitted malformed tool-call arguments"
        ) from exc
    if not isinstance(arguments, dict):
        raise MalformedToolCallError("Model emitted non-object tool-call arguments")
    if tool.strict:
        try:
            Draft202012Validator(tool.parameters or {}).validate(arguments)
        except ValidationError as exc:
            raise MalformedToolCallError(
                "Model emitted tool-call arguments that do not match the declared schema"
            ) from exc
