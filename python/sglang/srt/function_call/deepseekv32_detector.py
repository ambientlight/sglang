import json
import logging
import re

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    MalformedToolCallError,
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)

_GENERIC_INVOKE_RE = re.compile(
    r'^[ \t]*<invoke\s+name="(?P<name>[^"]+)"\s*>', re.MULTILINE
)
_DSML_TAG_PREFIX_RE = re.compile(r"</?｜DSML｜")
_DSML_TAG_NAME_RE = re.compile(r" ?[A-Za-z_][A-Za-z0-9_-]*")
_DSML_WRAPPER_NAMES = frozenset({"tool_calls", "function_calls", " calls"})
_DSML_PROTOCOL_PADDING = " \t\r\n\f\v\u200b\ufeff"
_DSML_PROTOCOL_IN_VALUE_RE = re.compile(
    r"</｜DSML｜ ?[A-Za-z_][A-Za-z0-9_-]*\s*>"
    r"|</?｜DSML｜ ?(?:tool_calls|function_calls|calls|invoke|parameter)\b"
    r"|</parameter>\s*<\]"
    r"|<[^<>\r\n]{0,8}DSML｜ ?(?:tool_calls|function_calls|calls|invoke|parameter)\b"
)


def _strip_dsml_protocol_padding(text: str) -> str:
    """Strip whitespace plus invisible padding only at DSML boundaries.

    U+200B and U+FEFF are semantically empty between tags but Python's
    ``str.strip`` does not remove them. Inside a parameter value they remain
    ordinary argument data.
    """

    return text.strip(_DSML_PROTOCOL_PADDING)


def find_malformed_generic_tool_call(
    text: str, tools: list[Tool]
) -> tuple[str, int] | None:
    """Find a line-anchored generic invocation of a declared tool.

    DeepSeek V3.2/V4 must emit ``<｜DSML｜invoke ...>``. A generic
    ``<invoke ...>`` is therefore model-generated protocol drift, but only
    classify it when the name belongs to this request so ordinary inline XML
    examples remain content.
    """
    tool_names = {tool.function.name for tool in tools if tool.function.name}
    for match in _GENERIC_INVOKE_RE.finditer(text):
        name = match.group("name").strip()
        if name in tool_names:
            return name, match.start()
    return None


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    # Tag names after the DSML marker; subclasses override for newer formats.
    dsml_token = "｜DSML｜"
    tool_calls_block_name = "function_calls"
    invoke_tag_name = "invoke"
    parameter_tag_name = "parameter"
    preserve_string_whitespace = False

    _GENERIC_INVOKE_PREFIX = "<invoke"

    def __init__(self):
        super().__init__()
        block = f"{self.dsml_token}{self.tool_calls_block_name}"
        invoke = f"{self.dsml_token}{self.invoke_tag_name}"
        parameter = f"{self.dsml_token}{self.parameter_tag_name}"
        self.bot_token = f"<{block}>"
        self.eot_token = f"</{block}>"
        self.invoke_start_token = f"<{invoke}"
        self.invoke_end_token = f"</{invoke}>"
        # Derive every containment boundary from the selected protocol dialect.
        # V4.1's single leading space is significant, including across chunks.
        self._DSML_TOOL_TAGS = [
            f"{prefix}{tag}" for prefix in ("<", "</") for tag in (invoke, block)
        ]
        self._KNOWN_DSML_TAG_NAMES = frozenset(
            {self.tool_calls_block_name, self.invoke_tag_name, self.parameter_tag_name}
        )
        self._invoke_open_re = re.compile(
            rf'<{re.escape(invoke)}\s+name="[^"<>\r\n]+"\s*(?:>|/>)'
        )
        self.parameter_regex = (
            rf'<{parameter}\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</{parameter}>'
        )
        self.partial_parameter_regex = (
            rf'<{parameter}\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        self.function_calls_regex = rf"<{block}>(.*?)</{block}>"
        # Long-form `<｜DSML｜invoke name="x">...</｜DSML｜invoke>` and the
        # self-closing `<｜DSML｜invoke name="x"/>` shape V4 emits for zero-arg
        # tools. The `end` group is empty when the closer hasn't streamed in.
        self.invoke_regex = (
            rf'<{invoke}\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)"
            rf"|>(?P<body>.*?)(?P<end>(?:</{invoke}>|$)))"
        )
        self.parameter_end_token = f"</{parameter}>"
        self.current_tool_id = -1
        # The terminal validator needs the complete protocol envelope even
        # after ordinary text and completed invokes have been consumed from the
        # incremental buffer. This remains request-local and is released by
        # ``finish``.
        self._stream_text = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return (
            self.bot_token in text
            or self.invoke_start_token in text
            or _GENERIC_INVOKE_RE.search(text) is not None
        )

    @classmethod
    def _ends_with_partial_generic_invoke(cls, text: str) -> bool:
        """Hold a line-start generic invoke until it can be classified."""
        tail = text.rsplit("\n", 1)[-1].lstrip(" \t")
        if not tail:
            return False
        if cls._GENERIC_INVOKE_PREFIX.startswith(tail):
            return True
        return tail.startswith(cls._GENERIC_INVOKE_PREFIX) and ">" not in tail

    @staticmethod
    def _reject_malformed_generic_tool_call(text: str, tools: list[Tool]) -> None:
        malformed = find_malformed_generic_tool_call(text, tools)
        if malformed is None:
            return
        name, offset = malformed
        logger.warning(
            "DeepSeek output used malformed generic invocation for declared "
            "tool %r at character %d",
            name,
            offset,
        )
        raise MalformedToolCallError(
            "Model emitted malformed DeepSeek tool-call protocol"
        )

    def _is_inside_dsml_invoke(self, text: str, offset: int) -> bool:
        """Return whether *offset* is inside a valid unclosed invoke payload.

        Tool arguments are opaque user/model data and may legitimately contain
        strings that look like DSML examples.  The fail-closed check below is
        therefore limited to the normal-text/protocol layer around invokes.  A
        malformed ``<...invoke`` prefix must not establish that protection or it
        could hide every later degenerate marker in the turn.
        """
        last_open = -1
        for match in self._invoke_open_re.finditer(text, 0, offset):
            if not match.group(0).endswith("/>"):
                last_open = match.start()
        last_close = text.rfind(self.invoke_end_token.removesuffix(">"), 0, offset)
        return last_open > last_close

    def _is_valid_complete_dsml_tag(
        self, tag: str, tag_name: str, expected_wrapper: str
    ) -> bool:
        """Validate protocol control tags once their closing ``>`` arrives."""
        if tag_name in _DSML_WRAPPER_NAMES:
            return bool(
                re.fullmatch(rf"</?｜DSML｜{re.escape(expected_wrapper)}\s*>", tag)
            )
        if tag_name == self.invoke_tag_name:
            if tag.startswith("</"):
                return bool(
                    re.fullmatch(
                        rf"</{re.escape(self.dsml_token + self.invoke_tag_name)}\s*>",
                        tag,
                    )
                )
            return self._invoke_open_re.fullmatch(tag) is not None
        # Parameter tags can be ordinary explanatory text outside an invoke.
        # Inside a valid invoke they are opaque argument syntax and are skipped
        # before this helper is called.
        return tag_name == self.parameter_tag_name

    def _reject_malformed_reserved_dsml_tag(self, text: str) -> None:
        """Reject tag-shaped use of the reserved DSML namespace.

        The model sometimes degenerates into non-protocol markers such as
        ``</｜DSML｜tool_api_call...`` or ``<｜DSML｜tool_crossing``.  Those
        markers are neither ordinary response text nor valid tool calls.  Letting
        them through makes clients ingest raw protocol and often amplifies the
        broken turn in subsequent prompts.

        Known DSML parameter tags remain valid as explanatory text, and markers
        inside an invoke payload are left untouched as tool argument data.
        """
        for marker in _DSML_TAG_PREFIX_RE.finditer(text):
            if self._is_inside_dsml_invoke(text, marker.start()):
                continue

            tail = text[marker.end() :]
            if not tail or (tail == " " and self.invoke_tag_name.startswith(" ")):
                # A complete tag name may arrive in the next stream chunk.
                continue

            name_match = _DSML_TAG_NAME_RE.match(tail)
            if name_match is None:
                malformed = True
                tag_name = tail[:32]
            else:
                tag_name = name_match.group(0)
                next_char = tail[name_match.end() : name_match.end() + 1]
                expected_wrapper = self.tool_calls_block_name
                if tag_name in _DSML_WRAPPER_NAMES and (tag_name != expected_wrapper):
                    # V4 and V3.2 use different wrapper names. The alternate
                    # wrapper is reserved protocol, but is not valid for this
                    # detector and must not become response text.
                    malformed = True
                elif tag_name in self._KNOWN_DSML_TAG_NAMES:
                    # A known name must end at an XML-like boundary.  This also
                    # catches a valid-looking prefix followed by degenerate text.
                    malformed = bool(
                        next_char and not (next_char.isspace() or next_char in ">/")
                    )
                elif not next_char and any(
                    known.startswith(tag_name) for known in self._KNOWN_DSML_TAG_NAMES
                ):
                    # Hold an incomplete known tag such as ``tool_`` until the
                    # next chunk decides whether it becomes ``tool_calls``.
                    malformed = False
                else:
                    malformed = True

                if not malformed and tag_name != self.parameter_tag_name:
                    tag_end = text.find(">", marker.end())
                    nested_tag = text.find("<", marker.end())
                    if nested_tag != -1 and (tag_end == -1 or nested_tag < tag_end):
                        # A second tag began before this control tag closed.
                        # This is the exact v7 ``<tool_calls <invoke...`` and
                        # ``<invoke name=\"... <parameter...`` degeneration.
                        malformed = True
                    elif tag_end != -1:
                        malformed = not self._is_valid_complete_dsml_tag(
                            text[marker.start() : tag_end + 1],
                            tag_name,
                            expected_wrapper,
                        )
                    elif tag_name in _DSML_WRAPPER_NAMES:
                        # An incomplete wrapper may contain only padding while
                        # waiting for ``>``. Any other suffix is already invalid.
                        malformed = bool(tail[name_match.end() :].strip())

            if not malformed:
                continue

            logger.warning(
                "DeepSeek output used malformed reserved DSML tag %r at character %d",
                tag_name,
                marker.start(),
            )
            raise MalformedToolCallError(
                "Model emitted malformed DeepSeek tool-call protocol"
            )

    @staticmethod
    def _raise_malformed_tool_protocol() -> None:
        raise MalformedToolCallError(
            "Model emitted malformed DeepSeek tool-call protocol"
        )

    @staticmethod
    def _validate_tool_name(name: str, tools: list[Tool]) -> None:
        """Reject canonical invokes for tools absent from this request.

        ``BaseFormatDetector.parse_base_json`` historically drops unknown tools.
        Once a model emitted a canonical DSML invoke, silently dropping it turns
        an attempted tool call into a false ``end_turn`` success.  A streaming
        client also cannot safely accept the name before this check.
        """
        declared = {tool.function.name for tool in tools if tool.function.name}
        if name in declared or envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
            return
        logger.warning("DeepSeek output attempted undeclared tool %r", name)
        DeepSeekV32Detector._raise_malformed_tool_protocol()

    def _validate_complete_wrapper_structure(self, text: str) -> list[tuple[int, int]]:
        """Require complete, non-nested wrappers outside invoke payloads.

        Tool argument data is opaque. In particular, direct-JSON arguments may
        legitimately contain literal DSML wrapper strings, so wrapper balance
        is evaluated only in the protocol layer around complete invokes.
        """
        invoke_spans = [
            (match.start(), match.end())
            for match in re.finditer(self.invoke_regex, text, re.DOTALL)
            if self._unpack_invoke_match(match)[2]
        ]

        def is_inside_invoke(offset: int) -> bool:
            return any(start < offset < end for start, end in invoke_spans)

        events: list[tuple[int, bool]] = []
        for is_open, token in ((True, self.bot_token), (False, self.eot_token)):
            start = 0
            while (offset := text.find(token, start)) != -1:
                if not is_inside_invoke(offset):
                    events.append((offset, is_open))
                start = offset + len(token)

        depth = 0
        content_start = -1
        wrapper_spans: list[tuple[int, int]] = []
        for offset, is_open in sorted(events):
            if is_open:
                if depth != 0:
                    self._raise_malformed_tool_protocol()
                depth = 1
                content_start = offset + len(self.bot_token)
            else:
                if depth != 1:
                    self._raise_malformed_tool_protocol()
                depth = 0
                wrapper_spans.append((content_start, offset))
        if depth != 0:
            self._raise_malformed_tool_protocol()
        return wrapper_spans

    def _ends_with_partial_tool_tag(self, text: str) -> bool:
        """Return whether ``text`` ends inside a DSML tool-call tag."""
        stripped = text.rstrip()
        last_lt = stripped.rfind("<")
        if last_lt == -1:
            return False
        tail = stripped[last_lt:]
        return any(tag.startswith(tail) for tag in self._DSML_TOOL_TAGS)

    @staticmethod
    def _ends_with_incomplete_reserved_dsml_tag(text: str) -> bool:
        """Hold a split reserved tag until its name can be classified."""
        stripped = text.rstrip()
        last_lt = stripped.rfind("<")
        if last_lt == -1:
            return False
        tail = stripped[last_lt:]
        return tail.startswith(("<｜DSML｜", "</｜DSML｜")) and ">" not in tail

    @staticmethod
    def _unpack_invoke_match(m: "re.Match[str]") -> tuple[str, str, bool]:
        """Returns (name, body, is_complete) for an invoke_regex match.

        Self-closing invokes have empty body and are always complete.
        Long-form bodies are always strings (possibly empty); they're
        incomplete when matched against `$` because the closing tag
        hasn't streamed in yet.
        """
        name = m.group("name").strip()
        if m.group("self_close"):
            return name, "", True
        return name, m.group("body"), bool(m.group("end"))

    @staticmethod
    def _remove_partial_closing_token(value: str, token: str) -> str:
        """Remove the longest trailing prefix of a structural closing token.

        Streaming tokenization can split a closer at every character. The old
        component-wise ``removesuffix`` loop only removed a few exact fragments
        and let shapes such as ``</｜DSML｜invo`` enter public JSON arguments.
        """
        max_size = min(len(value), len(token) - 1)
        for size in range(max_size, 0, -1):
            if token.startswith(value[-size:]):
                return value[:-size]
        return value

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = _strip_dsml_protocol_padding(invoke_content)
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                # Once a complete object exists, freeze the argument at its JSON
                # boundary. Any later bytes belong to the invoke delimiter (or
                # malformed residue that the complete parse will reject).
                try:
                    value, end = json.JSONDecoder().raw_decode(invoke_content_stripped)
                except (json.JSONDecodeError, ValueError):
                    return self._remove_partial_closing_token(
                        invoke_content_stripped,
                        self.invoke_end_token,
                    )
                if isinstance(value, dict):
                    return invoke_content_stripped[:end]
                return invoke_content_stripped

            try:
                direct_parameters = json.loads(invoke_content_stripped)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                raise MalformedToolCallError(
                    "Model emitted malformed DeepSeek tool-call parameters"
                ) from e
            if not isinstance(direct_parameters, dict):
                raise MalformedToolCallError(
                    "Model emitted non-object DeepSeek tool-call parameters"
                )
            return invoke_content_stripped

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        for match in param_matches:
            if not allow_partial and _strip_dsml_protocol_padding(
                invoke_content[last_match_end : match.start()]
            ):
                self._raise_malformed_tool_protocol()

            param_name = match.group(1)
            param_type = match.group(2)
            param_value = match.group(3)
            last_match_end = match.end()

            if param_name in parameters or param_type not in {"true", "false"}:
                self._raise_malformed_tool_protocol()
            if not allow_partial and _DSML_PROTOCOL_IN_VALUE_RE.search(param_value):
                # Structural tags inside a parameter value are ambiguous with a
                # damaged/nested parameter. Ordinary '<', shell redirection and
                # non-structural DSML examples remain valid argument data.
                self._raise_malformed_tool_protocol()

            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = (
                    param_value
                    if self.preserve_string_whitespace
                    else param_value.strip()
                )
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError) as e:
                    if not allow_partial:
                        raise MalformedToolCallError(
                            "Model emitted malformed DeepSeek tool-call parameters"
                        ) from e
                    parameters[param_name] = param_value.strip()

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            remaining_content = invoke_content[last_match_end:]

            # Keep a split closing tag out of the partial parameter value.
            remaining_content = self._remove_partial_closing_token(
                remaining_content,
                self.parameter_end_token,
            )

            # Match start of a parameter tag + value (potentially incomplete)
            # Regex: <tag name="..." string="...">VALUE... (no end tag)
            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )

            if partial_match and (param_value := partial_match.group(3)):
                param_name = partial_match.group(1)
                if partial_match.group(2) == "true":
                    parameters[param_name] = (
                        param_value
                        if self.preserve_string_whitespace
                        else param_value.strip()
                    )
                else:
                    try:
                        parameters[param_name] = _partial_json_loads(
                            param_value, Allow.ALL
                        )[0]
                    except (json.JSONDecodeError, MalformedJSON, ValueError):
                        parameters[param_name] = param_value.strip()

        if not allow_partial and _strip_dsml_protocol_padding(
            invoke_content[last_match_end:]
        ):
            self._raise_malformed_tool_protocol()

        return json.dumps(parameters, ensure_ascii=False)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        self._reject_malformed_generic_tool_call(text, tools)
        self._reject_malformed_reserved_dsml_tag(text)
        if self._ends_with_incomplete_reserved_dsml_tag(text):
            self._raise_malformed_tool_protocol()
        has_wrapper = self.bot_token in text
        has_bare_invoke = self.invoke_start_token in text
        idx = (
            text.find(self.bot_token)
            if has_wrapper
            else text.find(self.invoke_start_token)
        )
        normal_text = self._trim_tool_boundary_prefix(text[:idx]) if idx != -1 else text
        if not has_wrapper and not has_bare_invoke:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        wrapper_spans = self._validate_complete_wrapper_structure(text)

        calls = []
        try:
            invoke_matches = list(re.finditer(self.invoke_regex, text, re.DOTALL))
            if not invoke_matches:
                self._raise_malformed_tool_protocol()

            # Every wrapper is a protocol-only region. Validate its residue
            # around invokes without using a non-greedy wrapper regex, which
            # would mistake a literal ``</...tool_calls>`` inside direct JSON
            # argument data for the real outer closer.
            for section_start, section_end in wrapper_spans:
                section_matches = [
                    match
                    for match in invoke_matches
                    if section_start <= match.start() and match.end() <= section_end
                ]
                if not section_matches:
                    self._raise_malformed_tool_protocol()
                cursor = section_start
                for invoke_match in section_matches:
                    if _strip_dsml_protocol_padding(
                        text[cursor : invoke_match.start()]
                    ):
                        self._raise_malformed_tool_protocol()
                    cursor = invoke_match.end()
                if _strip_dsml_protocol_padding(text[cursor:section_end]):
                    self._raise_malformed_tool_protocol()

            for invoke_match in invoke_matches:
                func_name, invoke_content, is_complete = self._unpack_invoke_match(
                    invoke_match
                )
                if not is_complete:
                    self._raise_malformed_tool_protocol()
                self._validate_tool_name(func_name, tools)
                func_args = self._parse_parameters_from_xml(invoke_content)
                match_result = {
                    "name": func_name,
                    "parameters": json.loads(func_args),
                }
                calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except MalformedToolCallError:
            raise
        except Exception as e:
            logger.exception("Error in detect_and_parse")
            raise MalformedToolCallError(
                "Model emitted malformed DeepSeek tool-call protocol"
            ) from e

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks and argument streaming.
        """
        self._stream_text += new_text
        self._buffer += new_text
        current_text = self._buffer
        self._reject_malformed_generic_tool_call(current_text, tools)
        self._reject_malformed_reserved_dsml_tag(current_text)

        # Check if buffer contains any DSML tool-call markers or ends with a
        # partial tag prefix.  Use specific tool-call tags instead of bare
        # "｜DSML｜" to avoid trapping response text that merely mentions DSML
        # sub-tags (e.g. <｜DSML｜parameter> in explanatory text).
        potentially_dsml = any(tag in current_text for tag in self._DSML_TOOL_TAGS)

        # Check if text ends with a partial prefix of any DSML tool-call tag.
        # Find the last "<" in the text; everything from there to the end is a
        # candidate partial tag.  If any complete tag starts with this tail,
        # we buffer and wait for more data.
        ends_with_prefix = self._ends_with_partial_tool_tag(
            current_text
        ) or self._ends_with_partial_generic_invoke(current_text)
        ends_with_prefix = (
            ends_with_prefix
            or self._ends_with_incomplete_reserved_dsml_tag(current_text)
        )

        if (
            not self.has_tool_call(current_text)
            and not potentially_dsml
            and not ends_with_prefix
        ):
            # A standalone whitespace token is ambiguous: it can be ordinary
            # response text, or structural padding immediately before/after a
            # DSML tool-call delimiter.  Hold it until the next chunk resolves
            # that ambiguity.  ``finish`` flushes it for a non-tool response,
            # while a confirmed tool call consumes it as wrapper structure.
            if current_text and not _strip_dsml_protocol_padding(current_text):
                return StreamingParseResult()
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        all_calls: list[ToolCallItem] = []
        normal_text_parts: list[str] = []
        ordered_items: list[str | ToolCallItem] = []

        def append_normal_text(text: str) -> None:
            if not text:
                return
            normal_text_parts.append(text)
            ordered_items.append(text)

        def append_call(call: ToolCallItem) -> None:
            all_calls.append(call)
            ordered_items.append(call)

        def normal_prefix(end: int) -> str:
            """Return ordinary text before an invoke, excluding DSML wrappers."""
            prefix = current_text[:end]
            bot_pos = current_text.rfind(self.bot_token, 0, end)
            if bot_pos != -1:
                prefix = current_text[:bot_pos]
            for token in (self.eot_token, self.invoke_end_token):
                prefix = prefix.replace(token, "")
            if self.eot_token in current_text[:end]:
                prefix = prefix.lstrip("\n")
            return self._trim_tool_boundary_prefix(prefix)

        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=self.invoke_regex,
                    string=current_text,
                    flags=re.DOTALL,
                )
                if not invoke_match:
                    break

                func_name, invoke_content, is_tool_end = self._unpack_invoke_match(
                    invoke_match
                )
                self._validate_tool_name(func_name, tools)

                # Record the text before each invoke.  The first call's
                # preamble and any text between later calls are both normal
                # response content, not part of the tool-call payload.
                first_tool_call = self.current_tool_id == -1
                if first_tool_call:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                call_start = invoke_match.start()
                bot_pos = current_text.rfind(self.bot_token, 0, call_start)
                if bot_pos != -1:
                    call_start = bot_pos
                # Same trailing-newline trim as detect_and_parse, so both agree.
                append_normal_text(normal_prefix(call_start))

                # An incomplete invoke remains buffered and is reparsed for every
                # subsequent stream chunk.  Drop the ordinary prefix as soon as it
                # has been emitted, while retaining the DSML wrapper/invoke, so the
                # same preamble or inter-call text cannot be replayed on each retry.
                self._buffer = current_text[call_start:]

                # Ensure arrays are large enough for current tool
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # 1. Send tool name if not sent yet
                if not self.current_tool_name_sent:
                    append_call(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True

                # 2. Parse current parameters (partial or complete)
                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end
                )

                # 3. Calculate and send incremental arguments
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff = None

                if is_tool_end:
                    # If complete, send everything remaining
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None:
                    # If partial, send stable prefix diff
                    if current_params != prev_params:
                        prefix = _find_common_prefix(current_params, prev_params)
                        if len(prefix) > sent_len:
                            argument_diff = prefix[sent_len:]

                if argument_diff:
                    append_call(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # Update the stored arguments
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer  # Update for next iteration

                    # Move to next tool call
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # Preserve ordinary text after a completed invoke.  Previously this
            # suffix stayed in ``_buffer`` but was omitted from the result, so
            # streaming responses lost text generated after a tool call.
            postamble = current_text
            had_wrapper_end = self.eot_token in postamble
            for token in (self.eot_token, self.invoke_end_token):
                postamble = postamble.replace(token, "")
            if had_wrapper_end:
                postamble = postamble.lstrip("\n")

            if (
                postamble
                and not any(tag in postamble for tag in self._DSML_TOOL_TAGS)
                and not self._ends_with_partial_tool_tag(postamble)
                and not self._ends_with_incomplete_reserved_dsml_tag(postamble)
                and not self._ends_with_partial_generic_invoke(postamble)
            ):
                if (
                    not _strip_dsml_protocol_padding(postamble)
                    and self.current_tool_id != -1
                    and self.prev_tool_call_arr
                ):
                    # A newline between ``</invoke>`` and the closing tool-call
                    # wrapper may arrive as its own token.  Keep it buffered so
                    # the wrapper consumes it instead of exposing an empty text
                    # block to OpenAI/Anthropic clients.
                    self._buffer = "" if had_wrapper_end else postamble
                else:
                    append_normal_text(postamble)
                    self._buffer = ""

            return StreamingParseResult(
                normal_text="".join(normal_text_parts),
                calls=all_calls,
                ordered_items=ordered_items,
            )

        except MalformedToolCallError:
            raise
        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            # Do NOT clear self._buffer — retaining it lets the next chunk
            # retry parsing with more data.  Reset transient tool state so
            # a partial parse doesn't corrupt the next attempt.
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]
            normal_text = "".join(normal_text_parts)
            return StreamingParseResult(
                normal_text=normal_text,
                ordered_items=[normal_text] if normal_text else [],
            )

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        """Flush any text trapped in the buffer when the stream ends.

        Ordinary held text is released. Reserved tool-call syntax must form a
        balanced wrapper with complete invokes; otherwise the model attempted a
        tool call that cannot safely be represented as either a tool result or
        normal text, so termination fails closed instead of fabricating an
        ``end_turn`` success.
        """
        stream_text = self._stream_text
        self._stream_text = ""
        self._validate_complete_wrapper_structure(stream_text)

        if not self._buffer:
            return StreamingParseResult()
        buffered = self._buffer
        self._buffer = ""
        stripped = buffered.rstrip()
        last_lt = stripped.rfind("<")
        trailing_tag = stripped[last_lt:] if last_lt != -1 else ""
        if trailing_tag.startswith(("<｜DSML", "</｜DSML")) and ">" not in trailing_tag:
            logger.warning("DeepSeek tool-call protocol ended mid-tag or wrapper")
            self._raise_malformed_tool_protocol()
        # If we successfully parsed tool calls, the buffer holds only
        # trailing whitespace or closing tags — don't emit those.
        if self.current_tool_id != -1 and self.prev_tool_call_arr:
            remainder = _strip_dsml_protocol_padding(
                buffered.replace(self.eot_token, "")
            )
            if remainder:
                logger.warning(
                    "DeepSeek tool-call protocol left trailing unparsed content"
                )
                self._raise_malformed_tool_protocol()
            return StreamingParseResult()
        # No tool calls were parsed — the buffer is trapped text.
        # Strip everything from the earliest DSML tool-call tag onward:
        # tags are delimiters, not content.  Use _DSML_TOOL_TAGS (without
        # ">") so partial tags like "<｜DSML｜tool_calls" are also stripped.
        if self.has_tool_call(buffered) or any(
            tag in buffered for tag in self._DSML_TOOL_TAGS
        ):
            logger.warning("DeepSeek tool-call intent ended without a parsed invoke")
            self._raise_malformed_tool_protocol()

        earliest = len(buffered)
        for tag in self._DSML_TOOL_TAGS:
            idx = buffered.find(tag)
            if idx != -1 and idx < earliest:
                earliest = idx
        buffered = buffered[:earliest]
        return StreamingParseResult(normal_text=buffered)

    @staticmethod
    def _trim_tool_boundary_prefix(text: str) -> str:
        """Remove structural padding before a tool call without losing prose."""
        text = text.removesuffix("\n\n")
        return "" if text and not _strip_dsml_protocol_padding(text) else text

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'{self.invoke_start_token} name="{name}">',
            end=self.invoke_end_token,
            trigger=self.invoke_start_token,
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v3_2"
