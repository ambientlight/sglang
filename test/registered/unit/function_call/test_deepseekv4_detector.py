"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import MalformedToolCallError
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


def _generic_weather_call(*, mixed: bool = False) -> str:
    if mixed:
        return (
            '<invoke name="get_weather">\n'
            f'{_param("city", "true", "SF")}\n'
            f"</{DSML}invoke>\n</{DSML}tool_calls>"
        )
    return (
        '<invoke name="get_weather">\n'
        '<parameter name="city" string="true">SF</parameter>\n'
        "</invoke>"
    )


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_text_after_tool_call_is_not_dropped(self):
        """Plain text following a completed invoke must survive streaming."""
        text = _weather_call() + "\nThe rest of the answer."
        normal, calls = self._feed([text])

        self.assertEqual(normal, "The rest of the answer.")
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_text_between_multiple_tool_calls_is_not_dropped(self):
        text = _weather_call("SF") + "\nChecking another city.\n" + _weather_call("NY")
        normal, calls = self._feed([text])

        self.assertEqual(normal, "Checking another city.\n")
        self.assertEqual(
            [c.name for c in calls if c.name], ["get_weather", "get_weather"]
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        # A structural newline may arrive before the closing wrapper, but the
        # ordinary prose itself must appear exactly once.
        self.assertEqual(normal.rstrip(), "Prose.")
        self.assertNotIn(DSML, normal)

    def test_split_structural_whitespace_is_not_emitted_as_text(self):
        """Captured token boundaries put newlines before the wrapper, between
        consecutive invokes, and between ``</invoke>`` and ``</tool_calls>``.
        None of those delimiters may become a whitespace-only content block.
        """
        detector = DeepSeekV4Detector()
        chunks = [
            "\n\n",
            f"<{DSML}tool_calls>\n"
            + _invoke("get_weather", _param("city", "true", "SF")),
            "\n",
            _invoke("get_weather", _param("city", "true", "NY")),
            "\n",
            f"</{DSML}tool_calls>",
        ]

        results = [
            detector.parse_streaming_increment(chunk, self.tools) for chunk in chunks
        ]
        results.append(detector.finish(self.tools))

        self.assertEqual("".join(result.normal_text for result in results), "")
        self.assertFalse(
            any(
                result.normal_text.isspace() for result in results if result.normal_text
            )
        )
        self.assertEqual(
            [call.name for result in results for call in result.calls if call.name],
            ["get_weather", "get_weather"],
        )

    def test_pending_whitespace_before_plain_text_is_preserved(self):
        """Whitespace is delayed, not globally stripped, for non-tool output."""
        detector = DeepSeekV4Detector()

        first = detector.parse_streaming_increment("\n\n", self.tools)
        second = detector.parse_streaming_increment("Plain text", self.tools)

        self.assertEqual(first.normal_text, "")
        self.assertEqual(second.normal_text, "\n\nPlain text")

    def test_pending_whitespace_only_response_is_flushed_at_end(self):
        detector = DeepSeekV4Detector()

        incremental = detector.parse_streaming_increment(" \n", self.tools)
        terminal = detector.finish(self.tools)

        self.assertEqual(incremental.normal_text, "")
        self.assertEqual(terminal.normal_text, " \n")

    def test_capture_derived_incomplete_invoke_does_not_replay_preamble(self):
        """A buffered incomplete invoke must not replay its preamble for every
        subsequent token. The capture emitted the same prefix twelve times."""
        detector = DeepSeekV4Detector()
        preamble = "Preparing the command.\n"
        opening = (
            preamble
            + f"<{DSML}tool_calls>\n"
            + f'<{DSML}invoke name="get_weather">\n'
            + f'<{DSML}parameter name="city" string="true">N'
        )

        results = [detector.parse_streaming_increment(opening, self.tools)]
        results.extend(
            detector.parse_streaming_increment("x", self.tools) for _ in range(11)
        )

        self.assertEqual("".join(result.normal_text for result in results), preamble)
        self.assertEqual(
            sum(bool(result.normal_text) for result in results),
            1,
        )
        self.assertEqual(results[0].normal_text, preamble)
        self.assertTrue(all(not result.normal_text for result in results[1:]))
        self.assertEqual(
            [call.name for result in results for call in result.calls if call.name],
            ["get_weather"],
        )
        continuations = [call for result in results[1:] for call in result.calls]
        self.assertTrue(continuations)
        self.assertTrue(all(call.name is None for call in continuations))
        self.assertNotIn(preamble, detector._buffer)

    def test_inter_call_text_before_partial_invoke_is_emitted_once(self):
        """The consume-once rule also applies to prose before a later call."""
        detector = DeepSeekV4Detector()
        inter_call_text = "Checking another city.\n"
        second_call_start = (
            f"<{DSML}tool_calls>\n"
            f'<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="true">N'
        )
        first = detector.parse_streaming_increment(
            _weather_call("SF") + "\n" + inter_call_text + second_call_start,
            self.tools,
        )
        retries = [
            detector.parse_streaming_increment(chunk, self.tools)
            for chunk in ("e", "w", " ", "Y", "o", "r", "k")
        ]
        final = detector.parse_streaming_increment(
            f"</{DSML}parameter></{DSML}invoke></{DSML}tool_calls>",
            self.tools,
        )
        results = [first, *retries, final]

        self.assertEqual(
            "".join(result.normal_text for result in results), inter_call_text
        )
        self.assertEqual(
            [call.name for result in results for call in result.calls if call.name],
            ["get_weather", "get_weather"],
        )

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must retain the buffer for retry; only
        the preamble (text before the first DSML tag) is emitted as
        normal_text so the tool-call text is neither swallowed permanently
        nor duplicated across deltas."""
        detector = DeepSeekV4Detector()
        preamble = "Let me check.\n"

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(
                preamble + _weather_call(), self.tools
            )
            # Buffer is retained for retry — NOT cleared
            self.assertNotEqual(detector._buffer, "")
            self.assertNotIn(preamble, detector._buffer)

        # Mock removed — the retained buffer should now parse successfully
        # on the next delta, proving the retry works.
        second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertEqual(first.normal_text, preamble)
        self.assertEqual(first.calls, [])
        # The tool call is emitted as a call, NOT as normal_text
        self.assertNotIn(preamble, second.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        self.assertTrue(any(c.name == "get_weather" for c in second.calls))

    def test_capture_derived_generic_and_mixed_calls_fail_closed(self):
        """Captured generic openings are protocol drift, not normal text."""
        for mixed in (False, True):
            with self.subTest(mixed=mixed), self.assertRaises(MalformedToolCallError):
                DeepSeekV4Detector().detect_and_parse(
                    _generic_weather_call(mixed=mixed), self.tools
                )

    def test_capture_derived_generic_call_is_held_across_stream_chunks(self):
        detector = DeepSeekV4Detector()
        first = detector.parse_streaming_increment("<in", self.tools)

        self.assertEqual(first.normal_text, "")
        self.assertEqual(first.calls, [])
        with self.assertRaises(MalformedToolCallError):
            detector.parse_streaming_increment(
                'voke name="get_weather">\n<parameter name="city"', self.tools
            )

    def test_inline_generic_xml_example_remains_normal_text(self):
        text = 'The literal `<invoke name="get_weather">` is invalid.'
        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_capture_derived_malformed_reserved_dsml_fails_closed(self):
        samples = [
            f"</{DSML}tool_api_call让头先看看这些文件",
            f"</{DSML} bottled </parameter>",
            f"Text before <{DSML}tool_crossing",
            f"</{DSML}block>",
            f"</{DSML}function_calls> → Fix the malformed closing tag",
            f"Text before <{DSML}tool_calls\n<{DSML}invoke " 'name="get_weather">',
            f'<{DSML}invoke name="get_weather <{DSML}parameter ' "name subject para?>",
            f'<{DSML}invoke name="get_weather" false=""/>',
        ]
        for text in samples:
            with self.subTest(path="non-stream", text=text):
                with self.assertRaises(MalformedToolCallError):
                    DeepSeekV4Detector().detect_and_parse(text, self.tools)
            with self.subTest(path="stream", text=text):
                with self.assertRaises(MalformedToolCallError):
                    DeepSeekV4Detector().parse_streaming_increment(text, self.tools)

    def test_malformed_reserved_dsml_is_held_across_chunks(self):
        detector = DeepSeekV4Detector()
        first = detector.parse_streaming_increment(
            f"Safe preamble. <{DSML}tool_", self.tools
        )

        self.assertEqual(first.normal_text, "")
        self.assertEqual(first.calls, [])
        with self.assertRaises(MalformedToolCallError):
            detector.parse_streaming_increment("crossing", self.tools)

    def test_capture_derived_unclosed_wrapper_fails_across_chunks(self):
        detector = DeepSeekV4Detector()
        first = detector.parse_streaming_increment(
            f"Safe preamble. <{DSML}tool_calls\n", self.tools
        )

        self.assertEqual(first.normal_text, "")
        self.assertEqual(first.calls, [])
        with self.assertRaises(MalformedToolCallError):
            detector.parse_streaming_increment(
                f'<{DSML}invoke name="get_weather">', self.tools
            )

    def test_incomplete_reserved_dsml_is_not_leaked_at_stream_end(self):
        detector = DeepSeekV4Detector()
        detector.parse_streaming_increment(f"Safe preamble. <{DSML}tool_", self.tools)

        result = detector.finish(self.tools)

        self.assertEqual(result.normal_text, "Safe preamble. ")
        self.assertNotIn(DSML, result.normal_text)

    def test_dsml_parameter_example_remains_normal_text(self):
        text = f'The literal `<{DSML}parameter name="x">` is a parameter tag.'
        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_reserved_dsml_text_inside_tool_argument_is_not_rejected(self):
        value = f"literal <{DSML}custom_marker> data"
        text = _wrapped(_invoke("get_weather", _param("city", "true", value)))

        result = DeepSeekV4Detector().detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertIn("custom_marker", result.calls[0].parameters)


if __name__ == "__main__":
    import unittest

    unittest.main()
