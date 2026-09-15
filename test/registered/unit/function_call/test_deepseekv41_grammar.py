"""Compile actual V4.1 constraints and check accepted/rejected completions."""

import unittest
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser


def tool(name="lookup", strict=True):
    return Tool(
        type="function",
        function=Function(
            name=name,
            strict=strict,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
    )


def invoke(name="lookup", body='{"query":"alpha"}'):
    return f'<｜DSML｜ invoke name="{name}">{body}</｜DSML｜ invoke>\n'


def block(*calls):
    return "\n\n<｜DSML｜ calls>\n" + "".join(calls) + "</｜DSML｜ calls>"


class V41CompiledGrammarTest(unittest.TestCase):
    def grammar(self, tools=None, choice="required", **kwargs):
        parser = FunctionCallParser(tools or [tool()], "deepseekv41")
        constraint_type, value = parser.get_structure_constraint(choice, **kwargs)
        self.assertEqual(constraint_type, "structural_tag")
        return xgr.Grammar.from_structural_tag(value)

    def test_required_matches_spaced_wrapper_not_bare_invoke(self):
        grammar = self.grammar()
        self.assertTrue(_is_grammar_accept_string(grammar, block(invoke())))
        self.assertFalse(_is_grammar_accept_string(grammar, invoke()))
        self.assertFalse(
            _is_grammar_accept_string(
                grammar, block(invoke()).replace("｜DSML｜ ", "｜DSML｜")
            )
        )
        self.assertFalse(_is_grammar_accept_string(grammar, "normal prose"))

    def test_strict_fields_reject_wrong_types_missing_extra_fields(self):
        grammar = self.grammar()
        for body in ("{}", '{"query":42}', '{"query":"a","extra":1}', "[]"):
            with self.subTest(body=body):
                self.assertFalse(
                    _is_grammar_accept_string(grammar, block(invoke(body=body)))
                )

    def test_non_strict_keeps_object_requirement_without_enforcing_fields(self):
        grammar = self.grammar([tool(strict=False)])
        for body in ("{}", '{"query":42}', '{"undeclared":"ok"}'):
            self.assertTrue(
                _is_grammar_accept_string(grammar, block(invoke(body=body)))
            )
        for body in ("[]", "42", '"text"', "null"):
            self.assertFalse(
                _is_grammar_accept_string(grammar, block(invoke(body=body)))
            )

    def test_named_choice_selects_only_that_tool(self):
        grammar = self.grammar(
            [tool(), tool("other")],
            choice=ToolChoice(
                type="function", function=ToolChoiceFuncName(name="other")
            ),
        )
        self.assertTrue(_is_grammar_accept_string(grammar, block(invoke("other"))))
        self.assertFalse(_is_grammar_accept_string(grammar, block(invoke())))

    def test_parallel_flag_controls_number_of_calls(self):
        text = block(invoke(), invoke())
        self.assertTrue(
            _is_grammar_accept_string(self.grammar(parallel_tool_calls=True), text)
        )
        self.assertFalse(
            _is_grammar_accept_string(self.grammar(parallel_tool_calls=False), text)
        )

    def test_auto_allows_prose_but_constrains_calls(self):
        grammar = self.grammar(choice="auto")
        self.assertTrue(_is_grammar_accept_string(grammar, "No call is needed."))
        self.assertTrue(
            _is_grammar_accept_string(grammar, "Checking." + block(invoke()))
        )
        self.assertFalse(
            _is_grammar_accept_string(grammar, block(invoke(body='{"query":42}')))
        )
        self.assertFalse(_is_grammar_accept_string(grammar, block(invoke("unknown"))))

    def test_thinking_prefix_has_exactly_one_closer(self):
        grammar = self.grammar(thinking_mode=True)
        self.assertTrue(
            _is_grammar_accept_string(grammar, "reason</think>" + block(invoke()))
        )
        self.assertFalse(_is_grammar_accept_string(grammar, block(invoke())))


if __name__ == "__main__":
    unittest.main()
