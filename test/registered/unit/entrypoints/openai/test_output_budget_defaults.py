"""Operator output limits are missing-field defaults, not hard request caps."""

import unittest

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ResponsesRequest,
)


class OutputBudgetDefaultsTest(unittest.TestCase):
    DEFAULTS = {"max_new_tokens": 32768, "temperature": 1.0, "top_p": 0.95}

    def chat(self, **kwargs):
        return ChatCompletionRequest(
            model="fixture", messages=[{"role": "user", "content": "hi"}], **kwargs
        )

    def test_chat_omitted_limit_uses_operator_default(self):
        self.assertEqual(
            self.chat().to_sampling_params([], self.DEFAULTS)["max_new_tokens"], 32768
        )

    def test_chat_explicit_limits_and_zero_take_precedence(self):
        for kwargs, expected in (
            ({"max_tokens": 65536}, 65536),
            ({"max_completion_tokens": 8192}, 8192),
            ({"max_tokens": 8192, "max_completion_tokens": 65536}, 65536),
            ({"max_completion_tokens": 0}, 0),
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(
                    self.chat(**kwargs).to_sampling_params([], self.DEFAULTS)[
                        "max_new_tokens"
                    ],
                    expected,
                )

    def test_chat_without_operator_default_is_unchanged(self):
        self.assertIsNone(self.chat().to_sampling_params([], {})["max_new_tokens"])

    def test_responses_omitted_budget_obeys_default_and_context(self):
        request = ResponsesRequest(model="fixture", input="hi")
        # The pinned Responses implementation reserves two framing tokens.
        for context_room, expected in ((100000, 32766), (256, 254)):
            with self.subTest(context_room=context_room):
                self.assertEqual(
                    request.to_sampling_params(context_room, self.DEFAULTS)[
                        "max_new_tokens"
                    ],
                    expected,
                )

    def test_responses_explicit_larger_budget_is_not_capped_at_default(self):
        request = ResponsesRequest(model="fixture", input="hi", max_output_tokens=65536)
        self.assertEqual(
            request.to_sampling_params(100000, self.DEFAULTS)["max_new_tokens"], 65534
        )

    def test_responses_without_operator_default_is_unchanged(self):
        request = ResponsesRequest(model="fixture", input="hi")
        self.assertEqual(
            request.to_sampling_params(100000, {})["max_new_tokens"], 99998
        )

    def test_explicit_sampling_settings_win(self):
        params = self.chat(temperature=0.2, top_p=0.7).to_sampling_params(
            [], self.DEFAULTS
        )
        self.assertEqual((params["temperature"], params["top_p"]), (0.2, 0.7))
        request = ResponsesRequest(
            model="fixture", input="hi", temperature=0.2, top_p=0.7
        )
        params = request.to_sampling_params(100000, self.DEFAULTS)
        self.assertEqual((params["temperature"], params["top_p"]), (0.2, 0.7))


if __name__ == "__main__":
    unittest.main()
