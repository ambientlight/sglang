"""V4.1 progress phases and opt-in plaintext reasoning through Responses."""

import asyncio
import unittest

from test_dsv41_api_contract import SCHEMA, chunks, protocol, serving
from utils import StreamFixture, engine_chunk, event_payloads

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)


def request(*, summary=None, store=False):
    return ResponsesRequest(
        model="deepseek-v4-flash",
        input="Check the value, then answer.",
        stream=True,
        store=store,
        tools=[{"type": "function", "name": "Bash", "parameters": SCHEMA}],
        reasoning={"effort": "high", **({"summary": summary} if summary else {})},
    )


def full_response(handler, req, chunk):
    async def run():
        context = SimpleContext()
        context.append_output(chunk)

        async def generator():
            yield context

        response = await handler.responses_full_generator(
            req,
            sampling_params={},
            result_generator=generator(),
            context=context,
            model_name=req.model,
            tokenizer=handler.tokenizer_manager.tokenizer,
            request_metadata=RequestResponseMetadata(request_id=req.request_id),
            require_reasoning=True,
        )
        return response.model_dump()

    return asyncio.run(run())


class V41ResponsesProgressTests(unittest.TestCase):
    def test_preamble_and_final_phases_survive_wire_snapshot_and_store(self):
        for text, expected_phase in (
            ("Checking the value." + protocol(), "commentary"),
            ("The value is 42.", "final_answer"),
            ("Checking." + protocol() + "Waiting for the tool.", "commentary"),
        ):
            for size in (1, 7, 4096):
                with self.subTest(text=text, chunk_size=size):
                    handler = serving()
                    req = request(store=True)
                    output_chunks = chunks(
                        "<think>Inspect the value.</think>" + text, size
                    )
                    payloads = event_payloads(
                        StreamFixture(handler, req).run(output_chunks)
                    )
                    added = [
                        p["item"]
                        for p in payloads
                        if p["type"] == "response.output_item.added"
                        and p["item"]["type"] == "message"
                    ]
                    done = [
                        p["item"]
                        for p in payloads
                        if p["type"] == "response.output_item.done"
                    ]
                    messages = [item for item in done if item["type"] == "message"]
                    self.assertTrue(messages)
                    self.assertEqual(
                        [item["phase"] for item in added], ["commentary"] * len(added)
                    )
                    self.assertEqual(
                        [item["phase"] for item in messages],
                        [expected_phase] * len(messages),
                    )
                    self.assertEqual(
                        [item["id"] for item in added],
                        [item["id"] for item in messages],
                    )
                    final = payloads[-1]
                    self.assertEqual(final["type"], "response.completed")
                    self.assertEqual(final["response"]["output"], done)
                    self.assertEqual(
                        handler.response_store[final["response"]["id"]].model_dump()[
                            "output"
                        ],
                        done,
                    )
                    self.assertEqual(
                        [p["sequence_number"] for p in payloads],
                        list(range(len(payloads))),
                    )

    def test_full_responses_classify_preambles_and_terminal_answers(self):
        for text, phase in (
            ("Checking." + protocol(), "commentary"),
            ("The value is 42.", "final_answer"),
        ):
            for finish in ("stop", "length", "abort"):
                with self.subTest(text=text, finish=finish):
                    chunk = engine_chunk("<think>Inspect.</think>" + text, finish=True)
                    chunk["meta_info"]["finish_reason"] = {"type": finish}
                    output = full_response(serving(), request(), chunk)["output"]
                    self.assertEqual(
                        [item["phase"] for item in output if item["type"] == "message"],
                        [phase if finish == "stop" else "commentary"],
                    )

    def test_unfinished_and_failed_streams_do_not_publish_final_answer(self):
        for finish in ("length", "abort", None):
            with self.subTest(finish=finish):
                chunk = engine_chunk("<think>Inspect.</think>Still checking.")
                chunk["meta_info"]["finish_reason"] = (
                    {"type": finish} if finish else None
                )
                payloads = event_payloads(
                    StreamFixture(serving(), request()).run([chunk])
                )
                messages = [
                    p["item"]
                    for p in payloads
                    if p["type"] == "response.output_item.done"
                    and p["item"]["type"] == "message"
                ]
                self.assertEqual([item["phase"] for item in messages], ["commentary"])

        payloads = event_payloads(
            StreamFixture(serving(), request()).run(
                chunks("<think>Inspect.</think>Checking.<｜DSML｜ calls>", 1)
            )
        )
        self.assertEqual(payloads[-1]["type"], "response.failed")
        messages = [
            p["item"]
            for p in payloads
            if p["type"] == "response.output_item.done"
            and p["item"]["type"] == "message"
        ]
        self.assertEqual([item["phase"] for item in messages], ["commentary"])

    def test_text_still_streams_before_generation_finishes(self):
        async def run():
            handler = serving()
            req = request()
            emitted = []

            async def generate():
                yield engine_chunk("<think>Inspect.</think>Checking.")
                self.assertTrue(
                    any(p["type"] == "response.output_text.delta" for p in emitted)
                )
                yield engine_chunk(
                    "<think>Inspect.</think>Checking." + protocol(), finish=True
                )

            async for frame in handler.responses_stream_generator_non_harmony(
                req,
                sampling_params={},
                result_generator=generate(),
                model_name=req.model,
                tokenizer=handler.tokenizer_manager.tokenizer,
                request_metadata=RequestResponseMetadata(request_id=req.request_id),
                require_reasoning=True,
            ):
                emitted.extend(event_payloads([frame]))
            self.assertEqual(emitted[-1]["type"], "response.completed")

        asyncio.run(run())

    def test_plaintext_reasoning_is_separate_and_summary_is_opt_in(self):
        trace = "Inspect the test value before answering."
        for summary in (None, "auto", "detailed"):
            with self.subTest(summary=summary):
                req = request(summary=summary)
                output_chunks = chunks(f"<think>{trace}</think>The value is 42.", 3)
                payloads = event_payloads(
                    StreamFixture(serving(), req).run(output_chunks)
                )
                expected_summary = (
                    [{"type": "summary_text", "text": trace}] if summary else []
                )
                for output in (
                    payloads[-1]["response"]["output"],
                    full_response(serving(), req, output_chunks[-1])["output"],
                ):
                    reasoning = next(
                        item for item in output if item["type"] == "reasoning"
                    )
                    self.assertEqual(reasoning["summary"], expected_summary)
                    self.assertEqual(
                        reasoning["content"],
                        [{"type": "reasoning_text", "text": trace}],
                    )
                    self.assertIsNone(reasoning.get("encrypted_content"))
                    messages = [item for item in output if item["type"] == "message"]
                    self.assertEqual(
                        [item["content"][0]["text"] for item in messages],
                        ["The value is 42."],
                    )
                delta_type = (
                    "response.reasoning_summary_text.delta"
                    if summary
                    else "response.reasoning_text.delta"
                )
                self.assertEqual(
                    "".join(p["delta"] for p in payloads if p["type"] == delta_type),
                    trace,
                )


if __name__ == "__main__":
    unittest.main()
