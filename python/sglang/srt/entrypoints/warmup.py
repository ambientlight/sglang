from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


def _make_warmup_req(
    size: int,
    max_new_tokens: int = 30,
    disaggregation_mode: str = "null",
) -> GenerateReqInput:
    """Create a warmup GenerateReqInput with random input_ids."""
    req = GenerateReqInput(
        input_ids=(np.random.randint(2**16, size=[size])).tolist(),
        sampling_params={
            "max_new_tokens": max_new_tokens,
            "temperature": 0.8,
            "stop_token_ids": [1],
            "min_p": 0.0,
        },
    )
    if disaggregation_mode != "null":
        req.bootstrap_room = 0
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
    return req


async def _drain_one(tokenizer_manager: TokenizerManager, req: GenerateReqInput):
    """Send one request and drain it (wait for first token)."""
    await tokenizer_manager.generate_request(req, None).__anext__()


@warmup("moe_w4a4")
async def moe_w4a4(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Warm up CuTe-DSL NVFP4 (W4A4) MoE kernels for SM120 Blackwell.

    Strategy: static kernel (bucketed) for decode, dynamic for prefill.
    - Static: m-bucketing limits unique compilations to ~17 sizes
    - Dynamic: shape-agnostic, compiles once on first use

    Phase 1: Single request — calibration + dynamic kernel compile (prefill)
    Phase 2: Concurrent decode at 2, then 8 — static/micro kernel compilation
    Phase 3: Full concurrency verification
    """
    MAX_CONCURRENT = 8
    DECODE_TOKENS = 16

    # Phase 1: single request for calibration + dynamic kernel compile
    logger.info(
        "moe_w4a4 warmup: Phase 1/3 — single request "
        "(calibration + first kernel compile)..."
    )
    req = _make_warmup_req(256, max_new_tokens=DECODE_TOKENS,
                           disaggregation_mode=disaggregation_mode)
    await _drain_one(tokenizer_manager, req)
    logger.info("moe_w4a4 warmup: Phase 1/3 complete.")

    # Phase 2: concurrent requests to trigger static/micro kernel compilation
    # for decode bucket sizes (m=1..8 in decode).
    for concurrency in (2, MAX_CONCURRENT):
        logger.info(
            f"moe_w4a4 warmup: Phase 2/3 — {concurrency} concurrent requests "
            f"(batch decode kernel compile)..."
        )
        reqs = [
            _make_warmup_req(256, max_new_tokens=DECODE_TOKENS,
                             disaggregation_mode=disaggregation_mode)
            for _ in range(concurrency)
        ]
        tasks = [_drain_one(tokenizer_manager, r) for r in reqs]
        await asyncio.gather(*tasks)
    logger.info("moe_w4a4 warmup: Phase 2/3 complete.")

    # Phase 3: full concurrency verification — should be fast
    logger.info(
        f"moe_w4a4 warmup: Phase 3/3 — {MAX_CONCURRENT} concurrent requests "
        f"(verification, should be fast)..."
    )
    reqs = [
        _make_warmup_req(512, max_new_tokens=DECODE_TOKENS,
                         disaggregation_mode=disaggregation_mode)
        for _ in range(MAX_CONCURRENT)
    ]
    tasks = [_drain_one(tokenizer_manager, r) for r in reqs]
    await asyncio.gather(*tasks)
    logger.info("moe_w4a4 warmup: Phase 3/3 complete. All kernels warm.")


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()
