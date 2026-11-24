"""Experimental SGLang hooks for host-local Pearl arbitration; disabled by default."""
import asyncio
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

SOCKET = os.environ.get("PEARL_BROKER_SOCKET")
SHAPE_DIR = os.environ.get("PEARL_SHAPE_DIR")
_OPERATOR_MODULE = os.environ.get("PEARL_WOB_MODULE")
OPERATOR = importlib.import_module(_OPERATOR_MODULE) if _OPERATOR_MODULE else None
GRAPH_VERIFY = os.environ.get("PEARL_GRAPH_VERIFY") == "1"
OBSERVE = bool(SHAPE_DIR or OPERATOR or GRAPH_VERIFY)
TIMEOUT = 6.0  # broker's default worker-drain deadline is five seconds
_rank = None


def validate_config():
    if not SOCKET and OPERATOR is None and not GRAPH_VERIFY:
        return
    from sglang.srt.runtime_context import get_flags, get_parallel
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend, Phase, check_cuda_graph_backend, cuda_graph_fully_disabled,
    )
    parallel = get_parallel()
    if SOCKET and (parallel.num_dp_ranks != 1 or parallel.pp_size != 1):
        raise ValueError("Pearl broker handoff requires one host-local TP group (DP1, PP1)")
    if GRAPH_VERIFY:
        if (not SOCKET or OPERATOR is None
                or not all(hasattr(OPERATOR, method) for method in
                           ("capture_project", "begin_graph", "end_graph", "counters"))
                or os.environ.get("PEARL_PUBLIC_REPLAY") != "1"
                or os.environ.get("PEARL_GPU_PREPARE") != "1"
                or not os.environ.get("PEARL_REPLAY_POOL_PORT")):
            raise ValueError("Pearl target graph requires rank admission, a public replay job peer, and GPU preparation")
        if (parallel.tp_size != 4 or get_flags().capture.enable_torch_compile
                or not check_cuda_graph_backend(Phase.DECODE, Backend.FULL)
                or not check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)):
            raise ValueError("Pearl target graph requires TP4, full decode graphs, eager prefill, and disabled torch.compile")
        return
    if OPERATOR is not None and (
        not cuda_graph_fully_disabled() or get_flags().capture.enable_torch_compile
    ):
        raise ValueError("Pearl public projection requires eager prefill and decode. "
                         "Disable both CUDA graph backends and torch.compile.")


async def _rpc(path, line, expected="OK"):
    async with asyncio.timeout(TIMEOUT):
        reader, writer = await asyncio.open_unix_connection(path)
        try:
            writer.write((line + "\n").encode("ascii"))
            await writer.drain()
            reply = await reader.readline()
            if reply != (expected + "\n").encode("ascii"):
                raise RuntimeError(f"Pearl {line.split()[0]} failed: {reply[:160]!r}")
        finally:
            writer.close()
            await writer.wait_closed()


async def _ordered_rpc(path, line, expected="OK"):
    # An incoming cancellation may not roll back a command until the broker
    # has processed it. Otherwise a late DISPATCH could strand a claim.
    op = asyncio.create_task(_rpc(path, line, expected))
    try:
        return await asyncio.shield(op)
    except asyncio.CancelledError:
        await op
        raise


class Admission:
    """Own admission until every wire dispatch has acquired its own child lease."""

    def __init__(self, path):
        self.path = path
        self.token = uuid.uuid4().hex
        self.reader = self.writer = None
        self.drained = False

    async def start(self):
        # A dedicated socket owns this parent. The broker observes EOF even
        # before DRAINED, so tokenizer death or cancellation frees just this lease.
        async with asyncio.timeout(TIMEOUT):
            self.reader, self.writer = await asyncio.open_unix_connection(self.path)
            self.writer.write(f"LEASE {self.token}\n".encode("ascii"))
            await self.writer.drain()

    async def wait(self):
        if not self.drained:
            async with asyncio.timeout(TIMEOUT):
                reply = await self.reader.readline()
            if reply != b"DRAINED\n":
                raise RuntimeError(f"Pearl LEASE failed: {reply[:160]!r}")
            self.drained = True

    async def close(self):
        if self.writer is None:
            return
        try:
            if self.drained:
                async with asyncio.timeout(TIMEOUT):
                    self.writer.write(f"CANCEL {self.token}\n".encode("ascii"))
                    await self.writer.drain()
                    if await self.reader.readline() != b"OK\n":
                        raise RuntimeError("Pearl admission cancellation failed")
        finally:
            self.writer.close()
            await self.writer.wait_closed()
            self.writer = self.reader = None

    async def dispatch(self, send, payload, items):
        await self.wait()
        children = []
        try:
            for item in items:
                child = uuid.uuid4().hex
                children.append(child)
                # Fields survive both the msgspec wire codec and TP broadcast.
                # Opaque tokens are generated locally and never disclose client rids.
                item.pearl_dispatch_id = child
                await _ordered_rpc(self.path, f"DISPATCH {child} {self.token}")
        except BaseException:
            for child in children:
                await _ordered_rpc(self.path, f"DONE {child}")
            raise
        # An ambiguous send exception leaves the child held. Releasing it here
        # could race a message already committed to the scheduler IPC queue.
        send(payload)


class Rank:
    """One ordered persistent stream owned by the scheduler on one TP rank."""

    def __init__(self, index, path, shape_dir=None):
        self.index = index
        self.path = path
        self.shape_dir = Path(shape_dir) if shape_dir else None
        self.counts = Counter()
        self.next_receipt = 0.0
        self.last_counters = None
        self.stream = None
        self.active = None

    def _command(self, line):
        if not self.path:
            return
        if self.stream is None:
            stream = socket.socket(socket.AF_UNIX)
            stream.settimeout(TIMEOUT)
            try:
                stream.connect(self.path)
            except BaseException:
                stream.close()
                raise
            self.stream = stream
        self.stream.sendall((line + "\n").encode("ascii"))
        # Exactly one command is outstanding on this stream.
        data = bytearray()
        while b"\n" not in data and len(data) < 256:
            fragment = self.stream.recv(256)
            if not fragment:
                break
            data.extend(fragment)
        if data != b"OK\n":
            self.close()
            raise RuntimeError(f"Pearl rank command failed: {data[:160]!r}")

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def receive(self, items):
        if not items:
            return
        if self.active is not True:
            self._command(f"BUSY {self.index}")
            self.active = True
        if self.path:
            for item in items:
                child = item.pearl_dispatch_id
                if not child:
                    raise RuntimeError("tokenized request reached scheduler without a Pearl admission")
                self._command(f"TAKE {child} {self.index}")

    def _write_graph_counters(self, force=False):
        now = time.monotonic()
        if not force and now < self.next_receipt:
            return
        self.next_receipt = now + 0.1
        body = json.dumps(OPERATOR.counters(), sort_keys=True, separators=(",", ":")) + "\n"
        if body == self.last_counters:
            return
        out = self.shape_dir / f"pearl-counters-{os.getpid()}-tp{self.index}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(body)
        os.replace(tmp, out)
        self.last_counters = body

    def idle(self, synchronize):
        if self.active is False:
            # A proof or peer response can finish after the final rank fence. Refresh
            # changed receipts on idle polls without delaying handoff for their completion.
            if GRAPH_VERIFY and self.shape_dir and self.counts:
                self._write_graph_counters()
            return
        if self.path:
            # The forward, sampler, transfers and NCCL may use different streams.
            # Synchronize once on this rank's actual device per busy-to-idle edge.
            synchronize()
        self._command(f"IDLE {self.index}")
        self.active = False
        if self.shape_dir and self.counts:
            out = self.shape_dir / f"pearl-shapes-{os.getpid()}-tp{self.index}.json"
            tmp = out.with_suffix(".tmp")
            rows = [dict(mode=mode, path=path, real_rows=real, executed_rows=executed, calls=n)
                    for (mode, path, real, executed), n in self.counts.items()]
            tmp.write_text(json.dumps(rows, separators=(",", ":")) + "\n")
            os.replace(tmp, out)
            if GRAPH_VERIFY:
                self._write_graph_counters(force=True)


def scheduler_rank(index):
    global _rank
    if _rank is None:
        _rank = Rank(index, SOCKET, SHAPE_DIR)
    elif _rank.index != index:
        raise RuntimeError("Pearl prototype expects one TP scheduler per process")
    return _rank


def real_rows(batch):
    # DP attention padding can replace the visible forward mode and batch size.
    # Under sequence sharding the CPU count is global. Reading the local device
    # scalar here would synchronize inference just to decide whether to mine.
    if batch.attn_tp_sequence_sharded or getattr(batch, "_original_forward_mode", None) is not None:
        return None
    mode = batch.forward_mode.name
    if mode not in ("EXTEND", "DECODE", "TARGET_VERIFY"):
        return None
    if mode == "EXTEND":
        count = batch.extend_num_tokens
        unpadded = batch.global_num_token_non_padded_cpu
        return min(count, unpadded) if count is not None and unpadded is not None else count
    original = getattr(batch, "_original_batch_size", None)
    batch_size = batch.batch_size if original is None else original
    if mode == "DECODE":
        return batch_size
    if mode == "TARGET_VERIFY":
        info = batch.spec_info
        layout = getattr(batch, "ragged_verify_layout", None) or getattr(info, "ragged_verify_layout", None)
        if layout is not None:
            # Device-only ragged layouts can omit the host total. Skip that
            # measurement instead of counting rounded capture-tier padding.
            return layout.total_verify_tokens
        width = getattr(info, "draft_token_num", None)
        return batch_size * width if width is not None else None
    return None


def _record(batch, path, rows, executed):
    if SHAPE_DIR and _rank is not None and _rank.active and rows is not None and 0 < rows <= executed:
        _rank.counts[batch.forward_mode.name, path, rows, executed] += 1


def _capturing():
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    return get_is_capture_mode()


def project(layer, x, bias, batch):
    # Only the full target graph carries this marker, including its two warmup forwards.
    if GRAPH_VERIFY and getattr(batch, "pearl_graph_verify_capture", False):
        return OPERATOR.capture_project(layer, x, bias, batch)
    if _rank is not None and _rank.active and not _capturing():
        rows = real_rows(batch)
        executed = (x[0] if isinstance(x, tuple) else x).shape[0]
        _record(batch, "eager_wo_b", rows, executed)
        if OPERATOR is not None and rows == executed and rows > 0 and not isinstance(x, tuple):
            return OPERATOR.project(layer, x, bias, batch)
    return layer.quant_method.apply(layer, x, bias=bias)


def prepare_graph_capture(runner, batch):
    if runner.model_runner.is_draft_worker:
        return
    if (not runner.model_runner.spec_algorithm.is_dspark() or runner.ragged_verify_mode
            or not batch.forward_mode.is_target_verify() or runner.captured_req_width != 6
            or batch.batch_size not in range(1, 9)):
        raise ValueError("Pearl public target capture requires 1–8 fixed six-row DSPARK verify slots")
    batch.pearl_graph_verify_capture = True


def before_replay(runner, batch):
    # Real CPU batch metadata is available here even when the captured forward pads it.
    if runner.model_runner.is_draft_worker:
        return
    active = _rank is not None and _rank.active
    if not active and not GRAPH_VERIFY:
        return
    rows = real_rows(batch)
    executed = runner._replay_graph_key.size if runner.ragged_verify_mode else runner.bs * runner.captured_req_width
    _record(batch, "target_graph", rows, executed)
    if GRAPH_VERIFY:
        eligible = bool(active and batch.forward_mode.is_target_verify() and rows == executed
                        and not runner.ragged_verify_mode and runner.raw_bs == runner.bs
                        and runner.raw_num_token == executed and batch.global_num_token_non_padded_cpu == executed
                        and batch.ragged_verify_layout is None
                        and getattr(batch.spec_info, "ragged_verify_layout", None) is None)
        path, state = OPERATOR.begin_graph(runner.model_runner.gpu_id, rows, executed, eligible)
        _record(batch, path, rows, executed)
        return state
    if OPERATOR is not None:
        raise RuntimeError("Pearl public projection requires eager execution for both phases")
