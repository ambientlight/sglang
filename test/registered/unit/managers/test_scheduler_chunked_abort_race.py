"""Tests for deferred chunked-prefill aborts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only fields touched by abort handling."""

    def __init__(self, rid: str):
        self.rid = rid
        self.req_pool_idx = 1
        self.kv = SimpleNamespace(holds_kv=True)
        self.mamba_pool_idx = None
        self.to_finish = None
        self._finished = False

    def finished(self):
        return self._finished


def _make_scheduler(pending_req, *, chunked_req, running_reqs) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = chunked_req
    scheduler._pending_chunked_abort_req = pending_req
    scheduler.waiting_queue = []
    scheduler.dllm_config = None
    scheduler.grammar_manager = Mock()
    scheduler.disaggregation_mode = None
    scheduler.enable_hicache_storage = False
    scheduler.mm_receiver = None
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.running_batch = SimpleNamespace(reqs=running_reqs)
    scheduler.last_batch = None
    return scheduler


class TestPendingChunkedAbortRace(CustomTestCase):
    def test_req_that_left_chunked_slot_is_aborted(self):
        req = _FakeReq("zombie_rid")
        scheduler = _make_scheduler(req, chunked_req=None, running_reqs=[req])

        scheduler.process_pending_chunked_abort()

        self.assertIsNotNone(req.to_finish, "recorded abort was never applied")
        self.assertIsNone(scheduler._pending_chunked_abort_req)

    def test_finished_req_only_clears_marker(self):
        req = _FakeReq("done_rid")
        req._finished = True
        scheduler = _make_scheduler(req, chunked_req=None, running_reqs=[])

        scheduler.process_pending_chunked_abort()

        self.assertIsNone(req.to_finish)
        self.assertIsNone(scheduler._pending_chunked_abort_req)


if __name__ == "__main__":
    unittest.main(verbosity=2)
