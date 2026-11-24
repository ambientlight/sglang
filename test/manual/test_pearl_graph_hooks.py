"""Run with python3. Only the public TP4 DSPARK target-verify canary enables these hooks."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

source = Path(__file__).resolve().parents[2] / "python/sglang/srt/pearl_runtime.py"
spec = importlib.util.spec_from_file_location("graph_hook", source)
hook = importlib.util.module_from_spec(spec)
with patch.dict(os.environ, {"PEARL_WOB_MODULE": "", "PEARL_NATIVE_IDLE_CONFIG": ""}):
    spec.loader.exec_module(hook)


def batch(bs=1, **kwargs):
    attrs = dict(forward_mode=NS(name="TARGET_VERIFY", is_target_verify=lambda: True),
                 batch_size=bs, global_num_token_non_padded_cpu=bs * 6, ragged_verify_layout=None, spec_info=NS(draft_token_num=6), attn_tp_sequence_sharded=False)
    attrs.update(kwargs)
    return NS(**attrs)


def runner():
    return NS(model_runner=NS(gpu_id=3, is_draft_worker=False, spec_algorithm=NS(is_dspark=lambda: True)),
              bs=1, raw_bs=1, raw_num_token=6, captured_req_width=6, ragged_verify_mode=False)


class GraphHooks(unittest.TestCase):
    def test_native_rank_drains_before_control_and_fences_before_idle(self):
        rank = hook.Rank(2, "/unused")
        rank.native_idle = object()
        order = []
        rank._command = order.append
        rank.receive((), control=True)
        rank.receive((), control=True)
        rank.idle(lambda: order.append("gpu fence"))
        self.assertEqual(order, ["BUSY 2", "gpu fence", "IDLE 2"])

    def setUp(self):
        self.op = MagicMock()
        self.op.begin_graph.return_value = ("target_graph_mined", "state")
        self.rank = hook.Rank(3, None)
        self.rank.active = True
        self.patch = patch.multiple(hook, GRAPH_VERIFY=True, OPERATOR=self.op, _rank=self.rank, SHAPE_DIR="trace")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_replay_gates_on_real_complete_target_rows(self):
        r = runner()
        self.assertEqual(hook.before_replay(r, batch()), "state")
        self.op.begin_graph.assert_called_with(3, 6, 6, True)
        r.bs = 4
        hook.before_replay(r, batch())
        self.op.begin_graph.assert_called_with(3, 6, 24, False)
        r.raw_bs = r.bs = 1
        for b in (batch(attn_tp_sequence_sharded=True), batch(_original_forward_mode="PADDED"),
                  batch(spec_info=NS(ragged_verify_layout=NS(total_verify_tokens=None)))):
            hook.before_replay(r, b)
            self.op.begin_graph.assert_called_with(3, None, 6, False)
        for b in (batch(global_num_token_non_padded_cpu=5),
                  batch(ragged_verify_layout=NS(total_verify_tokens=6))):
            hook.before_replay(r, b)
            self.op.begin_graph.assert_called_with(3, 6, 6, False)
        self.rank.active = False
        hook.before_replay(r, batch())
        self.op.begin_graph.assert_called_with(3, 6, 6, False)
        r.model_runner.is_draft_worker = True
        self.op.begin_graph.reset_mock()
        self.assertIsNone(hook.before_replay(r, batch()))
        self.op.begin_graph.assert_not_called()

    def test_only_capture_of_known_target_marks_projection(self):
        r, b = runner(), batch()
        hook.prepare_graph_capture(r, b)
        self.assertTrue(b.pearl_graph_verify_capture)
        self.rank.active = False
        self.assertIs(hook.project(None, None, None, b), self.op.capture_project.return_value)
        r.captured_req_width = 5
        with self.assertRaises(ValueError):
            hook.prepare_graph_capture(r, batch())
        r.model_runner.is_draft_worker = True
        b = batch()
        hook.prepare_graph_capture(r, b)
        self.assertFalse(hasattr(b, "pearl_graph_verify_capture"))

    def test_config_requires_public_peer_full_graph_and_topology(self):
        parallel = NS(num_dp_ranks=1, pp_size=1, tp_size=4)
        flags = NS(capture=NS(enable_torch_compile=False))
        phases = dict(decode="full", prefill="disabled")
        imports = {"sglang.srt.runtime_context": NS(get_parallel=lambda: parallel, get_flags=lambda: flags),
                   "sglang.srt.model_executor.cuda_graph_config": NS(
                       Backend=NS(FULL="full", DISABLED="disabled"), Phase=NS(DECODE="decode", PREFILL="prefill"),
                       check_cuda_graph_backend=lambda p, b: phases[p] == b, cuda_graph_fully_disabled=lambda: False)}
        env = {"PEARL_PUBLIC_REPLAY": "1", "PEARL_REPLAY_POOL_PORT": "1234", "PEARL_GPU_PREPARE": "1"}
        with patch.dict(sys.modules, imports), patch.dict(os.environ, env), patch.object(hook, "SOCKET", "/broker"):
            hook.validate_config()
            for key in env:
                with patch.dict(os.environ, {key: ""}), self.assertRaises(ValueError):
                    hook.validate_config()
            parallel.tp_size = 1
            with self.assertRaises(ValueError):
                hook.validate_config()
            parallel.tp_size = 4
            phases["decode"] = "disabled"
            with self.assertRaises(ValueError):
                hook.validate_config()


if __name__ == "__main__":
    unittest.main()
