"""Only real, replicated rows from an eager target forward can enter the miner."""
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from sglang.srt import pearl_runtime as pearl


class PearlPaddingTest(TestCase):
    def test_dispatch(self):
        layer = Mock()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(name="EXTEND"),
            extend_num_tokens=16,
            global_num_token_non_padded_cpu=12,
            attn_tp_sequence_sharded=False,
        )
        x = SimpleNamespace(shape=(16, 128))
        with (patch.object(pearl, "_rank", SimpleNamespace(active=True)),
              patch.object(pearl, "_capturing", return_value=False),
              patch.object(pearl, "SHAPE_DIR", None),
              patch.object(pearl, "OPERATOR") as operator):
            self.assertEqual(pearl.real_rows(batch), 12)
            pearl.project(layer, x, None, batch)
            operator.project.assert_not_called()
            layer.quant_method.apply.assert_called_once_with(layer, x, bias=None)
            batch.global_num_token_non_padded_cpu = 16
            batch.attn_tp_sequence_sharded = True
            self.assertIsNone(pearl.real_rows(batch))
            pearl.project(layer, x, None, batch)
            operator.project.assert_not_called()
            batch.attn_tp_sequence_sharded = False
            pearl.project(layer, x, None, batch)
            operator.project.assert_called_once_with(layer, x, None, batch)


if __name__ == "__main__":
    main()
