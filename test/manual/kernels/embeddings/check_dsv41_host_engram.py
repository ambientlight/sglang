"""Replay mapped CPU Engram rows from the original DSV4.1 checkpoint on SM120."""

import argparse
import json
import struct
from pathlib import Path

import torch

from sglang.kernels.ops.embeddings.engram_gather import engram_gather


def read_rows(root, weights, key, chunk, rows):
    with (root / weights[key]).open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(size))[key]
        total, dim = meta["shape"]
        assert (dim, meta["dtype"]) == ((8, "F8_E8M0") if key.endswith(".scale") else (256, "F8_E4M3"))
        lo = total * chunk // 4
        begin, end = meta["data_offsets"]
        assert end - begin == total * dim
        f.seek(8 + size + begin + lo * dim)
        values = torch.frombuffer(bytearray(f.read(rows * dim)), dtype=torch.uint8).reshape(rows, dim)
    return lo, values


def check(model, weights, layer, chunk):
    rows = 513
    lo, w = read_rows(model, weights, f"layers.{layer}.engram.embed.weight", chunk, rows)
    scale_lo, s = read_rows(model, weights, f"layers.{layer}.engram.embed.scale", chunk, rows)
    assert scale_lo == lo
    host_w, host_s = w.pin_memory(), s.pin_memory()
    reference = (
        w.view(torch.float8_e4m3fn).float() * s.view(torch.float8_e8m0fnu).float().repeat_interleave(32, dim=-1)
    ).bfloat16()
    ids = torch.cat((torch.tensor([-1, lo - 1, lo, lo + rows - 1, lo + rows]), torch.arange(lo + rows - 1, lo - 1, -1)))
    device_ids = ids.cuda()
    out = torch.empty((len(ids), 256), dtype=torch.bfloat16, device="cuda")

    def launch():
        return engram_gather(
            host_w.data_ptr(), host_s.data_ptr(), device_ids, out, 256, 32, row_lo=lo, row_hi=lo + rows, sort_ids=True
        )

    def assert_bits(ids):
        valid = (lo <= ids) & (ids < lo + rows)
        expected = torch.zeros_like(out, device="cpu")
        expected[valid] = reference[ids[valid] - lo]
        assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16))

    launch()
    assert_bits(ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    new_ids = ids.flip(0)
    device_ids.copy_(new_ids)
    graph.replay()
    assert_bits(new_ids)
    return dict(
        layer=layer,
        first_global_row=lo,
        weight_rows=rows,
        outputs_per_forward=out.numel(),
        sorted_host_gather_exact=True,
        changed_input_graph_exact=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    weights = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    assert torch.cuda.get_device_capability() == (12, 0)
    cases = [check(args.model, weights, layer, shard) for layer in (1, 14) for shard in range(4)]
    result = dict(
        model=args.model.name,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        scope="Eight small pinned CPU slices; hardware gather used by EngramEmbedding._owned_rows. The full table, sharding topology and model were not loaded.",
        cases=cases,
    )
    print(json.dumps(result, indent=2))
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
