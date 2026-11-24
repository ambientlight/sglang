"""Check real TP4 DSV4.1 SM120 FlashInfer attention with c1 and c2 cache pages."""

import argparse
import json
from pathlib import Path

import flashinfer
import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import flash_mla_with_kvcache_sm120


def pack_cache(page_size, generator):
    """Construct page-footer E4M3/E8M0 NOPE and BF16 ROPE from independent rows."""
    pages = 5
    data = torch.randn((pages, page_size, 448), generator=generator).to(torch.float8_e4m3fn)
    rope = torch.randn((pages, page_size, 64), generator=generator).bfloat16()
    scales = torch.randint(-2, 3, (pages, page_size, 7), generator=generator)
    stride = (page_size * 584 + 575) // 576 * 576
    raw = torch.zeros((pages, stride), dtype=torch.uint8)
    content = raw[:, : page_size * 576].view(pages, page_size, 576)
    content[..., :448] = data.view(torch.uint8)
    content[..., 448:] = rope.view(torch.uint8)
    footers = raw[:, page_size * 576 : page_size * 584].view(pages, page_size, 8)
    footers[..., :7] = (127 + scales).to(torch.uint8)
    actual = raw.cuda().as_strided((pages, page_size, 1, 584), (stride, 584, 584, 1))
    keys = data.float() * scales.float().exp2().repeat_interleave(64, dim=-1)
    return actual, torch.cat((keys, rope.float()), dim=-1).view(-1, 512)


def test_case(tokens, extra_page, generator):
    main, main_ref = pack_cache(256, generator)
    extra, extra_ref = pack_cache(extra_page, generator)
    heads = 16  # 64 target query heads / TP4
    q = torch.randn((tokens, heads, 512), generator=generator).bfloat16()
    idx = torch.randint(len(main_ref), (tokens, 128), dtype=torch.int32, generator=generator)
    extra_idx = torch.randint(len(extra_ref), (tokens, 512), dtype=torch.int32, generator=generator)
    # Include the first/last rows of consecutive physical pages and padding.
    for ids, size in ((idx, 256), (extra_idx, extra_page)):
        ids[:, :4] = torch.tensor([size - 1, size, 2 * size - 1, 2 * size])
        ids[:, -3:] = -1
    main_len = torch.full((tokens,), idx.size(1) - 3, dtype=torch.int32)
    extra_len = torch.full((tokens,), extra_idx.size(1) - 3, dtype=torch.int32)
    sink = torch.randn(heads, generator=generator)
    scale = 512**-0.5
    args = dict(
        q=q.cuda().unsqueeze(1),
        k_cache=main,
        indices=idx.cuda().unsqueeze(1),
        topk_length=main_len.cuda(),
        attn_sink=sink.cuda(),
        head_dim_v=512,
        softmax_scale=scale,
        extra_k_cache=extra,
        extra_indices_in_kvcache=extra_idx.cuda().unsqueeze(1),
        extra_topk_length=extra_len.cuda(),
    )

    def launch():
        return flash_mla_with_kvcache_sm120(**args)[0].squeeze(1)

    out = launch().float().cpu()
    if tokens <= 64:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graphed = launch()
        graph.replay()
        torch.testing.assert_close(graphed.float().cpu(), out, atol=0, rtol=0)
        # Change the request's secondary rows at stable pointer addresses.
        args["extra_indices_in_kvcache"].copy_(extra_idx.roll(1, dims=-1).cuda().unsqueeze(1))
        graph.replay()
        changed = graphed.clone()
        torch.testing.assert_close(changed, launch(), atol=0, rtol=0)
        if torch.equal(changed.float().cpu(), out):
            raise AssertionError("CUDA graph replay ignored the new secondary rows")
    reference = []
    for i in range(tokens):
        kv = torch.cat((main_ref[idx[i, : main_len[i]].long()], extra_ref[extra_idx[i, : extra_len[i]].long()]))
        scores = q[i].float() @ kv.T * scale
        with_sink = torch.cat((scores, sink[:, None]), dim=1)
        reference.append(with_sink.softmax(-1)[:, :-1] @ kv)
    ref = torch.stack(reference)
    if not torch.isfinite(out).all():
        raise AssertionError("attention output contains nonfinite values")
    err = ((out - ref).norm() / ref.norm()).item()
    if err > 0.075:
        raise AssertionError(f"{tokens=} {extra_page=} relative L2 error {err:.5f}")
    return dict(
        tokens=tokens,
        heads=heads,
        main_page=256,
        extra_page=extra_page,
        compared_values=out.numel(),
        relative_l2=err,
        changed_input_graph=tokens <= 64,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (12, 0)
    generator = torch.Generator().manual_seed(1017)
    cases = [test_case(t, p, generator) for p in (128, 256) for t in (1, 6, 65)]
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        flashinfer=flashinfer.__version__,
        reference="Independent FP32 reference on BF16 queries, E4M3/E8M0 + BF16 decoded cache rows and sink",
        cases=cases,
    )
    print(json.dumps(result, indent=2))
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
