"""Replay real flash-qlens/paged attention inputs with the deployed NPU APIs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--layers", default="0")
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--cube-num", type=int)
parser.add_argument("--vector-num", type=int)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch_npu.npu.set_device(0)
if args.cube_num is not None:
    torch_npu.npu.set_device_limit(
        0, cube_num=args.cube_num, vector_num=args.vector_num
    )
runtime = {
    "properties": str(torch_npu.npu.get_device_properties(0)),
    "limits": torch_npu.npu.get_device_limit(0),
}
(args.output / "runtime.json").write_text(json.dumps(runtime, indent=2))
print("runtime", json.dumps(runtime), flush=True)


def sha(t):
    return hashlib.sha256(
        t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def load(frame, layer, direction):
    path = (
        args.source
        / f"r{args.rank}-f{frame}-layers.{layer}.self_attn.attn-{direction}.pt"
    )
    return torch.load(path, weights_only=True, map_location="cpu")


results = []
for layer in map(int, args.layers.split(",")):
    k = v = None
    for frame in range(4):
        inp = load(frame, layer, "input")
        q = inp["args.0"].contiguous()
        ki, vi = inp["args.1"], inp["args.2"]
        k = ki if frame == 0 else torch.cat((k, ki))
        v = vi if frame == 0 else torch.cat((v, vi))
        length = k.shape[0]
        pages = (length + 127) // 128
        # Block zero is reserved, matching the model's paged allocator.
        kc = torch.zeros(pages + 1, 128, 1, 128, dtype=torch.bfloat16, device="npu")
        vc = torch.zeros_like(kc)
        slots = torch.arange(128, 128 + length, dtype=torch.int32, device="npu")
        torch_npu._npu_reshape_and_cache(
            key=k.reshape(-1, 1, 128).to("npu"),
            value=v.reshape(-1, 1, 128).to("npu"),
            key_cache=kc,
            value_cache=vc,
            slot_indices=slots,
        )
        block_table = torch.arange(
            1, pages + 1, dtype=torch.int32, device="npu"
        ).reshape(1, -1)
        seq = torch.tensor([length], dtype=torch.int32)
        qn = q.to("npu")
        mask = (
            torch.ones(128, 128, dtype=torch.bfloat16, device="cpu").triu(1).to("npu")
        )

        def compute():
            if frame == 0:
                out = torch.empty_like(qn)
                torch_npu._npu_flash_attention_qlens(
                    query=qn,
                    key_cache=kc,
                    value_cache=vc,
                    mask=mask,
                    block_table=block_table,
                    seq_len=torch.tensor([q.shape[0]], dtype=torch.int32),
                    context_lens=seq,
                    scale_value=128**-0.5,
                    num_heads=8,
                    num_kv_heads=1,
                    out=out,
                )
            else:
                query = qn.reshape(-1, 8, 128)
                out = torch.empty_like(query)
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=kc,
                    value_cache=vc,
                    num_heads=8,
                    num_kv_heads=1,
                    scale_value=128**-0.5,
                    block_table=block_table,
                    context_lens=seq,
                    out=out,
                )
            torch_npu.npu.synchronize()
            return out.reshape(-1, 1024).cpu()

        ys = [compute() for _ in range(3)]
        assert all(torch.equal(ys[0], y) for y in ys[1:])
        captured = load(frame, layer, "output")["result"]
        tail = ys[0][-1:]
        diff = tail.double() - captured.double()
        record = {
            "layer": layer,
            "frame": frame,
            "rank": args.rank,
            "repeats_identical": True,
            "replay_full_sha256": sha(ys[0]),
            "replay_full_shape": list(ys[0].shape),
            "captured_tail_sha256": sha(captured),
            "replay_tail_sha256": sha(tail),
            "tail_matches_capture": torch.equal(tail, captured),
            "different_elements": int((diff != 0).sum()),
            "max_abs": diff.abs().max().item(),
            "relative_l2": (diff.norm() / captured.double().norm()).item(),
            "full_query_shape": list(q.shape),
            "cache_shape": list(kc.shape),
        }
        torch.save(tail, args.output / f"layer-{layer}-frame-{frame}-tail.pt")
        results.append(record)
        print("ATTENTION_REPLAY", json.dumps(record), flush=True)
        if args.profile and frame == 0:
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(args.output / "profile")
                ),
                record_shapes=True,
            ):
                compute()
(args.output / "result.json").write_text(json.dumps(results, indent=2))
