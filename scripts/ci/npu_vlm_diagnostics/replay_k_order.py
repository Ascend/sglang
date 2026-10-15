"""Reconstruct frozen V2 outputs using equivalent simultaneous K permutations."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import torch
import torch_npu
from replay_supported_matmul import digest, info, read_captures, reconstruct_weight

sys.path.insert(0, str(Path(__file__).parent / "hooks"))
from npu_vlm_supported_matmul import compute, configure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
report = {"runtime": configure("gmm"), "chunk_size": 512, "ranks": {}}
captures = read_captures(args.source)
# Exhaust all four-chunk orders. Identical permutations of X and W preserve X @ W.T.
orders = list(itertools.permutations(range(4)))
for rank, capture in captures.items():
    x, reference = capture["x"], capture["y"]
    w = reconstruct_weight(args.model, rank)
    assert info(w)["sha256"] == capture["expected_weight"]["sha256"]
    xn, wn = x.npu(), w.npu()
    reconstructed = torch.empty_like(reference)
    best_diff = torch.full((len(x),), reference.shape[1] + 1, dtype=torch.int64)
    best_order = torch.zeros(len(x), dtype=torch.int64)
    row_counts = []
    variants = []
    for index, order in enumerate(orders):
        idx = torch.cat([torch.arange(i * 512, (i + 1) * 512) for i in order]).npu()
        y = compute(xn[:, idx].contiguous(), wn[:, idx].contiguous(), "gmm").cpu()
        torch.npu.synchronize()
        differences = (y != reference).sum(dim=1)
        improved = differences < best_diff
        reconstructed[improved] = y[improved]
        best_diff[improved] = differences[improved]
        best_order[improved] = index
        row_counts.append(differences.tolist())
        variants.append({"order": list(order), "sha256": digest(y), "different": int(differences.sum()), "row_blocks": {str(size): [{"rows": [lo, min(lo + size, len(x))], "different": int(differences[lo:lo + size].sum())} for lo in range(0, len(x), size)] for size in (512, 640)}})
    result = {"input": info(x), "weight": info(w), "reference": info(reference), "reconstructed": info(reconstructed), "remaining_differences": int(best_diff.sum()), "row_orders": best_order.tolist(), "differences_per_order_and_row": row_counts, "variants": variants}
    report["ranks"][str(rank)] = result
    print("NPU_VLM_K_ORDER", json.dumps({"rank": rank, "remaining_differences": result["remaining_differences"], "reference": result["reference"]["sha256"], "reconstructed": result["reconstructed"]["sha256"]}), flush=True)
    (args.output / "result.json").write_text(json.dumps(report, indent=2))
