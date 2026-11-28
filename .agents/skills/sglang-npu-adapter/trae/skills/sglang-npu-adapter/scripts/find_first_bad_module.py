#!/usr/bin/env python3
"""
find_first_bad_module.py -- Binary search over per-layer dumps to locate
the first layer where torch.allclose fails between HF reference and SGLang.

Inputs: two directories of layer_{i}.npz (each with key 'x').
Output: layer_diff.json (schema in references/layer_diff_protocol.md).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cosine_sim(a, b, eps=1e-12):
    a_flat = a.flatten()
    b_flat = b.flatten()
    na = np.linalg.norm(a_flat) + eps
    nb = np.linalg.norm(b_flat) + eps
    return float((a_flat @ b_flat) / (na * nb))


def allclose_np(a, b, rtol, atol):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def load_layer(layer_dir, idx):
    """Load layer dump npz. Returns array under key 'x'."""
    fp = Path(layer_dir) / f"layer_{idx}.npz"
    if not fp.exists():
        raise FileNotFoundError(f"missing {fp}")
    return np.load(fp)["x"]


def metrics(hf, sgl):
    diff = hf - sgl
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": cosine_sim(hf, sgl),
    }


def find_first_bad_binary(num_layers, hf_dir, sgl_dir, rtol, atol):
    """Return (first_bad_layer_idx, method).

    first_bad_layer_idx == num_layers means no bad layer found.
    """
    if num_layers == 0:
        return 0, "not_found"

    # Quick checks: layer 0 and last
    hf0 = load_layer(hf_dir, 0)
    sgl0 = load_layer(sgl_dir, 0)
    if not allclose_np(hf0, sgl0, rtol, atol):
        return 0, "binary_search"

    hf_last = load_layer(hf_dir, num_layers - 1)
    sgl_last = load_layer(sgl_dir, num_layers - 1)
    if allclose_np(hf_last, sgl_last, rtol, atol):
        return num_layers, "not_found"

    # Binary search [lo, hi]; lo passes, hi fails
    lo, hi = 0, num_layers - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        hf_m = load_layer(hf_dir, mid)
        sgl_m = load_layer(sgl_dir, mid)
        if allclose_np(hf_m, sgl_m, rtol, atol):
            lo = mid
        else:
            hi = mid
    return hi, "binary_search"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--sgl-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    sgl_dir = Path(args.sgl_dir)
    layer_files = sorted(hf_dir.glob("layer_*.npz"))
    num_layers = len(layer_files)
    if num_layers == 0:
        sys.stderr.write("error: no layer dumps found\n")
        sys.exit(2)

    first_bad, method = find_first_bad_binary(num_layers, hf_dir, sgl_dir, args.rtol, args.atol)

    # Compute per-layer metrics for full diagnostic
    layers_info = []
    for i in range(num_layers):
        hf = load_layer(hf_dir, i)
        sgl = load_layer(sgl_dir, i)
        m = metrics(hf, sgl)
        layers_info.append({
            "i": i,
            "max_abs": m["max_abs"],
            "cosine": m["cosine"],
            "passed": allclose_np(hf, sgl, args.rtol, args.atol),
            "non_deterministic": False,  # set by upstream NPU non-det detector
        })

    out = {
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "naming_strategy": "by_name",  # set by upstream dump scripts
        "naming_fallback_reason": None,
        "first_bad_layer": first_bad,
        "first_bad_layer_method": method,
        "layers": layers_info,
        "non_deterministic_layers": [],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[find_first_bad_module] first_bad_layer={first_bad} method={method}")


if __name__ == "__main__":
    main()
