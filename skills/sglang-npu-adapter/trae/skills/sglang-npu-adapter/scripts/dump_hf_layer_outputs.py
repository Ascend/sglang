#!/usr/bin/env python3
"""
dump_hf_layer_outputs.py — HF NPU eager forward + per-layer hooks.

Loads HF transformers model, runs forward on failing_prompts, saves last-token
hidden state per transformer block to {output_dir}/layer_{i}.npz.

Each prompt is run twice for non-determinism detection; both runs saved as
'x_run0' and 'x_run1' arrays.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_model(model_path, dtype_str):
    try:
        import torch_npu  # noqa: F401 — registers npu device
    except ImportError:
        sys.stderr.write("error: torch_npu not available; this script requires NPU env\n")
        sys.exit(2)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = DTYPE_MAP[dtype_str]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to("npu").eval()
    return model, tokenizer


def install_hooks(model, num_layers, store):
    """Install forward hook on each model.layers.{i}, dumping last-token hidden state."""
    handles = []
    for i in range(num_layers):
        # Try common module path
        layer = None
        for prefix in ("model.layers", "transformer.h", "model.decoder.layers"):
            try:
                cur = model
                for part in prefix.split("."):
                    cur = getattr(cur, part)
                layer = cur[i]
                break
            except AttributeError:
                continue
        if layer is None:
            raise RuntimeError(f"could not locate layer {i}; supported prefixes: model.layers, transformer.h, model.decoder.layers")

        def make_hook(idx):
            def hook(module, inputs, outputs):
                out = outputs[0] if isinstance(outputs, tuple) else outputs
                last_tok = out[:, -1, :].detach().to(torch.float32).cpu().numpy()
                store.setdefault(idx, []).append(last_tok)
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))
    return handles


def run_one_pass(model, tokenizer, prompts, max_length=2048):
    """Run all prompts through one forward pass; hooks accumulate per-layer outputs."""
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to("npu")
        with torch.no_grad():
            model(**inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="Path to prompts JSON array or text file (one per line)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    ap.add_argument("--num-layers", type=int, default=None,
                    help="Override layer count (auto-detect if unset)")
    ap.add_argument("--runs", type=int, default=2, help="Number of independent runs for non-det detection")
    args = ap.parse_args()

    # Load prompts
    p = Path(args.prompts)
    if p.suffix == ".json":
        prompts = json.loads(p.read_text())
    else:
        prompts = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.dtype)
    num_layers = args.num_layers or model.config.num_hidden_layers

    # Run twice
    runs = []
    for run_i in range(args.runs):
        store = {}
        handles = install_hooks(model, num_layers, store)
        try:
            run_one_pass(model, tokenizer, prompts)
        finally:
            for h in handles:
                h.remove()
        runs.append(store)

    # Save per-layer npz with one array per run
    for i in range(num_layers):
        arrs = {}
        for run_i, store in enumerate(runs):
            stacked = np.concatenate(store[i], axis=0)  # [n_prompts, hidden]
            arrs[f"x_run{run_i}"] = stacked
        # Default 'x' key = run 0 for compatibility with find_first_bad_module.py
        arrs["x"] = arrs["x_run0"]
        np.savez(out_dir / f"layer_{i}.npz", **arrs)

    # Free
    del model
    import gc
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    print(f"[dump_hf] wrote {num_layers} layers to {out_dir}")


if __name__ == "__main__":
    main()
