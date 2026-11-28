#!/usr/bin/env python3
"""
dump_sgl_layer_outputs.py — SGLang model standalone forward + per-layer hooks.

Loads the SGLang model directly (not via server) using the same code paths
as the production server. Attaches hooks at each transformer block, runs
forward on the same prompts, dumps last-token hidden state.

Output structure mirrors dump_hf_layer_outputs.py for find_first_bad_module.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_sglang_model(model_path, dtype_str, tp_size=1, quant=None):
    """Load SGLang model standalone.

    Uses sglang.srt.model_executor pipeline minus the scheduler/server layers.
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        sys.stderr.write("error: torch_npu not available\n")
        sys.exit(2)

    # Import lazily to avoid pulling sglang's full server graph
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader.loader import get_model_loader
    from sglang.srt.model_loader.weight_utils import default_weight_loader  # noqa: F401

    dtype = DTYPE_MAP[dtype_str]

    # Build a minimal model_config; SGLang's loader takes care of architecture dispatch
    model_config = ModelConfig(
        path=model_path,
        trust_remote_code=True,
        dtype=dtype_str,
        quantization=quant,
    )

    # Use SGLang's model loader — same code path as server uses
    from sglang.srt.model_loader import get_model
    model = get_model(model_config=model_config, load_config=None, device_config=None)
    model = model.to("npu").eval()
    return model, model_config


def install_hooks(model, num_layers, store):
    handles = []
    for i in range(num_layers):
        # SGLang model module path (sometimes differs from HF)
        layer = None
        for prefix in ("model.layers", "model.decoder.layers"):
            try:
                cur = model
                for part in prefix.split("."):
                    cur = getattr(cur, part)
                layer = cur[i]
                break
            except AttributeError:
                continue
        if layer is None:
            raise RuntimeError(f"could not locate SGLang layer {i}")

        def make_hook(idx):
            def hook(module, inputs, outputs):
                out = outputs[0] if isinstance(outputs, tuple) else outputs
                last_tok = out[:, -1, :].detach().to(torch.float32).cpu().numpy()
                store.setdefault(idx, []).append(last_tok)
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))
    return handles


def run_one_pass(model, model_config, prompts, max_length=2048):
    """Tokenize + forward through SGLang model. Tokenizer comes from model_config.path."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.path, trust_remote_code=True)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        ids = inputs.input_ids.to("npu")
        with torch.no_grad():
            # SGLang's standalone forward signature varies; the canonical entry is
            # model(input_ids, positions, ...) — adjust per version.
            positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            model(input_ids=ids, positions=positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--quant", default=None, help="None / fp8 / w8a8 — match server config")
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    p = Path(args.prompts)
    if p.suffix == ".json":
        prompts = json.loads(p.read_text())
    else:
        prompts = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_config = load_sglang_model(args.model, args.dtype, quant=args.quant)
    num_layers = args.num_layers or model.config.num_hidden_layers

    runs = []
    for run_i in range(args.runs):
        store = {}
        handles = install_hooks(model, num_layers, store)
        try:
            run_one_pass(model, model_config, prompts)
        finally:
            for h in handles:
                h.remove()
        runs.append(store)

    for i in range(num_layers):
        arrs = {}
        for run_i, store in enumerate(runs):
            stacked = np.concatenate(store[i], axis=0)
            arrs[f"x_run{run_i}"] = stacked
        arrs["x"] = arrs["x_run0"]
        np.savez(out_dir / f"layer_{i}.npz", **arrs)

    del model
    import gc
    gc.collect()
    if hasattr(torch, "npu"):
        torch.npu.empty_cache()
    print(f"[dump_sgl] wrote {num_layers} layers to {out_dir}")


if __name__ == "__main__":
    main()
