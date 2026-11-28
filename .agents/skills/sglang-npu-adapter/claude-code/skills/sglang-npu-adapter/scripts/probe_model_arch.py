#!/usr/bin/env python3
"""
probe_model_arch.py — Lightweight HF-config-based architecture probe.

Used by sglang-precision-rca standalone entry to fill output/output_summary.json
without running the full architecture-analyst sub-agent.

Outputs only the fields precision-rca actually consumes:
  - model_type, num_hidden_layers, hidden_size, num_attention_heads
  - dtype (bf16/fp16/fp32)
  - module_tree (default per family)
"""
import argparse
import json
import sys
from pathlib import Path


DTYPE_MAP = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


# Default module_tree per model family. Extended as new families are supported.
DEFAULT_MODULE_TREES = {
    "qwen2": [
        "model.layers.{i}.self_attn",
        "model.layers.{i}.self_attn.q_proj",
        "model.layers.{i}.self_attn.k_proj",
        "model.layers.{i}.self_attn.v_proj",
        "model.layers.{i}.self_attn.o_proj",
        "model.layers.{i}.mlp",
        "model.layers.{i}.mlp.gate_proj",
        "model.layers.{i}.mlp.up_proj",
        "model.layers.{i}.mlp.down_proj",
        "model.layers.{i}.input_layernorm",
        "model.layers.{i}.post_attention_layernorm",
    ],
    "qwen3": [
        # same shape as qwen2
        "model.layers.{i}.self_attn",
        "model.layers.{i}.self_attn.q_proj",
        "model.layers.{i}.self_attn.k_proj",
        "model.layers.{i}.self_attn.v_proj",
        "model.layers.{i}.self_attn.o_proj",
        "model.layers.{i}.mlp",
        "model.layers.{i}.input_layernorm",
        "model.layers.{i}.post_attention_layernorm",
    ],
    "llama": [
        "model.layers.{i}.self_attn",
        "model.layers.{i}.self_attn.q_proj",
        "model.layers.{i}.self_attn.k_proj",
        "model.layers.{i}.self_attn.v_proj",
        "model.layers.{i}.self_attn.o_proj",
        "model.layers.{i}.mlp",
        "model.layers.{i}.mlp.gate_proj",
        "model.layers.{i}.mlp.up_proj",
        "model.layers.{i}.mlp.down_proj",
        "model.layers.{i}.input_layernorm",
        "model.layers.{i}.post_attention_layernorm",
    ],
    "deepseek_v3": [
        "model.layers.{i}.self_attn",
        "model.layers.{i}.mlp",
        "model.layers.{i}.input_layernorm",
        "model.layers.{i}.post_attention_layernorm",
    ],
}

DEFAULT_MODULE_TREE = DEFAULT_MODULE_TREES["qwen2"]  # safe default


def normalize_dtype(raw):
    if raw is None:
        return "bf16"  # safe default
    return DTYPE_MAP.get(str(raw), "bf16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to HF model directory")
    ap.add_argument("--output", required=True, help="Output JSON path")
    args = ap.parse_args()

    cfg_path = Path(args.model) / "config.json"
    if not cfg_path.exists():
        sys.stderr.write(f"error: config.json not found at {cfg_path}\n")
        sys.exit(2)

    cfg = json.loads(cfg_path.read_text())

    model_type = cfg.get("model_type", "unknown")
    out = {
        "model_type": model_type,
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "hidden_size": cfg.get("hidden_size"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "dtype": normalize_dtype(cfg.get("torch_dtype")),
        "module_tree": DEFAULT_MODULE_TREES.get(model_type, DEFAULT_MODULE_TREE),
        "_source": "probe_model_arch.py",
        "_note": (
            "This is a lightweight subset of architecture-analyst output, "
            "produced by the standalone precision-rca entry."
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[probe_model_arch] wrote {args.output}")


if __name__ == "__main__":
    main()
