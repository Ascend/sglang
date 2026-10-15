"""Capture the first four Qwen forwards for cross-run layer/operator replay."""

import contextvars
import hashlib
import json
import os
from pathlib import Path

_INSTALLED = set()


def install(owner, frame_context):
    if id(owner) in _INSTALLED:
        return
    import torch

    root = Path(os.environ["VLM_DIAGNOSTICS_DIR"]).parent / "layer-probes"
    root.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank()
    active_layer = contextvars.ContextVar("vlm_probe_layer", default=None)
    layer_tokens = {}

    def capture(module, name, direction, args, kwargs, result=None):
        frame = frame_context.get()
        if frame is None or frame["index"] > 3:
            return
        tensors = {}
        shapes = {}

        def select(value, key):
            if isinstance(value, torch.Tensor):
                shapes[key] = list(value.shape)
                # Preserve full Q/K/V at the attention boundary for exact-shape replay.
                # Other prefill activations retain the last token; decode retains all.
                if frame["index"] == 0 and not (
                    name.endswith("self_attn.attn") and direction == "input"
                ):
                    if value.ndim and value.shape[0] > 1:
                        value = value[-1:]
                tensors[key] = value.detach().cpu().contiguous().clone()
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    select(item, f"{key}.{index}")
            elif isinstance(value, dict):
                for label, item in value.items():
                    select(item, f"{key}.{label}")

        if direction == "input":
            select(args, "args")
            select(kwargs, "kwargs")
        else:
            select(result, "result")
        filename = f"r{rank}-f{frame['index']}-{name}-{direction}.pt"
        torch.save(tensors, root / filename)
        metadata = {
            "rank": rank,
            "frame": frame,
            "module": name,
            "direction": direction,
            "module_type": type(module).__name__,
            "file": filename,
            "original_shapes": shapes,
            "tensors": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(
                        value.reshape(-1).view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                for key, value in tensors.items()
            },
        }
        if direction == "input" and name.endswith("self_attn.attn"):
            batch = kwargs.get("forward_batch") or (args[3] if len(args) > 3 else None)
            if batch is not None:
                from sglang.srt.layers.radix_attention import get_attn_backend

                backend = get_attn_backend()
                metadata["backend"] = type(backend).__name__
                metadata["backend_options"] = {
                    field: getattr(backend, field, None)
                    for field in (
                        "use_fia",
                        "use_fa",
                        "graph_mode",
                        "page_size",
                        "use_alibi",
                    )
                }
                metadata["forward_mode"] = str(batch.forward_mode)
        with (root / f"rank-{rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    def observe(module, name):
        is_layer = name.startswith("layers.") and name.count(".") == 1
        layer_index = int(name.split(".")[1]) if name.startswith("layers.") else None

        def before(module, args, kwargs):
            if is_layer:
                layer_tokens[name] = active_layer.set(layer_index)
            elif active_layer.get() is not None and layer_index != active_layer.get():
                return
            try:
                capture(module, name, "input", args, kwargs)
            except Exception as error:
                print(f"VLM_LAYER_PROBE_ERROR {name}.input: {error!r}", flush=True)

        def after(module, args, kwargs, result):
            if (
                not is_layer
                and active_layer.get() is not None
                and layer_index != active_layer.get()
            ):
                return
            try:
                capture(module, name, "output", args, kwargs, result)
            except Exception as error:
                print(f"VLM_LAYER_PROBE_ERROR {name}.output: {error!r}", flush=True)
            finally:
                if is_layer and name in layer_tokens:
                    active_layer.reset(layer_tokens.pop(name))

        module.register_forward_pre_hook(before, with_kwargs=True)
        module.register_forward_hook(after, with_kwargs=True)

    names = (
        "input_layernorm",
        "self_attn.qkv_proj",
        "self_attn.rotary_emb",
        "self_attn.attn",
        "self_attn.o_proj",
        "post_attention_layernorm",
        "mlp.gate_up_proj",
        "mlp.act_fn",
        "mlp.down_proj",
    )
    for index, layer in enumerate(owner.model.layers):
        observe(layer, f"layers.{index}")
        for name in names:
            module = layer
            for part in name.split("."):
                module = getattr(module, part, None)
            if module is not None and hasattr(module, "register_forward_hook"):
                observe(module, f"layers.{index}.{name}")
    observe(owner.model.norm, "final_norm")
    _INSTALLED.add(id(owner))
