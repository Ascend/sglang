"""Layer accumulation and FP64 last-query attention references from actual probes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument("--local", type=Path, required=True)
parser.add_argument("--ci", type=Path)
parser.add_argument("--local-label", choices=("88", "ci"), default="88")
parser.add_argument("--ci-precomputed", type=Path)
parser.add_argument("--ci-errors", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def load(root, rank, frame, module, direction):
    return torch.load(
        root / f"r{rank}-f{frame}-{module}-{direction}.pt",
        map_location="cpu",
        weights_only=True,
    )


def first(d):
    for key in ("result", "result.0"):
        if key in d:
            return d[key]
    raise KeyError(list(d))


def delta(a, b):
    a, b = a.double(), b.double()
    d = a - b
    return {
        "different": int((d != 0).sum()),
        "total": d.numel(),
        "max_abs": d.abs().max().item(),
        "mean_abs": d.abs().mean().item(),
        "relative_l2": (d.norm() / b.norm().clamp_min(1e-30)).item(),
    }


def digest(t):
    return hashlib.sha256(
        t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


sources = {args.local_label: args.local}
if args.ci:
    sources["ci"] = args.ci
result = {"attention_fp64": {}, "layer_deltas": [], "component_deltas": []}
references = {}
for label, root in sources.items():
    if label == "ci" and args.ci_precomputed:
        saved = torch.load(args.ci_precomputed, weights_only=True, map_location="cpu")
        for key, value in saved.items():
            references["ci", key] = value
        result["attention_fp64"]["ci"] = json.loads(args.ci_errors.read_text())[
            "attention_fp64"
        ]["ci"]
        continue
    rows = []
    for rank in (0, 1):
        for layer in range(36):
            keys = values = None
            for frame in range(4):
                inp = load(root, rank, frame, f"layers.{layer}.self_attn.attn", "input")
                q = inp.get("args.0", inp.get("kwargs.q")).reshape(-1, 8, 128)
                k = inp.get("args.1", inp.get("kwargs.k")).reshape(-1, 1, 128)
                v = inp.get("args.2", inp.get("kwargs.v")).reshape(-1, 1, 128)
                keys = k if frame == 0 else torch.cat((keys, k), dim=0)
                values = v if frame == 0 else torch.cat((values, v), dim=0)
                assert keys.shape[0] == 2526 + frame
                q_tail = q[-1].double()
                # GQA: each TP rank has 8 query heads sharing 1 KV head.
                scores = (q_tail @ keys[:, 0].double().T) * (128**-0.5)
                probabilities = torch.softmax(scores, dim=-1)
                ref = (probabilities @ values[:, 0].double()).reshape(1, 1024)
                actual = first(
                    load(root, rank, frame, f"layers.{layer}.self_attn.attn", "output")
                )
                rounded = ref.to(torch.bfloat16)
                key = (rank, frame, layer)
                references[label, key] = ref
                rows.append(
                    {
                        "rank": rank,
                        "frame": frame,
                        "layer": layer,
                        "reference_error": delta(actual, ref),
                        "rounded_reference_error": delta(actual, rounded),
                        "q_tail_sha": digest(q[-1:]),
                        "kv_sha": [digest(keys), digest(values)],
                        "actual_norm": actual.double().norm().item(),
                        "reference_norm": ref.norm().item(),
                    }
                )
    result["attention_fp64"][label] = rows
    worst = sorted(
        rows, key=lambda x: x["reference_error"]["relative_l2"], reverse=True
    )[:6]
    print("ATTENTION_WORST", label, json.dumps(worst), flush=True)
if args.ci:
    components = [
        "input_layernorm",
        "self_attn.qkv_proj",
        "self_attn.rotary_emb",
        "self_attn.attn",
        "self_attn.o_proj",
        "post_attention_layernorm",
        "mlp.gate_up_proj",
        "mlp.act_fn",
        "mlp.down_proj",
    ]
    for rank in (0, 1):
        for frame in range(4):
            for layer in range(36):
                name = f"layers.{layer}"
                a = load(args.local, rank, frame, name, "output")
                b = load(args.ci, rank, frame, name, "output")
                row = {
                    "rank": rank,
                    "frame": frame,
                    "layer": layer,
                    "mlp_output": delta(a["result.0"], b["result.0"]),
                    "residual": delta(a["result.1"], b["result.1"]),
                    "sum_before_next_norm": delta(
                        a["result.0"].double() + a["result.1"].double(),
                        b["result.0"].double() + b["result.1"].double(),
                    ),
                }
                result["layer_deltas"].append(row)
                for part in components:
                    module = name + "." + part
                    aa, bb = (
                        load(root, rank, frame, module, "output")
                        for root in (args.local, args.ci)
                    )
                    item = {
                        "rank": rank,
                        "frame": frame,
                        "layer": layer,
                        "component": part,
                        "output": delta(first(aa), first(bb)),
                    }
                    if part == "self_attn.attn":
                        ref_a = references["88", (rank, frame, layer)]
                        ref_b = references["ci", (rank, frame, layer)]
                        item["fp64_reference_difference"] = delta(ref_a, ref_b)
                        if frame == 0:
                            ai, bi = (
                                load(root, rank, frame, module, "input")
                                for root in (args.local, args.ci)
                            )
                            item["q_last_row"] = delta(
                                ai["args.0"][-1:], bi["args.0"][-1:]
                            )
                            if ai["args.1"].shape == bi["args.1"].shape:
                                item["full_k"] = delta(ai["args.1"], bi["args.1"])
                                item["full_v"] = delta(ai["args.2"], bi["args.2"])
                            else:
                                item["k_tail"] = delta(
                                    ai["args.1"][-1:], bi["args.1"][-1:]
                                )
                                item["v_tail"] = delta(
                                    ai["args.2"][-1:], bi["args.2"][-1:]
                                )
                    result["component_deltas"].append(item)
            aa, bb = (
                first(load(root, rank, frame, "final_norm", "output"))
                for root in (args.local, args.ci)
            )
            print("FINAL_NORM", rank, frame, json.dumps(delta(aa, bb)), flush=True)
args.output.write_text(json.dumps(result, indent=2))
torch.save(
    {
        key: value
        for (label, key), value in references.items()
        if label == args.local_label
    },
    args.output.with_suffix(".pt"),
)
