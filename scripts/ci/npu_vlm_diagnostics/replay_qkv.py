"""Replay captured first-layer QKV with verified TP shards and CPU references."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(value):
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def shards(model, rank, tp):
    import torch
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    results = []
    for suffix in ("weight", "bias"):
        parts = []
        for proj in ("q", "k", "v"):
            key = f"model.layers.0.self_attn.{proj}_proj.{suffix}"
            with safe_open(model / index[key], framework="pt", device="cpu") as handle:
                value = handle.get_tensor(key)
            if value.shape[0] % tp:
                raise ValueError(f"Unsupported replicated KV-head sharding: {key}")
            parts.append(value.chunk(tp, dim=0)[rank])
        results.append(torch.cat(parts, dim=0).contiguous())
    return results


def capture(source, rank):
    import torch

    records = [json.loads(p.read_text()) for p in (source / "snapshots").glob("*.json")]
    x_record = min(
        (
            d
            for d in records
            if d["rank"] == rank and d["stage"] == "qkv.operator.input"
        ),
        key=lambda d: d["time_ns"],
    )
    rid = x_record["rid"]
    y_record = next(
        d
        for d in records
        if d["rank"] == rank and d["rid"] == rid and d["stage"] == "qkv.operator.output"
    )
    component = next(
        d
        for d in records
        if d["rank"] == rank
        and d["rid"] == rid
        and d["stage"] == "decoder_component.input"
        and d["payload"]["module"] == "layers.0.self_attn.qkv_proj"
        and d["payload"]["frame"]["index"] == 0
    )
    xs = torch.load(
        source / "snapshots" / x_record["tensor_file"],
        map_location="cpu",
        weights_only=True,
    )
    ys = torch.load(
        source / "snapshots" / y_record["tensor_file"],
        map_location="cpu",
        weights_only=True,
    )
    x, y = xs["value.x"], ys["value.y"]
    if (
        digest(x) != x_record["payload"]["x"]["sha256"]
        or digest(y) != y_record["payload"]["y"]["sha256"]
    ):
        raise ValueError("Captured QKV tensor does not match its hash")
    return x, y, component["payload"]["parameters"], rid


def metrics(value, reference):
    import torch

    x, y = value.double(), reference.double()
    delta = x - y
    return {
        "finite": bool(torch.isfinite(x).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(y)
        ),
        "mismatch_fraction": float((x != y).double().mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acl-import", choices=("never", "before-torch", "after-init"), default="never"
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--cube-num", type=int)
    parser.add_argument("--vector-num", type=int)
    parser.add_argument("--profile-npu", action="store_true")
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()
    if args.acl_import == "before-torch":
        import acl  # noqa: F401
    import torch

    torch.set_num_threads(8)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    results = {"acl_import": args.acl_import, "ranks": {}}
    if not args.cpu_only:
        import torch_npu

        torch.npu.set_device(0)
        if args.cube_num is not None:
            torch_npu.npu.set_device_limit(
                0, cube_num=args.cube_num, vector_num=args.vector_num
            )
        results["device_limits"] = torch_npu.npu.get_device_limit(0)
        torch.npu.set_compile_mode(jit_compile=False)
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": "enable"})
        if args.acl_import == "after-init":
            import acl  # noqa: F401
    for rank in range(args.tp):
        x, captured, parameters, rid = capture(args.source_dir, rank)
        w, b = shards(args.model, rank, args.tp)
        for name, value in (("weight", w), ("bias", b)):
            if digest(value) != parameters[name]["sha256"]:
                raise ValueError(
                    f"Reconstructed TP{rank} {name} does not match loaded model"
                )
        product = x.double() @ w.double().t()
        exact = product + b.double()
        refs = {
            "fp64_fused_bf16": exact.to(torch.bfloat16),
            "fp64_matmul_bf16_then_bias": (
                product.to(torch.bfloat16).double() + b.double()
            ).to(torch.bfloat16),
            "fp32_fused_bf16": (x.float() @ w.float().t() + b.float()).to(
                torch.bfloat16
            ),
        }
        outputs = {"captured": captured}
        repeat_hashes = []
        profile = []
        if not args.cpu_only:
            xn, wn, bn = (v.to("npu:0") for v in (x, w, b))
            with torch.inference_mode():
                for _ in range(3):
                    y = torch.nn.functional.linear(xn, wn, bn)
                    repeat_hashes.append(digest(y))
                outputs["linear"] = y.cpu()
                outputs["mm_then_bias"] = (xn @ wn.t() + bn).cpu()
                outputs["npu_fp32_fused"] = (
                    (xn.float() @ wn.float().t() + bn.float()).to(torch.bfloat16).cpu()
                )
                if hasattr(torch_npu, "npu_linear"):
                    outputs["npu_linear"] = torch_npu.npu_linear(xn, wn, bn).cpu()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    torch.nn.functional.linear(xn, wn, bn).cpu()
                profile = [
                    {"name": e.key, "count": e.count} for e in prof.key_averages()
                ]
                if args.profile_npu:
                    with torch_npu.profiler.profile(
                        activities=[
                            torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU,
                        ],
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                            str(root / f"profile-rank-{rank}")
                        ),
                        record_shapes=True,
                    ):
                        torch.nn.functional.linear(xn, wn, bn).cpu()
        summary = {
            "source_rid": rid,
            "x_sha256": digest(x),
            "weight_sha256": digest(w),
            "bias_sha256": digest(b),
            "repeat_hashes": repeat_hashes,
            "outputs": {
                name: {
                    "sha256": digest(v),
                    "vs_fp64_unrounded": metrics(v, exact),
                    "vs_refs": {key: metrics(v, r) for key, r in refs.items()},
                    "vs_captured": metrics(v, captured),
                }
                for name, v in outputs.items()
            },
            "references": {name: digest(v) for name, v in refs.items()},
            "cpu_dispatch_events": profile,
        }
        results["ranks"][str(rank)] = summary
        torch.save(
            {**outputs, **refs, "fp64_unrounded": exact},
            root / f"outputs-rank-{rank}.pt",
        )
        (root / "result.json").write_text(json.dumps(results, indent=2))
        print("NPU_QKV_REPLAY " + json.dumps({"rank": rank, **summary}), flush=True)
    if not args.cpu_only:
        sys.path.insert(0, str(Path(__file__).parent / "hooks"))
        from npu_vlm_fingerprint import runtime_manifest

        results["runtime"] = runtime_manifest()
        results["device_properties"] = str(torch.npu.get_device_properties(0))
    (root / "result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
