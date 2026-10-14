"""Replay full prefill gate/up projection inputs with verified TP2 weights.

This experiment uses F.linear without a bias, preserving M=2526/K=2048/N=11008.
The saved outputs can be compared across machines before computing FP64 dot
products for the coordinates that differ. Optional FP64 references are CPU-only.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def info(value):
    return {
        "sha256": digest(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
    }


def metrics(value, reference):
    import torch

    if value.shape != reference.shape:
        raise ValueError(f"Different shapes: {value.shape}, {reference.shape}")
    x, y = value.double(), reference.double()
    delta = x - y
    indices = torch.nonzero(x != y, as_tuple=False)
    norm, delta_norm = y.norm().item(), delta.norm().item()
    return {
        "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
        "different": len(indices),
        "total": value.numel(),
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": delta_norm / norm
        if norm
        else (0.0 if delta_norm == 0 else None),
        "first_differences": [
            {
                "index": index,
                "value": x[tuple(index)].item(),
                "reference": y[tuple(index)].item(),
            }
            for index in indices[:64].tolist()
        ],
        "difference_list_truncated": len(indices) > 64,
    }


def read_captures(source):
    import torch

    snapshots = source if source.name == "snapshots" else source / "snapshots"
    records = [json.loads(path.read_text()) for path in snapshots.glob("*.json")]
    captures = {}
    for rank in (0, 1):
        inputs = [
            r
            for r in records
            if r.get("rank") == rank and r["stage"] == "gate_up.operator.input"
        ]
        if not inputs:
            raise ValueError(
                f"No gate_up.operator.input snapshot for rank {rank}: {snapshots}"
            )
        x_record = min(inputs, key=lambda r: r["time_ns"])
        outputs = [
            r
            for r in records
            if r.get("rank") == rank
            and r["stage"] == "gate_up.operator.output"
            and r["rid"] == x_record["rid"]
            and r["pid"] == x_record["pid"]
            and r["payload"]["frame"] == x_record["payload"]["frame"]
        ]
        if len(outputs) != 1:
            raise ValueError(
                f"Expected one matching gate/up output, rank {rank}, got {len(outputs)}"
            )
        y_record = outputs[0]
        weight_records = [
            r
            for r in records
            if r.get("rank") == rank
            and r["stage"] == "decoder.loaded_weights"
            and r["pid"] == x_record["pid"]
        ]
        if len(weight_records) != 1:
            raise ValueError(f"Expected one loaded-weight manifest for rank {rank}")
        parameter = "model.layers.0.mlp.gate_up_proj.weight"
        manifest = weight_records[0]["payload"]
        if "model.layers.0.mlp.gate_up_proj.bias" in manifest:
            raise ValueError(
                "This no-bias gate/up replay does not support a loaded bias"
            )
        tensors = {}
        for label, record in (("x", x_record), ("y", y_record)):
            values = torch.load(
                snapshots / record["tensor_file"], weights_only=True, map_location="cpu"
            )
            value = values[f"value.{label}"]
            expected = record["payload"][label]
            actual = info(value)
            if any(
                actual[field] != expected[field]
                for field in ("sha256", "shape", "dtype")
            ):
                raise ValueError(f"Captured rank {rank} {label} metadata mismatch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Captured rank {rank} {label} has nonfinite values")
            tensors[label] = value
        if tuple(tensors["x"].shape) != (2526, 2048) or tuple(tensors["y"].shape) != (
            2526,
            11008,
        ):
            raise ValueError(
                "Expected full prefill X[2526,2048], Y[2526,11008]; a tail-only capture is invalid"
            )
        if any(value.dtype != torch.bfloat16 for value in tensors.values()):
            raise ValueError("Both gate/up capture tensors must be BF16")
        captures[rank] = {
            **tensors,
            "expected_weight": manifest[parameter],
            "rid": x_record["rid"],
            "pid": x_record["pid"],
            "source_tensor_files": {
                "x": x_record["tensor_file"],
                "y": y_record["tensor_file"],
            },
        }
    return captures


def reconstruct_weight(model, rank):
    import torch
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    parts = []
    for projection in ("gate", "up"):
        key = f"model.layers.0.mlp.{projection}_proj.weight"
        with safe_open(model / index[key], framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(key)
        if tuple(weight.shape) != (11008, 2048) or weight.dtype != torch.bfloat16:
            raise ValueError(
                f"Unexpected checkpoint gate/up weight: {key}: {info(weight)}"
            )
        parts.append(weight.chunk(2, dim=0)[rank])
    return torch.cat(parts, dim=0).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        "--source-dir",
        dest="source",
        type=Path,
        required=True,
        help="Capture snapshots directory or its parent",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output", "--output-dir", dest="output", type=Path, required=True
    )
    parser.add_argument("--cube-num", type=int)
    parser.add_argument("--vector-num", type=int)
    parser.add_argument(
        "--profile", "--profile-npu", dest="profile", action="store_true"
    )
    parser.add_argument(
        "--cpu-reference", choices=("none", "last", "all"), default="none"
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="Fail calibration when same-environment replay differs from capture",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Also save full replay Y when it exactly matches the source capture",
    )
    args = parser.parse_args()
    if args.cpu_only and args.cpu_reference == "none":
        parser.error("--cpu-only requires --cpu-reference last or all")
    if args.cpu_only and (
        args.profile or args.cube_num is not None or args.vector_num is not None
    ):
        parser.error(
            "CPU-only references cannot set NPU core limits or profile NPU kernels"
        )
    if args.cpu_only and args.require_match:
        parser.error("--require-match requires NPU replay")
    import torch

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    captures = read_captures(args.source)
    runtime = {
        "pid": os.getpid(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cpu_only": args.cpu_only,
    }
    if not args.cpu_only:
        import torch_npu

        torch.npu.set_device(0)
        before = torch_npu.npu.get_device_limit(0)
        if args.cube_num is not None or args.vector_num is not None:
            torch_npu.npu.set_device_limit(
                0,
                cube_num=args.cube_num if args.cube_num is not None else -1,
                vector_num=args.vector_num if args.vector_num is not None else -1,
            )
        actual = torch_npu.npu.get_device_limit(0)
        for field, requested in (
            ("cube_core_num", args.cube_num),
            ("vector_core_num", args.vector_num),
        ):
            expected = before[field] if requested is None else requested
            if actual[field] != expected:
                raise RuntimeError(
                    f"Core limit not applied: {field}, expected {expected}, observed {actual[field]}"
                )
        torch.npu.set_compile_mode(jit_compile=False)
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": "enable"})
        runtime.update(
            {
                "torch_npu": torch_npu.__version__,
                "properties": str(torch.npu.get_device_properties(0)),
                "before_limits": before,
                "actual_limits": actual,
                "jit_compile": False,
                "ALLOW_INTERNAL_FORMAT": "enable",
            }
        )
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print("GATE_UP_RUNTIME", json.dumps(runtime), flush=True)
    results = {
        "runtime": runtime,
        "source": str(args.source),
        "model": str(args.model),
        "ranks": {},
    }
    validation_failures = []
    for rank, capture in captures.items():
        x, captured = capture["x"], capture["y"]
        weight = reconstruct_weight(args.model, rank)
        parameter_info = info(weight)
        if any(
            parameter_info[field] != capture["expected_weight"][field]
            for field in ("sha256", "shape", "dtype")
        ):
            raise ValueError(
                f"Reconstructed rank {rank} gate/up weight differs from loaded model manifest"
            )
        summary = {
            "source_rid": capture["rid"],
            "source_pid": capture["pid"],
            "source_tensor_files": capture["source_tensor_files"],
            "input": info(x),
            "weight": parameter_info,
            "captured": info(captured),
            "captured_tail": info(captured[-1:]),
            "weight_manifest_verified": True,
            "operation": "F.linear, no bias, BF16 M2526 K2048 N11008",
        }
        saved = {}
        if not args.cpu_only:
            xn, wn = x.to("npu:0"), weight.to("npu:0")
            summary["device_layout"] = {
                name: {
                    "device": str(tensor.device),
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "storage_offset": tensor.storage_offset(),
                    "npu_format": int(torch_npu.get_npu_format(tensor)),
                }
                for name, tensor in (("x", xn), ("weight", wn))
            }

            def compute(xn=xn, wn=wn):
                with torch.inference_mode():
                    value = torch.nn.functional.linear(xn, wn)
                    torch.npu.synchronize()
                    return value.cpu()

            hashes = []
            for _ in range(3):
                replayed = compute()
                hashes.append(digest(replayed))
            saved["linear"] = replayed
            summary.update(
                {
                    "replay": info(replayed),
                    "replay_tail": info(replayed[-1:]),
                    "repeat_hashes": hashes,
                    "repeats_identical": len(set(hashes)) == 1,
                    "vs_captured": metrics(replayed, captured),
                    "tail_vs_captured": metrics(replayed[-1:], captured[-1:]),
                }
            )
            if args.profile:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
                ) as prof:
                    cpu_profiled = compute()
                summary["cpu_dispatch_events"] = [
                    {
                        "name": event.key,
                        "count": event.count,
                        "input_shapes": event.input_shapes,
                        "cpu_time_total_us": event.cpu_time_total,
                    }
                    for event in prof.key_averages(group_by_input_shape=True)
                ]
                profile_root = args.output / f"profile-rank-{rank}"
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(profile_root)
                    ),
                    record_shapes=True,
                ):
                    npu_profiled = compute()
                summary["profile_hashes"] = {
                    "cpu": digest(cpu_profiled),
                    "npu": digest(npu_profiled),
                }
                summary["profile_matches_unprofiled"] = all(
                    value == hashes[0] for value in summary["profile_hashes"].values()
                )
                summary["kernel_tables"] = []
                for path in sorted(profile_root.rglob("kernel_details.csv")):
                    with path.open(newline="") as handle:
                        kernel_rows = list(csv.DictReader(handle))
                    summary["kernel_tables"].append(
                        {
                            "file": str(path.relative_to(args.output)),
                            "rows": kernel_rows,
                        }
                    )
        if args.cpu_reference != "none":
            indices = (
                torch.arange(x.shape[0])
                if args.cpu_reference == "all"
                else torch.tensor([x.shape[0] - 1])
            )
            exact = x[indices].double() @ weight.double().t()
            rounded = exact.to(torch.bfloat16)
            saved.update(
                reference_row_indices=indices,
                fp64_unrounded=exact,
                fp64_rounded_bf16=rounded,
            )
            summary["cpu_reference"] = {
                "rows": args.cpu_reference,
                "fp64_unrounded": info(exact),
                "fp64_rounded_bf16": info(rounded),
                "captured_vs_fp64": metrics(captured[indices], exact),
                "captured_vs_fp64_rounded": metrics(captured[indices], rounded),
            }
            if "linear" in saved:
                summary["cpu_reference"]["replay_vs_fp64"] = metrics(
                    saved["linear"][indices], exact
                )
                summary["cpu_reference"]["replay_vs_fp64_rounded"] = metrics(
                    saved["linear"][indices], rounded
                )
        output_file = args.output / f"outputs-rank-{rank}.pt"
        tensors_to_save = {
            name: value
            for name, value in saved.items()
            if name != "linear"
            or args.save_output
            or summary["vs_captured"]["different"] != 0
        }
        if tensors_to_save:
            torch.save(tensors_to_save, output_file)
            summary["tensor_file"] = output_file.name
        else:
            summary["tensor_file"] = None
            summary["replay_reuses_exact_source_capture"] = True
        results["ranks"][str(rank)] = summary
        if not args.cpu_only:
            if not summary["repeats_identical"]:
                validation_failures.append(f"rank {rank}: repeated outputs differ")
            if args.profile and not summary["profile_matches_unprofiled"]:
                validation_failures.append(f"rank {rank}: profiling changed output")
            if args.profile and not any(
                table["rows"] for table in summary["kernel_tables"]
            ):
                validation_failures.append(
                    f"rank {rank}: profiler kernel tables missing"
                )
            if args.require_match and summary["vs_captured"]["different"]:
                validation_failures.append(
                    f"rank {rank}: replay differs from source capture"
                )
        results["validation"] = {
            "require_match": args.require_match,
            "failures": list(validation_failures),
        }
        (args.output / "result.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )
        print(
            "GATE_UP_REPLAY",
            json.dumps({"rank": rank, **summary}, allow_nan=False),
            flush=True,
        )
    if validation_failures:
        raise RuntimeError(
            "Gate/up replay validation failed: " + "; ".join(validation_failures)
        )


if __name__ == "__main__":
    main()
