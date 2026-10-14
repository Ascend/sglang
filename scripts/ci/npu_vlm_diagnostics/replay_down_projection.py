"""Replay captured TP2 BF16 down_proj local matmuls and a two-product sum.

This single-NPU experiment does not execute HCCL. Run the same script and captured
inputs on CI and 88 to compare the two local products before investigating the
collective. All tensors remain full single-token M=1, K=5504, N=2048 operands.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open


def sha(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def info(value):
    return {
        "sha256": sha(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
    }


def compare(value, reference):
    if value.shape != reference.shape:
        raise ValueError(
            f"Comparison shape mismatch {value.shape} vs {reference.shape}"
        )
    x, y = value.double(), reference.double()
    delta = x - y
    locations = torch.nonzero(x != y, as_tuple=False).tolist()
    denominator = y.norm().item()
    numerator = delta.norm().item()
    return {
        "different": len(locations),
        "total": value.numel(),
        "max_abs": delta.abs().max().item(),
        "relative_l2": numerator / denominator
        if denominator
        else (0.0 if numerator == 0 else None),
        "differences": [
            {
                "index": index,
                "value": x[tuple(index)].item(),
                "reference": y[tuple(index)].item(),
                "delta": delta[tuple(index)].item(),
            }
            for index in locations
        ],
    }


def load_capture(source, rank, layer, frame, direction):
    name = f"r{rank}-f{frame}-layers.{layer}.mlp.down_proj-{direction}.pt"
    values = torch.load(source / name, map_location="cpu", weights_only=True)
    if direction == "input":
        result = values.get("args.0", values.get("kwargs.x"))
        expected_shape = (1, 5504)
    else:
        key = "result" if "result" in values else "result.0"
        if set(values) != {key}:
            raise ValueError(
                f"Unexpected down_proj output tensors: {name}: {list(values)}"
            )
        result = values[key]
        expected_shape = (1, 2048)
    if (
        result is None
        or tuple(result.shape) != expected_shape
        or result.dtype != torch.bfloat16
    ):
        raise ValueError(f"Expected complete BF16 {expected_shape} in {name}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"Nonfinite capture: {name}")
    metadata_path = source / f"rank-{rank}.jsonl"
    metadata_verified = False
    if metadata_path.exists():
        records = [json.loads(line) for line in metadata_path.read_text().splitlines()]
        records = [record for record in records if record["file"] == name]
        if len(records) != 1:
            raise ValueError(
                f"Expected one metadata record for {name}, got {len(records)}"
            )
        expected = records[0]["tensors"]
        if set(expected) != set(values):
            raise ValueError(f"Capture metadata tensor keys differ: {name}")
        for key, value in values.items():
            actual = info(value)
            if any(
                actual[field] != expected[key][field]
                for field in ("sha256", "shape", "dtype")
            ):
                raise ValueError(f"Capture metadata differs: {name}:{key}")
        metadata_verified = True
    return result.contiguous(), {
        "file": name,
        "metadata_verified": metadata_verified,
        **info(result),
    }


def parse_numbers(value):
    numbers = list(dict.fromkeys(int(part) for part in value.split(",")))
    if not numbers:
        raise ValueError("An empty layer/frame selection is not supported")
    return numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path("/mnt/paas/weights/Qwen2.5-VL-3B-Instruct")
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--layers", default="5", help="Comma-separated zero-based decoder layer indices"
    )
    parser.add_argument(
        "--frames", default="2", help="Comma-separated decode forward indices 1,2,3"
    )
    parser.add_argument("--cube-num", type=int)
    parser.add_argument("--vector-num", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    layers, frames = parse_numbers(args.layers), parse_numbers(args.frames)
    if any(layer not in range(36) for layer in layers) or any(
        frame not in (1, 2, 3) for frame in frames
    ):
        raise ValueError(
            "Only decoder layers 0..35 and complete decode frames 1..3 are supported"
        )
    if args.repeats < 2:
        raise ValueError("At least two repeats are required")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    if args.cube_num is not None or args.vector_num is not None:
        torch_npu.npu.set_device_limit(
            0,
            cube_num=args.cube_num if args.cube_num is not None else -1,
            vector_num=args.vector_num if args.vector_num is not None else -1,
        )
    torch.npu.set_compile_mode(jit_compile=False)
    torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": "enable"})
    runtime = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "properties": str(torch.npu.get_device_properties(0)),
        "limits": torch_npu.npu.get_device_limit(0),
        "requested_limits": {"cube_num": args.cube_num, "vector_num": args.vector_num},
        "jit_compile": False,
        "ALLOW_INTERNAL_FORMAT": "enable",
        "cpu_threads": torch.get_num_threads(),
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print("DOWN_PROJECTION_RUNTIME", json.dumps(runtime), flush=True)
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    records = []
    for layer in layers:
        key = f"model.layers.{layer}.mlp.down_proj.weight"
        with safe_open(
            args.model / weight_map[key], framework="pt", device="cpu"
        ) as handle:
            global_weight = handle.get_tensor(key)
        if (
            tuple(global_weight.shape) != (2048, 11008)
            or global_weight.dtype != torch.bfloat16
        ):
            raise ValueError(f"Unsupported weight: {key}: {info(global_weight)}")
        weights = {
            rank: global_weight.chunk(2, dim=1)[rank].contiguous() for rank in (0, 1)
        }
        weight_npu = {rank: value.to("npu:0") for rank, value in weights.items()}
        for frame in frames:
            xs, captured, capture_info = {}, {}, {}
            for rank in (0, 1):
                xs[rank], x_info = load_capture(
                    args.source, rank, layer, frame, "input"
                )
                captured[rank], y_info = load_capture(
                    args.source, rank, layer, frame, "output"
                )
                capture_info[str(rank)] = {
                    "input": x_info,
                    "output": y_info,
                    "weight": info(weights[rank]),
                }
            x_npu = {rank: value.to("npu:0") for rank, value in xs.items()}

            def compute():
                values = {}
                with torch.inference_mode():
                    for rank in (0, 1):
                        with torch.profiler.record_function(f"down_proj.rank_{rank}"):
                            values[f"product_rank{rank}"] = torch.nn.functional.linear(
                                x_npu[rank], weight_npu[rank]
                            )
                    with torch.profiler.record_function("down_proj.tp2_sum_emulation"):
                        values["sum_npu_bf16"] = (
                            values["product_rank0"] + values["product_rank1"]
                        )
                    torch.npu.synchronize()
                    return {name: value.cpu() for name, value in values.items()}

            repetitions = [compute() for _ in range(args.repeats)]
            hashes = [
                {name: sha(value) for name, value in values.items()}
                for values in repetitions
            ]
            exact = {
                rank: xs[rank].double() @ weights[rank].double().t() for rank in (0, 1)
            }
            rounded_products = {
                rank: value.to(torch.bfloat16) for rank, value in exact.items()
            }
            references = {
                "product_rank0_fp64": exact[0],
                "product_rank1_fp64": exact[1],
                "product_rank0_fp64_rounded_bf16": rounded_products[0],
                "product_rank1_fp64_rounded_bf16": rounded_products[1],
                "sum_fp64_unrounded": exact[0] + exact[1],
                "sum_fp64_once_rounded_bf16": (exact[0] + exact[1]).to(torch.bfloat16),
                "sum_fp64_products_rounded_then_sum_rounded_bf16": (
                    rounded_products[0].double() + rounded_products[1].double()
                ).to(torch.bfloat16),
                "sum_actual_products_cpu_bf16": (
                    repetitions[0]["product_rank0"].double()
                    + repetitions[0]["product_rank1"].double()
                ).to(torch.bfloat16),
            }
            values = repetitions[0]
            comparisons = {}
            for rank in (0, 1):
                comparisons[f"product_rank{rank}_vs_fp64_rounded"] = compare(
                    values[f"product_rank{rank}"], rounded_products[rank]
                )
                comparisons[f"sum_vs_captured_rank{rank}"] = compare(
                    values["sum_npu_bf16"], captured[rank]
                )
                comparisons[f"captured_rank{rank}_vs_fp64_two_round"] = compare(
                    captured[rank],
                    references["sum_fp64_products_rounded_then_sum_rounded_bf16"],
                )
            comparisons["sum_vs_actual_products_cpu_bf16"] = compare(
                values["sum_npu_bf16"], references["sum_actual_products_cpu_bf16"]
            )
            comparisons["sum_vs_fp64_two_round"] = compare(
                values["sum_npu_bf16"],
                references["sum_fp64_products_rounded_then_sum_rounded_bf16"],
            )
            comparisons["sum_vs_fp64_once_round"] = compare(
                values["sum_npu_bf16"], references["sum_fp64_once_rounded_bf16"]
            )
            comparisons["captured_rank0_vs_rank1"] = compare(captured[0], captured[1])
            record = {
                "layer": layer,
                "frame": frame,
                "operation": "torch.nn.functional.linear BF16 M=1 K=5504 N=2048, no bias",
                "reduction": "single-device BF16 add of the two local products; HCCL is not executed",
                "source": str(args.source),
                "parameter": key,
                "capture": capture_info,
                "repeat_hashes": hashes,
                "repeats_identical": all(item == hashes[0] for item in hashes[1:]),
                "tensors": {
                    name: info(value)
                    for name, value in {**values, **references}.items()
                },
                "comparisons": comparisons,
            }
            tensor_file = args.output / f"layer-{layer}-frame-{frame}.pt"
            torch.save(
                {
                    **values,
                    **references,
                    "captured_rank0": captured[0],
                    "captured_rank1": captured[1],
                    "input_rank0": xs[0],
                    "input_rank1": xs[1],
                },
                tensor_file,
            )
            record["tensor_file"] = tensor_file.name
            if args.profile:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
                ) as cpu_profile:
                    cpu_profiled = compute()
                record["cpu_dispatch_events"] = [
                    {
                        "name": event.key,
                        "count": event.count,
                        "input_shapes": event.input_shapes,
                        "cpu_time_total_us": event.cpu_time_total,
                    }
                    for event in cpu_profile.key_averages(group_by_input_shape=True)
                ]
                profile_root = args.output / f"profile-layer-{layer}-frame-{frame}"
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
                record["profile_repeat_hashes"] = {
                    "cpu_profile": {
                        name: sha(value) for name, value in cpu_profiled.items()
                    },
                    "npu_profile": {
                        name: sha(value) for name, value in npu_profiled.items()
                    },
                }
                record["profile_matches_unprofiled"] = all(
                    item == hashes[0]
                    for item in record["profile_repeat_hashes"].values()
                )
                record["npu_kernel_tables"] = []
                for csv_path in sorted(profile_root.rglob("kernel_details.csv")):
                    with csv_path.open(newline="") as handle:
                        kernel_rows = list(csv.DictReader(handle))
                    record["npu_kernel_tables"].append(
                        {
                            "file": str(csv_path.relative_to(args.output)),
                            "rows": kernel_rows,
                        }
                    )
            records.append(record)
            result = {
                "runtime": runtime,
                "method": "Symmetric same-input local down_proj replay with FP64 CPU references; compare local-product hashes across machines before attributing a collective difference",
                "records": records,
            }
            (args.output / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            log_record = {
                key: value
                for key, value in record.items()
                if key
                not in ("comparisons", "cpu_dispatch_events", "npu_kernel_tables")
            }
            log_record["comparisons"] = {
                name: {
                    key: value for key, value in item.items() if key != "differences"
                }
                for name, item in comparisons.items()
            }
            # Keep the diagnostic mismatch coordinates available even when only
            # the GitHub job log can be retrieved. Full reference deltas are in JSON.
            log_record["sum_capture_differences"] = comparisons[
                "sum_vs_captured_rank0"
            ]["differences"]
            log_record["sum_capture_operand_details"] = [
                {
                    "index": delta["index"],
                    "local_products": {
                        str(rank): {
                            "actual": values[f"product_rank{rank}"][
                                tuple(delta["index"])
                            ].item(),
                            "fp64": exact[rank][tuple(delta["index"])].item(),
                            "fp64_rounded_bf16": rounded_products[rank][
                                tuple(delta["index"])
                            ].item(),
                        }
                        for rank in (0, 1)
                    },
                }
                for delta in log_record["sum_capture_differences"]
            ]
            log_record["product_reference_differences"] = {
                str(rank): comparisons[f"product_rank{rank}_vs_fp64_rounded"][
                    "differences"
                ]
                for rank in (0, 1)
            }
            print(
                "DOWN_PROJECTION_REPLAY",
                json.dumps(log_record, allow_nan=False),
                flush=True,
            )
            if args.profile:
                print(
                    "DOWN_PROJECTION_CPU_DISPATCH",
                    json.dumps(
                        {
                            "layer": layer,
                            "frame": frame,
                            "events": record["cpu_dispatch_events"],
                        }
                    ),
                    flush=True,
                )
                print(
                    "DOWN_PROJECTION_KERNELS",
                    json.dumps(
                        {
                            "layer": layer,
                            "frame": frame,
                            "tables": record["npu_kernel_tables"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
