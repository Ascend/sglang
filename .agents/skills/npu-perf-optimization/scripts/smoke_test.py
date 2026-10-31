#!/usr/bin/env python3
"""Local smoke test for the npu-perf-optimization scripts (no NPU needed).

Covers the pure-logic paths: bench command construction, bench output parsing,
compat-rule validation, and recommendation picking. Run from the scripts dir:

    python3 smoke_test.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect_baseline import build_bench_command, parse_bench_output
from run_ab_matrix import validate_variant, COMPAT_RULES

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# --- build_bench_command: spec -> official bench args -----------------------
workload = {
    "dataset_name": "random", "backend": "sglang", "host": "127.0.0.1",
    "port": 6688, "max_concurrency": 64, "num_prompts": 256,
    "random_input_len": 3500, "random_output_len": 1500,
    "random_range_ratio": 1, "seed": 1,
}
cmd = build_bench_command(workload)
check("bench cmd flags", "--dataset-name" in cmd and "--random-input-len" in cmd
      and "--max-concurrency" in cmd)
check("bench cmd values", cmd[cmd.index("--random-input-len") + 1] == "3500")
check("bench cmd module", cmd[1:3] == ["-m", "sglang.bench_serving"])

# --- parse_bench_output: JSON block extraction + metric candidates ---------
sample_stdout = """
Random dataset...
Benchmark duration: 61.23 s
{ "successful_requests": 256, "median_ttft_ms": 120.5,
  "median_otps_ms": 37.2, "output_throughput": 4123.4,
  "completed": 256, "benchmark_duration": 61.23 }
"""
parsed = parse_bench_output(sample_stdout)
check("parse tpot", parsed.get("tpot_ms") == 37.2, f"got {parsed.get('tpot_ms')}")
check("parse ttft", parsed.get("ttft_ms") == 120.5)
check("parse throughput", parsed.get("throughput_tok_s") == 4123.4)
check("parse completed", parsed.get("completed") == 256)

# --- compat rules on a realistic config --------------------------------------
base_cfg = {
    "launch_command": ["python3", "-m", "sglang.launch_server",
                        "--model-path", "/models/Qwen3-8B",
                        "--attention-backend", "ascend", "--device", "npu",
                        "--quantization", "modelslim", "--tp-size", "2",
                        "--cuda-graph-bs", "8", "16", "32",
                        "--port", "6688"],
    "env": {"HCCL_OP_EXPANSION_MODE": "AIV"},
    "port": 6688,
}
cfg, errors = validate_variant(base_cfg, {})
check("clean config passes", errors == [], str(errors))

# torch-compile vs graph: enabling compile while graph is on must reject
cfg2, errors2 = validate_variant(
    base_cfg, {"env": {"SGLANG_ENABLE_TORCH_COMPILE": "1"}})
check("torch_compile+graph rejected", any("torch-compile" in e for e in errors2),
      str(errors2))
# ...and disabling the graph clears it
cfg3, errors3 = validate_variant(
    base_cfg, {"env": {"SGLANG_ENABLE_TORCH_COMPILE": "1"},
                "launch_args": {"--disable-cuda-graph": ""}})
check("torch_compile+disable-graph ok",
      not any("torch-compile" in e for e in errors3), str(errors3))

# deepep dispatch cap: >1024 must reject
cfg4, errors4 = validate_variant(
    base_cfg, {"launch_args": {"--moe-a2a-backend": "deepep"},
                "env": {"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "2048"}})
check("deepep cap rejected", any("1024" in e for e in errors4), str(errors4))

# EAGLE3 without draft model must reject; NEXTN must pass
cfg5, errors5 = validate_variant(
    base_cfg, {"launch_args": {"--speculative-algorithm": "EAGLE3"}})
check("eagle3-no-draft rejected", any("draft-model-path" in e for e in errors5),
      str(errors5))
cfg6, errors6 = validate_variant(
    base_cfg, {"launch_args": {"--speculative-algorithm": "NEXTN"}})
check("nextn ok", not any("draft-model" in e for e in errors6), str(errors6))

# flag removal: --cuda-graph-bs removal via None
cfg7, _ = validate_variant(
    base_cfg, {"launch_args": {"--cuda-graph-bs": None}})
check("flag removal works", "--cuda-graph-bs" not in cfg7["launch_command"])

# --- gen_report recommendation picking (end-to-end on fake artifacts) --------
from gen_report import pick_recommendation, significant

ab = {"variants": [
    {"variant": "npugraph_off", "status": "ok",
     "switches": {"launch_args": {"--cuda-graph-bs": None}},
     "vs_baseline": {"tpot_delta_pct": 12.0, "throughput_delta_pct": -10.0}},
    {"variant": "mtp_eagle3", "status": "ok",
     "switches": {"launch_args": {"--speculative-algorithm": "EAGLE3",
                                    "--speculative-draft-model-path": "/d",
                                    "--speculative-num-steps": 3}},
     "vs_baseline": {"tpot_delta_pct": -25.0, "throughput_delta_pct": 30.0}},
    {"variant": "dp_attn", "status": "failed", "switches": {}},
]}
overrides, rec_cfg, rec_errors = pick_recommendation(base_cfg, ab)
check("picks improving switch", "--speculative-algorithm" in
      overrides["launch_args"], str(overrides))
check("skips worsening switch", "--cuda-graph-bs" not in
      overrides["launch_args"] or overrides["launch_args"]["--cuda-graph-bs"] is None)
check("rec compat clean", rec_errors == [], str(rec_errors))
check("noise floor", not significant(2.9) and significant(3.1))

print()
if FAILED:
    print(f"SMOKE TEST FAILED: {len(FAILED)} checks: {FAILED}")
    sys.exit(1)
print("ALL SMOKE CHECKS PASSED")
