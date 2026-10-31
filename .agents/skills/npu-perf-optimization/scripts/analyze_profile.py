#!/usr/bin/env python3
"""Identify performance bottlenecks from an SGLang Ascend profile.

Phase 2 of the npu-perf-optimization skill. Consumes the parsed profiler output
(operator_details.csv / step_trace_time.csv produced automatically after
`bench_serving --profile`) and writes `bottleneck.json` with top operators,
stage split, and an attribution class.

Usage:
    python3 analyze_profile.py --profile-root <workdir>/profiles/ \
        --out <workdir>/bottleneck.json

Design notes:
- CSV column names drift across CANN versions; every column lookup goes through
  pick_column() with candidates. Calibrate on first real trace.
- Attribution thresholds are named constants; they encode the skill's decision
  policy, not magic numbers scattered in logic.
- Category patterns are data-driven from CATEGORIES; add patterns there.
"""

import argparse
import csv
import glob
import json
import os
import time

TOP_N_OPERATORS = 15
COMM_BOUND_PCT = 30.0        # comm share above this => comm-bound attribution
LAUNCH_GAP_PCT = 25.0        # non-overlapped share above this => launch-bound
COMPUTE_BOUND_PCT = 60.0     # compute share above this => compute-bound

# Operator classification patterns (substring match, case-insensitive).
CATEGORIES = [
    ("communication", ["hccl", "hccd", "allreduce", "all_reduce", "allgather",
                        "all_gather", "broadcast", "reduce_scatter", "send", "recv",
                        "deepep", "dispatch", "combine"]),
    ("attention", ["flash", "attention", "fused_attention", "pfa", "fia"]),
    ("matmul", ["matmul", "gemm", "linear", "mm_kernel", "cublas", "blas"]),
    ("norm_rope", ["norm", "rope", "rmsnorm", "layernorm"]),
    ("elementwise", ["add", "mul", "silu", "activation", "cast", "transpose",
                      "reshape", "concat", "split", "gather", "scatter", "copy"]),
]

DURATION_COL_CANDIDATES = [
    "Device Total Duration With AICore(us)",  # ASCEND canonical (preferred: actual compute)
    "Device Total Duration(us)",
    "Device Self Duration With AICore(us)",
    "Device Self Duration(us)",
    "Host Total Duration(us)",
    "Duration(us)", "duration_us", "Duration(µs)", "dur_us",
]
NAME_COL_CANDIDATES = ["Name", "Operator Name", "Op Name", "name"]
TYPE_COL_CANDIDATES = ["Type", "Operator Type", "type"]


def find_parsed_dir(profile_root):
    """Locate the newest parsed ASCEND_PROFILER_OUTPUT under the trace root."""
    pattern = os.path.join(profile_root, "**", "*_ascend_pt", "ASCEND_PROFILER_OUTPUT")
    hits = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not hits:
        # also accept being handed the parsed dir directly
        if os.path.exists(os.path.join(profile_root, "operator_details.csv")):
            return profile_root
        raise FileNotFoundError(
            f"no *_ascend_pt/ASCEND_PROFILER_OUTPUT under {profile_root}; "
            "check the server log line 'Profiling done. Traces are saved to:'")
    return hits[-1]


def pick_column(fieldnames, candidates):
    for cand in candidates:
        if cand in fieldnames:
            return cand
    lowered = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def classify(name):
    low = name.lower()
    for category, patterns in CATEGORIES:
        if any(p in low for p in patterns):
            return category
    return "other"


def read_operator_table(parsed_dir):
    # Prefer kernel_details.csv — operator_details.csv's "Device Total Duration With AICore(us)"
    # column is populated 0 on newer ASCEND builds; kernel_details.csv has the same per-kernel
    # Name + working "Duration(us)" column.
    for fname in ("kernel_details.csv", "operator_details.csv"):
        path = os.path.join(parsed_dir, fname)
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            dur_col = pick_column(reader.fieldnames or [], DURATION_COL_CANDIDATES)
            name_col = pick_column(reader.fieldnames or [], NAME_COL_CANDIDATES)
            if not dur_col or not name_col:
                continue
            for row in reader:
                try:
                    dur = float(row[dur_col].replace(",", ""))
                except (ValueError, AttributeError, KeyError):
                    continue
                op_name = row.get(name_col, "")
                rows.append({"name": op_name, "duration_us": dur,
                             "category": classify(op_name)})
            if rows:
                return rows
    raise RuntimeError(
        f"no usable operator table in {parsed_dir} "
        f"(checked kernel_details.csv and operator_details.csv)")


def aggregate(rows):
    total = sum(r["duration_us"] for r in rows) or 1.0
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], 0.0)
        by_cat[r["category"]] += r["duration_us"]
    by_cat_pct = {k: round(v / total * 100, 2) for k, v in sorted(
        by_cat.items(), key=lambda kv: -kv[1])}
    # merge duplicate operator names, keep top-N by total duration
    by_op = {}
    for r in rows:
        agg = by_op.setdefault(r["name"], {"duration_us": 0.0, "count": 0,
                                           "category": r["category"]})
        agg["duration_us"] += r["duration_us"]
        agg["count"] += 1
    top = sorted(by_op.items(), key=lambda kv: -kv[1]["duration_us"])[:TOP_N_OPERATORS]
    top_ops = [{"name": k, "category": v["category"], "count": v["count"],
                "duration_us": round(v["duration_us"], 1),
                "pct_total": round(v["duration_us"] / total * 100, 2)}
               for k, v in top]
    return total, by_cat_pct, top_ops


def read_stage_split(parsed_dir):
    path = os.path.join(parsed_dir, "step_trace_time.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # columns vary; sum any column containing 'Duration' as a coarse stage total
    cols = [c for c in (rows[0].keys() if rows else [])
            if c and "duration" in c.lower()]
    split = {}
    for c in cols:
        try:
            split[c] = round(sum(float(r[c].replace(",", "")) for r in rows
                                 if r.get(c)), 1)
        except (ValueError, TypeError):
            continue
    return split or None


def attribute(by_cat_pct):
    comm = by_cat_pct.get("communication", 0.0)
    compute = (by_cat_pct.get("matmul", 0.0) + by_cat_pct.get("attention", 0.0))
    other = by_cat_pct.get("elementwise", 0.0) + by_cat_pct.get("other", 0.0) \
        + by_cat_pct.get("norm_rope", 0.0)
    if comm >= COMM_BOUND_PCT:
        return "comm"
    if compute >= COMPUTE_BOUND_PCT:
        return "compute"
    if other >= LAUNCH_GAP_PCT:
        return "launch"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-root", required=True,
                    help="trace root dir (Method C output dir) or parsed dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parsed_dir = find_parsed_dir(args.profile_root)
    print(f"[profile] parsed dir: {parsed_dir}")
    rows = read_operator_table(parsed_dir)
    total_us, by_cat_pct, top_ops = aggregate(rows)
    stage_split = read_stage_split(parsed_dir)
    attribution = attribute(by_cat_pct)

    artifact = {
        "schema_version": "1.0",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parsed_dir": parsed_dir,
        "total_duration_us": round(total_us, 1),
        "category_share_pct": by_cat_pct,
        "top_operators": top_ops,
        "stage_split_raw": stage_split,
        "attribution": attribution,
        "hints": {
            "comm": "consider deepep mode tuning, HCCL_BUFFSIZE, int8 dispatch quant",
            "compute": "consider quantization (modelslim W8A8/W4A8), MLAPO, NZ weight format",
            "launch": "consider NPU Graph (--cuda-graph-bs), overlap plan stream, tokenizer workers",
            "mixed": "compare per-stage splits; run profile-by-stage for prefill vs decode",
        }.get(attribution),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=1)
    print(f"[profile] attribution={attribution}; wrote {args.out}")


if __name__ == "__main__":
    main()
