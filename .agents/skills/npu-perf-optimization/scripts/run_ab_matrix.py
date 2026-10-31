#!/usr/bin/env python3
"""Run the NPU feature-switch A/B matrix against a collected baseline.

Phase 3 of the npu-perf-optimization skill. Takes a matrix plan (list of
variants, each = launch-arg/env overrides on top of the baseline server config),
validates each variant against the compatibility rules, then runs
baseline-equivalent benches per variant and writes `ab_result.json`.

Usage:
    python3 run_ab_matrix.py --plan matrix_plan.json \
        --baseline <workdir>/baseline/baseline.json --out-dir <workdir>/ab/
    python3 run_ab_matrix.py --plan matrix_plan.json --validate-only

Design notes:
- Same workload spec as the baseline (comparability rule); the script refuses
  to run if the plan's workload differs from baseline.json's.
- Circuit breaker: a failed variant is recorded with its log tail and skipped;
  the matrix continues.
- Checkpointing: `state.json` in the out-dir tracks completed variants; rerun
  resumes where it stopped.
- Compatibility rules live in COMPAT_RULES below (machine-checkable subset of
  references/feature-switch-matrix.md; the doc stays authoritative for humans).
"""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_baseline import (build_bench_command, median_of, run_bench,
                              sanity_check, wait_server_ready)

# Noise floor: deltas at or below this are "no significant change" (matches the
# skill's quality gate; derived from observed run-to-run variance).
NOISE_FLOOR_PCT = 3.0

# Machine-checkable compatibility rules (source: parameter_tuning.mdx feature
# matrix + launch args interplay). Each rule: (name, predicate, explanation).
def _has_arg(cfg, flag):
    return flag in cfg["launch_command"]

def _arg_value(cfg, flag):
    cmd = cfg["launch_command"]
    if flag in cmd:
        i = cmd.index(flag)
        return cmd[i + 1] if i + 1 < len(cmd) else True
    return None

def _env(cfg, key):
    return cfg.get("env", {}).get(key)

COMPAT_RULES = [
    ("torch_compile_vs_npu_graph",
     lambda c: not (_env(c, "SGLANG_ENABLE_TORCH_COMPILE") == "1"
                    or _has_arg(c, "--enable-torch-compile"))
               or _has_arg(c, "--disable-cuda-graph"),
     "--enable-torch-compile requires --disable-cuda-graph (mutually exclusive with NPU Graph)"),
    ("task_queue2_vs_graph",
     lambda c: _env(c, "TASK_QUEUE_ENABLE") != "2" or _has_arg(c, "--disable-cuda-graph"),
     "TASK_QUEUE_ENABLE=2 interferes with NPU Graph capture; use 0/1 or disable the graph"),
    ("eplb_vs_fuseep",
     lambda c: not _has_arg(c, "--enable-eplb")
               or _arg_value(c, "--moe-a2a-backend") != "ascend_fuseep",
     "EPLB is only compatible with the deepep A2A backend, not ascend_fuseep"),
    ("deepep_dispatch_cap",
     lambda c: _arg_value(c, "--moe-a2a-backend") != "deepep"
               or (_env(c, "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK")
                   in (None, "0"))
               or int(_env(c, "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK")) <= 1024,
     "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK hard cap is 1024"),
    ("deepep_dispatch_covers_requests",
     lambda c: _arg_value(c, "--moe-a2a-backend") != "deepep"
               or _env(c, "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK") in (None, "0")
               or (max_running := _arg_value(c, "--max-running-requests")) is None
               or int(max_running) <= int(_env(c,
                   "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK")),
     "need max-running-requests * (1 + draft_tokens) <= dispatch tokens per rank"),
    ("spec_draft_model_present",
     lambda c: _arg_value(c, "--speculative-algorithm") is None
               or _has_arg(c, "--speculative-draft-model-path")
               or _arg_value(c, "--speculative-algorithm") in ("NEXTN", "NGRAM"),
     "EAGLE/EAGLE3-style speculative decoding needs --speculative-draft-model-path; "
     "NEXTN uses built-in MTP heads, NGRAM has no draft worker at all"),
    ("dp_attention_needs_tp",
     lambda c: not _has_arg(c, "--enable-dp-attention")
               or (tp := _arg_value(c, "--tp-size")) is None or int(tp) >= 2,
     "DP-attention only pays off / is exercised with tp-size >= 2"),
]


def validate_variant(base_cfg, overrides):
    """Apply overrides to a copy of the base server config; return (cfg, errors)."""
    cfg = copy.deepcopy(base_cfg)
    cmd = cfg["launch_command"]
    for flag, value in overrides.get("launch_args", {}).items():
        name = flag if flag.startswith("--") else f"--{flag}"
        if value is None:                       # None removes the flag + its value(s)
            if name in cmd:
                i = cmd.index(name)
                cmd[i:i + 2] = []
            continue
        if value == "":                          # empty string = boolean toggle (bare flag)
            if name in cmd:
                i = cmd.index(name)
                cmd[i:i + 2] = []
            cmd.append(name)
            continue
        if name in cmd:
            i = cmd.index(name)
            cmd[i + 1] = str(value)
        else:
            cmd += [name, str(value)]
    for flag in overrides.get("launch_flags_off", []):
        name = flag if flag.startswith("--") else f"--{flag}"
        if name in cmd:
            i = cmd.index(name)
            cmd[i:i + 2] = []
    cfg.setdefault("env", {}).update(
        {k: str(v) for k, v in overrides.get("env", {}).items()})
    for k in overrides.get("env_unset", []):
        cfg["env"].pop(k, None)

    errors = [msg for _, pred, msg in COMPAT_RULES if not pred(cfg)]
    return cfg, errors


def delta_pct(new, base):
    if new is None or base in (None, 0):
        return None
    return round((new - base) / base * 100.0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True,
                    help="matrix plan JSON: {base_server_config, workload?, variants[]}")
    ap.add_argument("--baseline", help="baseline.json (required unless --validate-only)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--runs-per-variant", type=int, default=2)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    base_cfg = plan["base_server_config"]

    variants = plan["variants"]
    print(f"[matrix] {len(variants)} variants; validating compatibility rules")
    plans_ok = True
    for v in variants:
        _, errors = validate_variant(base_cfg, v.get("overrides", {}))
        v["_compat_errors"] = errors
        if errors:
            plans_ok = False
            print(f"[matrix]   REJECT {v['name']}: {'; '.join(errors)}")
        else:
            print(f"[matrix]   ok     {v['name']}")
    if args.validate_only:
        sys.exit(0 if plans_ok else 1)

    with open(args.baseline) as f:
        baseline = json.load(f)
    if plan.get("workload") and plan["workload"] != baseline["workload"]:
        sys.exit("[matrix] FATAL: plan workload != baseline workload; "
                 "A/B results would not be comparable")
    workload = baseline["workload"]
    base_tpot = baseline["metrics"]["tpot_ms_p50"]
    base_thr = baseline["metrics"].get("throughput_tok_s")

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.plan), "..", "ab")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "state.json")
    state = {"done": {}}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"[matrix] resuming: {len(state['done'])} variants already done")

    port = base_cfg["port"]
    results = []
    for v in variants:
        name = v["name"]
        if name in state["done"]:
            results.append(state["done"][name])
            continue
        if v["_compat_errors"]:
            results.append({"variant": name, "status": "rejected",
                            "compat_errors": v["_compat_errors"]})
            continue

        cfg, _ = validate_variant(base_cfg, v.get("overrides", {}))
        server_log = os.path.join(out_dir, f"server_{name}.log")
        env = os.environ.copy()
        env.update(cfg.get("env", {}))
        entry = {"variant": name, "switches": v.get("overrides", {}),
                 "status": "failed", "log": server_log}
        server = None
        try:
            print(f"[matrix] variant {name}: launching")
            log_f = open(server_log, "w")
            server = subprocess.Popen(cfg["launch_command"],
                                      stdout=log_f, stderr=subprocess.STDOUT, env=env)
            wait_server_ready(port, args.startup_timeout,
                              base_cfg.get("ready_pattern",
                                           "The server is fired up and ready to roll!"),
                              server_log)
            sanity_check(port, 5)
            cmd = build_bench_command(workload)
            runs = run_bench(cmd, args.runs_per_variant, out_dir, name)
            tpot = median_of("tpot_ms", runs)
            thr = median_of("throughput_tok_s", runs)
            if tpot is None:
                raise RuntimeError("no TPOT parsed; see bench logs")
            entry.update({
                "status": "ok",
                "metrics": {"tpot_ms_p50": tpot, "throughput_tok_s": thr,
                            "runs": runs},
                "vs_baseline": {
                    "tpot_delta_pct": delta_pct(tpot, base_tpot),
                    "throughput_delta_pct": delta_pct(thr, base_thr),
                },
            })
        except Exception as exc:  # circuit breaker: record, continue
            with open(server_log, "r", errors="replace") as f:
                tail = f.read()[-2000:]
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["log_tail"] = tail
            print(f"[matrix] variant {name} FAILED ({entry['error']}); continuing")
        finally:
            if server is not None and server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
        state["done"][name] = entry
        results.append(entry)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)

    artifact = {
        "schema_version": "1.0",
        "baseline_ref": os.path.abspath(args.baseline),
        "noise_floor_pct": NOISE_FLOOR_PCT,
        "variants": results,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = os.path.join(out_dir, "ab_result.json")
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=1)
    print(f"[matrix] wrote {out_path} "
          f"({sum(1 for r in results if r['status'] == 'ok')} ok / {len(results)})")


if __name__ == "__main__":
    main()
