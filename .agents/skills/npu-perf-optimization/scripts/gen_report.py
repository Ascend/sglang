#!/usr/bin/env python3
"""Synthesize the performance optimization report from collected artifacts.

Phase 4 of the npu-perf-optimization skill. Reads baseline.json, ab_result.json
and bottleneck.json, writes report.json + report.md, including the mandatory
per-switch before/after table and the recommended launch command.

Usage:
    python3 gen_report.py --baseline <workdir>/baseline/baseline.json \
        --ab <workdir>/ab/ab_result.json \
        [--bottleneck <workdir>/bottleneck.json] \
        --out-dir <workdir>/reports/

Quality gates enforced here (mirror SKILL.md):
- deltas at/below the noise floor are labeled "no significant change"
- the recommendation must re-run run_ab_matrix.py --validate-only and pass
- every number cites its source artifact
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ab_matrix import NOISE_FLOOR_PCT, validate_variant

# TPOT lower is better; throughput higher is better.
IMPROVE_SIGN = {"tpot_delta_pct": -1, "throughput_delta_pct": +1}


def significant(delta_pct):
    return delta_pct is not None and abs(delta_pct) > NOISE_FLOOR_PCT


def is_improvement(field, delta_pct):
    return significant(delta_pct) and delta_pct * IMPROVE_SIGN[field] > 0


def pick_recommendation(base_cfg, ab):
    """Greedy per-switch pick over ok-variants, then compatibility-check the merged config."""
    best_per_switch = {}
    merged_overrides = {}
    for v in ab["variants"]:
        if v.get("status") != "ok":
            continue
        vs = v.get("vs_baseline", {})
        score = 0.0
        for field in IMPROVE_SIGN:
            d = vs.get(field)
            if is_improvement(field, d):
                score += abs(d)
        if score <= 0:
            continue
        switch_keys = tuple(sorted(v.get("switches", {}).get("launch_args", {})
                                   .keys()))
        if score > best_per_switch.get(switch_keys, (0, None))[0]:
            best_per_switch[switch_keys] = (score, v)
    for _, (_, v) in best_per_switch.items():
        merged_overrides.update(v.get("switches", {}).get("launch_args", {}))
        merged_env = v.get("switches", {}).get("env", {})
        merged_overrides_env = merged_overrides.setdefault("__env__", {})
        merged_overrides_env.update(merged_env)

    overrides = {"launch_args": {k: v for k, v in merged_overrides.items()
                                  if k != "__env__"},
                 "env": merged_overrides.get("__env__", {})}
    cfg, errors = validate_variant(base_cfg, overrides)
    return overrides, cfg, errors


def fmt_cmd(cfg):
    """Render env exports + a copy-pasteable launch command (flag+values per line)."""
    env_lines = [f"export {k}={v}" for k, v in sorted(cfg.get("env", {}).items())]
    # a token starting a new option gets its own line; everything else (values,
    # list continuations like cuda-graph-bs sizes) joins the previous line
    lines = []
    for tok in cfg["launch_command"]:
        if lines and not tok.startswith("--"):
            lines[-1] += f" {tok}"
        else:
            lines.append(tok)
    cmd = " \\\n    ".join(lines)
    return "\n".join(env_lines + ["", cmd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ab", required=True)
    ap.add_argument("--bottleneck", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)
    with open(args.ab) as f:
        ab = json.load(f)
    bottleneck = None
    if args.bottleneck and os.path.exists(args.bottleneck):
        with open(args.bottleneck) as f:
            bottleneck = json.load(f)

    base_m = baseline["metrics"]
    rows = []
    for v in ab["variants"]:
        vs = v.get("vs_baseline", {})
        if v.get("status") != "ok":
            why = v.get("error") or "; ".join(v.get("compat_errors", [])) or v.get("status")
            rows.append((v["variant"], v.get("status"), None, None,
                         f"failed: {str(why)[:80]}"))
            continue
        tpot_d, thr_d = vs.get("tpot_delta_pct"), vs.get("throughput_delta_pct")
        verdict = []
        for field, d in (("TPOT", tpot_d), ("throughput", thr_d)):
            key = f"{'tpot' if field == 'TPOT' else 'throughput'}_delta_pct"
            if not significant(d):
                verdict.append(f"{field}: no significant change")
            elif is_improvement(key, d):
                verdict.append(f"{field}: {abs(d)}% better")
            else:
                verdict.append(f"{field}: {abs(d)}% worse")
        rows.append((v["variant"], "ok", tpot_d, thr_d, "; ".join(verdict)))

    overrides, rec_cfg, compat_errors = pick_recommendation(
        baseline["config"], ab)

    report = {
        "schema_version": "1.0",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provenance": {
            "baseline": os.path.abspath(args.baseline),
            "ab": os.path.abspath(args.ab),
            "bottleneck": os.path.abspath(args.bottleneck) if bottleneck else None,
            "meta": baseline["meta"],
            "workload": baseline["workload"],
        },
        "baseline_metrics": {k: base_m.get(k) for k in
                              ("tpot_ms_p50", "ttft_ms_p50", "throughput_tok_s")},
        "bottleneck": ({"attribution": bottleneck["attribution"],
                         "category_share_pct": bottleneck["category_share_pct"],
                         "top_operators": bottleneck["top_operators"][:5]}
                        if bottleneck else None),
        "comparisons": [
            {"variant": r[0], "status": r[1],
             "tpot_delta_pct": r[2], "throughput_delta_pct": r[3],
             "verdict": r[4] if r[4] else "rejected/failed"}
            for r in rows],
        "recommendation": {
            "overrides": overrides,
            "compat_errors": compat_errors,
            "launch_command": rec_cfg["launch_command"],
            "env": rec_cfg.get("env", {}),
            "expected": {
                v["variant"]: v.get("vs_baseline")
                for v in ab["variants"]
                if v.get("status") == "ok" and
                any(k in (v.get("switches", {}).get("launch_args") or {})
                    for k in overrides["launch_args"])},
        },
        "noise_floor_pct": NOISE_FLOOR_PCT,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=1)

    md = []
    md.append(f"# SGLang Ascend NPU Performance Optimization Report")
    md.append("")
    md.append(f"- Date: {report['date']}")
    m = baseline["meta"]
    md.append(f"- Hardware: {m.get('hardware')} x{m.get('cards')} | "
              f"Model: {m.get('model')} ({m.get('quant')}) | sglang: {m.get('sglang_commit')}")
    md.append(f"- Workload: `{json.dumps(baseline['workload'])}`")
    md.append(f"- Noise floor: ±{NOISE_FLOOR_PCT}% (deltas within this are noise)")
    md.append("")
    md.append(f"## Baseline")
    md.append(f"- TPOT p50: **{base_m.get('tpot_ms_p50')} ms** | "
              f"TTFT p50: {base_m.get('ttft_ms_p50')} ms | "
              f"Throughput: {base_m.get('throughput_tok_s')} tok/s")
    if bottleneck:
        md.append(f"- Bottleneck attribution: **{bottleneck['attribution']}** "
                  f"(share: {json.dumps(bottleneck['category_share_pct'])})")
        md.append(f"- Hint: {bottleneck.get('hints') or ''}")
    md.append("")
    md.append("## Feature-switch A/B (each vs baseline, identical workload)")
    md.append("")
    md.append("| Variant | Status | TPOT Δ% | Throughput Δ% | Verdict |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2] if r[2] is not None else '-'} | "
                  f"{r[3] if r[3] is not None else '-'} | {r[4] or '-'} |")
    md.append("")
    md.append("## Recommended launch command")
    if compat_errors:
        md.append("> ⚠️ compatibility check FAILED — do not use as-is:")
        md.append(">\n".join(f"> - {e}" for e in compat_errors))
    md.append("```bash")
    md.append(fmt_cmd(rec_cfg))
    md.append("```")
    md.append("")
    md.append(f"Applied overrides: `{json.dumps(overrides, ensure_ascii=False)}`")
    md.append("")
    md.append("*(all numbers trace to the artifacts listed in report.json provenance)*")

    md_path = os.path.join(args.out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[report] wrote {md_path} and {json_path}"
          + (" (⚠ compat errors in recommendation)" if compat_errors else ""))


if __name__ == "__main__":
    main()
