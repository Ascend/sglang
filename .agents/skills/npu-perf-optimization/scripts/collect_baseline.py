#!/usr/bin/env python3
"""Collect a reproducible SGLang-on-Ascend performance baseline.

Phase 1 of the npu-perf-optimization skill. Launches the server described by a
server config, sanity-checks it, runs `sglang.bench_serving` N times with a
frozen workload spec, and writes `baseline.json` (schema: references/json-schemas.md).

Usage:
    python3 collect_baseline.py --server-config server_config.json \
        --workload workload.json --out-dir <workdir>/baseline/

Design notes:
- Standard library only (runs on pristine NPU containers).
- The launch command and bench arguments come verbatim from the caller's config
  files (copy them from the official best-practice page); this script never
  invents model paths or tuning flags.
- Every run's raw stdout is preserved next to the JSON artifact.
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --- tunables (override via CLI flags, not by editing code) ----------------

DEFAULT_RUNS = 3              # median-of-N against run-to-run noise
DEFAULT_STARTUP_TIMEOUT_S = 900
DEFAULT_REQ_TIMEOUT_S = 120
SANITY_NUM_REQUESTS = 20      # short-generation guard before trusting a baseline
SANITY_MAX_NEW_TOKENS = 8
POLL_INTERVAL_S = 5

# bench_serving result field candidates, newest first. Source: the JSON block
# printed by `python -m sglang.bench_serving` (field names drift across
# versions). First candidate found wins; extend here, never inline.
METRIC_CANDIDATES = {
    "tpot_ms": ["median_otps_ms", "median_tpot_ms", "TPOT (ms)", "tpot_ms"],
    "ttft_ms": ["median_ttft_ms", "median_e2el_ttft_ms", "TTFT (ms)", "ttft_ms"],
    "throughput_tok_s": ["output_throughput", "total_output_throughput",
                          "Output token throughput (tok/s)"],
    "completed": ["completed", "Successful requests"],
    "duration_s": ["benchmark_duration"],
}


def http_json(url, payload=None, timeout=DEFAULT_REQ_TIMEOUT_S):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_server_ready(port, timeout_s, ready_pattern, server_log_path):
    """Poll /health and grep the server log for the ready pattern (either wins)."""
    deadline = time.time() + timeout_s
    health = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        try:
            with open(server_log_path, "r", errors="replace") as f:
                if ready_pattern in f.read():
                    return
        except FileNotFoundError:
            pass
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"server not ready within {timeout_s}s (log: {server_log_path})")


def sanity_check(port, n_requests):
    """Short generations; a server that cannot answer is not a baseline."""
    url = f"http://127.0.0.1:{port}/generate"
    for i in range(n_requests):
        payload = {"text": f"Sanity check {i}: say 'ok'.",
                   "sampling_params": {"max_new_tokens": SANITY_MAX_NEW_TOKENS,
                                        "temperature": 0.0}}
        out = http_json(url, payload)
        text = out.get("text") or ""
        if not text.strip():
            raise RuntimeError(f"sanity request {i} returned empty text: {out!r}")
    return True


def build_bench_command(workload):
    """Map the frozen workload spec onto the official bench_serving arguments.

    Field names mirror `docs/.../best-practices/*.mdx` benchmark blocks; unknown
    keys in the workload are passed through as `--<key-with-dashes> <value>` so
    new bench flags need no code change.
    """
    cmd = [sys.executable, "-m", "sglang.bench_serving"]
    passthrough = {"host", "port", "backend", "dataset_name", "max_concurrency",
                   "num_prompts", "random_input_len", "random_output_len",
                   "random_range_ratio", "seed"}
    for key in passthrough:
        if key in workload:
            cmd += [f"--{key.replace('_', '-')}", str(workload[key])]
    for key, val in workload.items():
        if key not in passthrough:
            cmd += [f"--{key.replace('_', '-')}", str(val)]
    return cmd


def parse_bench_output(stdout_text):
    """Extract the last balanced JSON object and pick metric fields.

    Falls back to aligned-column parsing for newer sglang (>=0.4) which prints
    "Median TPOT (ms):   14.98" style tables without any JSON.
    """
    import re
    result = {"raw_json": None}
    depth, start = 0, None
    for i, ch in enumerate(stdout_text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                result["raw_json"] = stdout_text[start:i + 1]
    obj = None
    if result["raw_json"]:
        try:
            obj = json.loads(result["raw_json"])
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        for metric, candidates in METRIC_CANDIDATES.items():
            for cand in candidates:
                if cand in obj and isinstance(obj[cand], (int, float)):
                    result[metric] = float(obj[cand])
                    break
        if "tpot_ms" in result and "ttft_ms" in result and "throughput_tok_s" in result:
            return result  # JSON path produced a complete triple; don't second-guess
    # Aligned-column fallback: bench_serving newer emits
    #   "Median TPOT (ms):   14.98"
    #   "Median TTFT (ms):   2765.93"
    #   "Output token throughput (tok/s):   38.65"
    # Lines may carry leading section headers + dashes; we just scan all lines.
    COLUMN_PATTERNS = [
        ("tpot_ms", re.compile(r"Median\s+TPOT\s*\(ms\):\s*([0-9]*\.?[0-9]+)")),
        ("ttft_ms", re.compile(r"Median\s+TTFT\s*\(ms\):\s*([0-9]*\.?[0-9]+)")),
        ("throughput_tok_s", re.compile(
            r"Output token throughput\s*\(tok/s\):\s*([0-9]*\.?[0-9]+)")),
        ("completed", re.compile(r"Completed:\s*([0-9]+)")),
        ("duration_s", re.compile(r"Total benchmark duration\s*\(s\):\s*([0-9]*\.?[0-9]+)")),
    ]
    # Build a synthetic obj so downstream METRIC_CANDIDATES loop also sees the values.
    column_obj = {}
    for key, rx in COLUMN_PATTERNS:
        m = rx.search(stdout_text)
        if m:
            try:
                column_obj[key] = float(m.group(1))
            except ValueError:
                pass
    if column_obj:
        for metric, candidates in METRIC_CANDIDATES.items():
            if metric in column_obj and metric not in result:
                result[metric] = column_obj[metric]
        # Surface a fake raw_json with the captured values for artifact audit.
        if not result.get("raw_json"):
            result["raw_json"] = json.dumps(column_obj)
    return result


def run_bench(cmd, runs, out_dir, tag):
    """Run bench N times; return list of parsed results (raw stdout kept)."""
    results = []
    for i in range(runs):
        log_path = os.path.join(out_dir, f"bench_{tag}_run{i + 1}.log")
        with open(log_path, "w") as log_f:
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                  timeout=None)
        with open(log_path, "r", errors="replace") as f:
            parsed = parse_bench_output(f.read())
        parsed["returncode"] = proc.returncode
        parsed["log"] = log_path
        results.append(parsed)
    return results


def median_of(field, results):
    vals = [r[field] for r in results
            if field in r and r.get("returncode") == 0]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-config", required=True,
                    help="JSON with launch_command/env/port/ready_pattern")
    ap.add_argument("--workload", required=True,
                    help="JSON with the frozen bench_serving workload spec")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    ap.add_argument("--keep-server", action="store_true",
                    help="do not kill the server at exit (for follow-up profiling)")
    args = ap.parse_args()

    with open(args.server_config) as f:
        server_cfg = json.load(f)
    with open(args.workload) as f:
        workload = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    port = server_cfg["port"]
    ready_pattern = server_cfg.get("ready_pattern",
                                   "The server is fired up and ready to roll!")
    server_log = os.path.join(args.out_dir, "server.log")

    env = os.environ.copy()
    env.update({k: str(v) for k, v in server_cfg.get("env", {}).items()})

    server = None
    try:
        print(f"[baseline] launching server: {' '.join(server_cfg['launch_command'])}")
        log_f = open(server_log, "w")
        server = subprocess.Popen(server_cfg["launch_command"],
                                  stdout=log_f, stderr=subprocess.STDOUT, env=env)
        wait_server_ready(port, args.startup_timeout, ready_pattern, server_log)
        print("[baseline] server ready, running sanity guard")
        sanity_ok = sanity_check(port, SANITY_NUM_REQUESTS)

        cmd = build_bench_command(workload)
        print(f"[baseline] bench x{args.runs}: {' '.join(cmd)}")
        results = run_bench(cmd, args.runs, args.out_dir, "baseline")

        tpot = median_of("tpot_ms", results)
        if tpot is None:
            raise RuntimeError("no TPOT metric parsed from any bench run; "
                               "check bench_*.log before trusting anything")
        artifact = {
            "schema_version": "1.0",
            "meta": {
                "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "hardware": server_cfg.get("hardware", "unknown"),
                "cards": server_cfg.get("cards"),
                "model": server_cfg.get("model", "unknown"),
                "quant": server_cfg.get("quant"),
                "tp_size": server_cfg.get("tp_size"),
                "sglang_commit": server_cfg.get("sglang_commit", "unknown"),
            },
            "config": server_cfg,
            "workload": workload,
            "metrics": {
                "tpot_ms_p50": tpot,
                "ttft_ms_p50": median_of("ttft_ms", results),
                "throughput_tok_s": median_of("throughput_tok_s", results),
                "completed": median_of("completed", results),
                "runs": results,
            },
            "sanity": {"short_gen_ok": sanity_ok,
                       "num_requests": SANITY_NUM_REQUESTS},
        }
        out_path = os.path.join(args.out_dir, "baseline.json")
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=1)
        print(f"[baseline] wrote {out_path} (median TPOT {tpot} ms)")
    finally:
        if server is not None and server.poll() is None and not args.keep_server:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
        if not args.keep_server:
            print("[baseline] server stopped")


if __name__ == "__main__":
    main()
