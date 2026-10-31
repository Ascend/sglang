# JSON Schemas (I/O contracts)

All inter-phase handoff is JSON on disk. `schema_version` gates parsing; bump
on breaking change. Unknown extra keys are allowed (forward-compatible) —
readers must ignore what they don't know.

## §A `server_config.json` (input to Phase 1 / matrix base)

```jsonc
{
  "launch_command": ["python3", "-m", "sglang.launch_server", "…"],  // verbatim, caller-supplied
  "env": {"HCCL_OP_EXPANSION_MODE": "AIV"},                          // merged over current env
  "port": 6688,
  "ready_pattern": "The server is fired up and ready to roll!",      // optional override
  "startup_timeout_s": 900,
  // meta (copied into artifacts; source of truth = the caller)
  "hardware": "Atlas 800I A3", "cards": 1, "model": "Qwen/Qwen3-8B",
  "quant": "W8A8(modelslim)", "tp_size": 2, "sglang_commit": "…"
}
```

## §B `baseline/baseline.json` (Phase 1 output)

```jsonc
{
  "schema_version": "1.0",
  "meta": { /* hardware, cards, model, quant, tp_size, sglang_commit, date */ },
  "config": { /* the server_config used */ },
  "workload": { /* frozen bench spec, §F */ },
  "metrics": {
    "tpot_ms_p50": 37.2, "ttft_ms_p50": 120.5, "throughput_tok_s": 4123.4,
    "completed": 256,
    "runs": [ {"returncode":0, "log":"…", /* parsed per-run metrics */, "raw_json":"…"} ]
  },
  "sanity": {"short_gen_ok": true, "num_requests": 20}
}
```

Invariant: `tpot_ms_p50` is the **median of ≥2 successful runs**; the raw runs
stay embedded for audit.

## §C `bottleneck.json` (Phase 2 output)

```jsonc
{
  "schema_version": "1.0",
  "parsed_dir": "…/*_ascend_pt/ASCEND_PROFILER_OUTPUT",
  "total_duration_us": 0.0,
  "category_share_pct": {"matmul": 41.2, "communication": 30.5, "…": 0},
  "top_operators": [ {"name":"…","category":"matmul","count":312,
                       "duration_us":0.0,"pct_total":12.3} ],
  "stage_split_raw": {"<column>": 0.0},      // step_trace_time.csv columns, version-dependent
  "attribution": "compute | comm | launch | mixed",
  "hints": "switch guidance matching the attribution"
}
```

## §D `ab/ab_result.json` (Phase 3 output)

```jsonc
{
  "schema_version": "1.0",
  "baseline_ref": "path/to/baseline.json",
  "noise_floor_pct": 3.0,
  "date": "…",
  "variants": [
    {
      "variant": "npugraph_bs64",
      "switches": { /* overrides applied */ },
      "status": "ok | failed | rejected",
      "metrics": {"tpot_ms_p50": 0, "throughput_tok_s": 0, "runs": [/* §B-style */]},
      "vs_baseline": {"tpot_delta_pct": -8.4, "throughput_delta_pct": 9.1},
      // failure-only:
      "error": "…", "log_tail": "last 2KB of server log", "compat_errors": ["…"]
    }
  ]
}
```

`state.json` (checkpoint) = `{"done": {<variant name>: <variant entry>}}`.

## §E `reports/report.json` (Phase 4 output)

```jsonc
{
  "schema_version": "1.0",
  "date": "…",
  "provenance": {"baseline": "…", "ab": "…", "bottleneck": "…", "meta": {}, "workload": {}},
  "baseline_metrics": {"tpot_ms_p50": 0, "ttft_ms_p50": 0, "throughput_tok_s": 0},
  "bottleneck": {"attribution": "…", "category_share_pct": {}, "top_operators": []},
  "comparisons": [ {"variant":"…","status":"…","tpot_delta_pct":0,
                     "throughput_delta_pct":0,"verdict":"human sentence"} ],
  "recommendation": {
    "overrides": {"launch_args": {}, "env": {}},
    "compat_errors": [],                  // must be [] in a usable recommendation
    "launch_command": ["…"], "env": {},
    "expected": { "<variant>": {"tpot_delta_pct": 0} }
  },
  "noise_floor_pct": 3.0
}
```

## §F frozen workload spec (embedded in §B/§D inputs)

Keys mirror `sglang.bench_serving` args of the official best-practice pages:
`dataset_name, backend, host, port, max_concurrency, num_prompts,
random_input_len, random_output_len, random_range_ratio, seed`. Extra keys
pass through as `--<dashed-key> <value>`. **Frozen rule**: once the baseline is
collected, every variant must reuse the identical object (the matrix runner
hard-fails on mismatch).

## Parsing notes (bench output drift)

`collect_baseline.py::METRIC_CANDIDATES` maps our canonical names
(`tpot_ms`, `ttft_ms`, `throughput_tok_s`, `completed`, `duration_s`) onto
version-drifting `bench_serving` JSON field names. Extend the candidate lists
there when a new sglang version renames fields; never branch on versions.
