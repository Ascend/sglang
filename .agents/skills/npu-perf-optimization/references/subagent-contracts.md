# Sub-agent Contracts (acceptance criterion #2)

Three sub-agents drive the pipeline. All handoff is **JSON files on disk**;
scripts are idempotent; no interactive session is required. The orchestrating
agent (SKILL.md workflow) only sequences phases and never invents parameters —
every tunable lives in the input JSONs.

## Common invocation rules

- Working directory: the skill's `<workdir>`, layout
  `{baseline,ab,profiles,reports}/` (never `/tmp`).
- Timeouts: server startup 900 s (large models / graph capture); single bench
  run 3600 s; sanity 120 s. On timeout: record `status:"failed"` with the log
  tail, continue (circuit breaker) — never retry blindly more than once.
- Every artifact written must be referenced by path in the agent's final
  message, so the orchestrator can chain phases without re-deriving state.
- Idempotency: rerunning a phase with the same inputs must either reuse the
  checkpoint (`state.json`) or overwrite outputs atomically.

---

## 1. `baseline-collector`

**Purpose**: produce `baseline/baseline.json` for one server config.

**System prompt template**:

```
You are the baseline collector for SGLang-on-Ascend performance work.
Given --server-config (launch_command, env, port, ready_pattern, meta) and
--workload (frozen bench_serving spec), you:
1. launch the server via collect_baseline.py (it handles readiness polling,
   the 20-request sanity guard, N bench runs, median-of-N);
2. verify the artifact: metrics.tpot_ms_p50 is non-null and every run's
   returncode is 0 — otherwise report failure with the bench log paths;
3. reply with exactly: artifact path, median TPOT/TTFT/throughput, and any
   anomaly you noticed (e.g. variance between runs > 5%).
You never modify the launch command or workload spec. If either seems wrong,
stop and report instead of improvising.
```

**Invocation template**:

```bash
python3 scripts/collect_baseline.py \
  --server-config <workdir>/server_config.json \
  --workload <workdir>/workload.json \
  --out-dir <workdir>/baseline/ \
  --runs 3
```

**Input contract**: `server_config.json` =
`{launch_command: [str], env: {str:str}, port: int, ready_pattern?: str,
startup_timeout_s?: int, hardware?: str, cards?: int, model?: str, quant?: str,
tp_size?: int, sglang_commit?: str}`.
`workload.json` = the frozen bench spec (see json-schemas.md).

**Output contract**: `baseline/baseline.json` (schema §B) + `bench_baseline_run*.log`.

**Failure semantics**: startup timeout / sanity failure / no TPOT parsed →
non-zero exit; orchestrator must not proceed to A/B without a valid baseline.

---

## 2. `bottleneck-analyzer`

**Purpose**: produce `bottleneck.json` from a profiling run.

**System prompt template**:

```
You are the bottleneck analyzer. You are given a running server (already
launched by the orchestrator with the baseline config) and you:
1. run the profiling bench exactly once:
   python -m sglang.bench_serving <same workload args> --profile \
     --profile-steps 10 --profile-output-dir <workdir>/profiles/
2. locate the parsed dir from the server log line
   "Profiling done. Traces are saved to: <path>";
3. run analyze_profile.py --profile-root <workdir>/profiles/ --out <workdir>/bottleneck.json;
4. summarize to the orchestrator: attribution class, top-3 operators with
   shares, and the matching hint from the artifact.
Do not rename or move trace files. If the parsed dir lacks
operator_details.csv, report the actual file list instead of guessing.
```

**Invocation template**:

```bash
# 1) profile + bench in one command (server already up)
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port <port> \
  <same frozen workload args> \
  --profile --profile-steps 10 --profile-output-dir <workdir>/profiles/

# 2) analyze
python3 scripts/analyze_profile.py \
  --profile-root <workdir>/profiles/ --out <workdir>/bottleneck.json
```

**Input contract**: running server + frozen workload; `profiles/` writable.

**Output contract**: `bottleneck.json` (schema §C).

**Failure semantics**: no trace dir → check server log for
`Profiling done`; missing CSVs → report file list; attribution always one of
`compute|comm|launch|mixed` (never null).

---

## 3. `matrix-runner`

**Purpose**: execute the validated variant matrix, resume-safe.

**System prompt template**:

```
You are the A/B matrix runner. Inputs: matrix_plan.json and the baseline
artifact. You:
1. first run run_ab_matrix.py --validate-only; if it rejects any variant,
   report the rejected variants and their rule violations to the orchestrator
   and wait for an updated plan (do not silently drop them);
2. then run the matrix (it enforces: same workload as baseline, per-variant
   circuit breaker, checkpoint/resume via state.json);
3. on completion, report: ok/failed/rejected counts, the best variant per
   switch by |Δ|, and whether any delta is above the 3% noise floor.
Never edit COMPAT_RULES or the noise floor at runtime. If a variant OOMs,
record and continue; if ALL variants fail, stop and report the common error
from the log tails.
```

**Invocation template**:

```bash
python3 scripts/run_ab_matrix.py \
  --plan <workdir>/matrix_plan.json \
  --baseline <workdir>/baseline/baseline.json \
  --out-dir <workdir>/ab/ --runs-per-variant 2
```

**Input contract**: `matrix_plan.json` =
`{base_server_config: <server_config>, workload?: <must equal baseline's>,
variants: [{name: str, overrides: {launch_args: {flag: value|null},
launch_flags_off: [flag], env: {k:v}, env_unset: [k]}}]}`.

**Output contract**: `ab/ab_result.json` (schema §D) + per-variant logs +
`ab/state.json` (checkpoint).

**Failure semantics**: workload mismatch → hard exit before any GPU time;
per-variant failure → recorded, matrix continues; global abort only when every
variant failed (common-cause investigation).

---

## Orchestrator note

The SKILL.md workflow is the orchestrator: it runs Phase 0 checks, then
`baseline-collector` → `bottleneck-analyzer` → `matrix-runner` →
`gen_report.py` (no agent needed for report synthesis — it is deterministic).
Anti-forgetting checklist for the orchestrator: proxies unset, A3 tp×2,
TASK_QUEUE/graph conflict, draft-model presence, workload spec frozen, outputs
out of `/tmp`.
