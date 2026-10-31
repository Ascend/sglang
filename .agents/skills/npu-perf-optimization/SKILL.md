---
name: npu-perf-optimization
description: Systematic performance optimization loop for SGLang on Ascend NPU — collect a reproducible baseline, identify operator-level bottlenecks, run a quantified A/B matrix over NPU feature switches (NPU Graph, DeepEP, DP-Attention, MTP/speculative), and emit a structured report with a recommended launch command. Run with /npu-perf-optimization.
---

# NPU Performance Optimization Loop (Ascend)

Turns "the model runs, now make it fast" into a reproducible pipeline:
**baseline → bottleneck attribution → feature-switch A/B matrix → report + recommended launch command**.

Complements existing skills — do not duplicate them:
- trace capture patterns come from `generate-profile`
- trace analysis comes from `llm-torch-profiler-analysis` (triage tables)
- this skill adds the **orchestration layer**: matrix planning, JSON contracts,
  comparability rules, and recommendation synthesis, specialized for Ascend NPU.

## When to run

- After a model is adapted and serving on Ascend NPU and you need a defensible
  "optimal config" rather than folklore tuning.
- When a deployment regressed and you need before/after numbers per switch.
- NOT for: CUDA/NVIDIA deployments (use upstream tuning guides), or when only a
  single trace analysis is needed (use `llm-torch-profiler-analysis` directly).

## Hard prerequisites (Phase 0 — mandatory checks)

Refuse to proceed (fail fast with a clear message) until every box passes:

- [ ] `source /usr/local/Ascend/ascend-toolkit/set_env.sh` and
      `source /usr/local/Ascend/nnal/atb/set_env.sh` succeeded
- [ ] `npu-smi info` shows the expected hardware; **A3 = 2 dies per card →
      `--tp-size` is 2 × card count**; record `hardware` into every JSON meta
- [ ] model weights exist; if speculative decoding is planned, draft-model
      weights exist too
- [ ] proxies unset: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`;
      `unset ASCEND_LAUNCH_BLOCKING`
- [ ] workdir layout created and writable:
      `<workdir>/{baseline,ab,profiles,reports}/` (never write traces to `/tmp`)
- [ ] output JSON paths planned; schema = `references/json-schemas.md`

## Workflow

### Phase 1 — Baseline (`scripts/collect_baseline.py`)

1. Launch the server from the closest official best-practice config
   (`docs/hardware-platforms/ascend-npus/model-deployment/best-practices/`).
2. Wait for readiness: poll `/health`, and grep server log for
   "The server is fired up and ready to roll!".
3. Sanity guard: 20 short generations; abort baseline if any fail.
4. Run `python -m sglang.bench_serving` with a **frozen workload spec**
   (dataset, lens, concurrency, seed) **3 times**; keep the median.
   Comparability rule: every later A/B variant MUST reuse the identical spec.
5. Emit `baseline/baseline.json` (schema in `references/json-schemas.md`).

### Phase 2 — Bottleneck attribution (`scripts/analyze_profile.py`)

1. `bench_serving --profile --profile-steps 10 --profile-output-dir profiles/`
   (Method C: benchmark + trace in one command; server auto-parses).
2. Read `operator_details.csv` / `step_trace_time.csv` from the parsed dir
   (path printed as `Profiling done. Traces are saved to: <path>`).
3. Emit `bottleneck.json`: top-N operators by share, prefill/decode split,
   attribution ∈ {compute, comm, launch, memory}.
4. Note: full Python-stack visibility requires `--disable-cuda-graph` at launch
   (perf-penalizing — profiling runs only). To profile graph capture itself use
   `--enable-profile-cuda-graph`.

### Phase 3 — Feature-switch A/B matrix (`scripts/run_ab_matrix.py`)

Matrix definition lives in `references/feature-switch-matrix.md`. Summary:

| Switch | Knobs | Models |
|---|---|---|
| NPU Graph | `--cuda-graph-bs` list / `--disable-cuda-graph` / `--cuda-graph-max-bs-decode` | all |
| DeepEP | `--moe-a2a-backend deepep`, `--deepep-mode`, dispatch-tokens env, int8 quant env | MoE only |
| DP-Attention | `--enable-dp-attention`, `--dp-size`, `--enable-dp-lm-head`, skip-all-gather env | TP ≥ 2 |
| MTP / speculative | `--speculative-algorithm` (NEXTN for DeepSeek, EAGLE3 for Qwen), steps/topk/draft-tokens | per model |

Rules (enforced by the script, not by memory):
- **Single-variable first**: P0 pass toggles each switch on/off vs baseline;
  P1 pass explores interactions only for switches that moved >3%.
- **Compatibility gate**: reject combos violating the official feature matrix
  (e.g. torch-compile with NPU Graph; EPLB with `ascend_fuseep`;
  `TASK_QUEUE_ENABLE=2` with graph capture) before spending GPU time.
- **Effect size**: |ΔTPOT or Δthroughput| ≤ 3% vs baseline ⇒ report as "no
  significant change"; never word it as an improvement.
- **Circuit breaker per variant**: server fails to start, sanity fails, or
  completed < expected prompts ⇒ record `status: failed` with the log tail and
  continue to the next variant; never abort the whole matrix.
- **Checkpointing**: matrix state file allows resume after interruption; each
  variant's raw bench JSON is kept.

### Phase 4 — Report (`scripts/gen_report.py`)

Emits `reports/report.md` + `reports/report.json` from
baseline/bottleneck/ab artifacts:

- per-switch before/after table (TPOT, TTFT, throughput) — **mandatory section**
- bottleneck summary and which switch addresses it
- **Recommended launch command**: the full, copy-pasteable command (env exports
  + `python -m sglang.launch_server ...`) with expected deltas vs baseline,
  and one-line rationale per switch
- provenance: hardware, sglang commit, workload spec, run dates

## Quality gates (before declaring done)

- [ ] every number in the report traces to a stored JSON artifact (no
      unsourced numbers)
- [ ] recommended config passes the compatibility matrix check in
      `run_ab_matrix.py --validate-only`
- [ ] 3-run median used everywhere; raw runs included
- [ ] report states workload spec verbatim (reproducibility)
- [ ] no claim of improvement below the 3% noise floor

## Sub-agent contracts

This skill is designed to be driven by coding agents. Machine contracts
(prompts, invocation templates, timeouts, failure semantics) for the three
sub-agents (`baseline-collector`, `bottleneck-analyzer`, `matrix-runner`) are in
[references/subagent-contracts.md](references/subagent-contracts.md). All
inter-agent handoff is via JSON files on disk — scripts are idempotent and do
not require an interactive session.

## Deep references (read on demand, don't inline)

- [references/feature-switch-matrix.md](references/feature-switch-matrix.md) —
  full parameter space per switch + compatibility constraints
- [references/json-schemas.md](references/json-schemas.md) —
  baseline / ab_result / bottleneck / report contracts
- [references/report-template.md](references/report-template.md) —
  the report skeleton with the mandatory sections
- [references/subagent-contracts.md](references/subagent-contracts.md) —
  prompts and invocation templates (acceptance criterion #2)
