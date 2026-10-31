# Feature-Switch Matrix for Ascend NPU (A/B parameter space)

Human-authoritative reference for the four switches evaluated by this skill.
The machine-checkable subset lives in `run_ab_matrix.py::COMPAT_RULES`; when
this document and the code disagree, fix one of them in the same PR.

Primary source: `docs/hardware-platforms/ascend-npus/optimization/parameter_tuning.mdx`
(official tuning guide incl. the 16×16 feature compatibility matrix) and the
per-model best-practice pages.

## 1. NPU Graph (task wording: "ACLGraph")

Captures the compute graph via `torch.npu.NPUGraph` and replays it to remove
kernel-launch overhead. Analogous to CUDA Graph. Best for decode with stable
batch sizes.

| Knob | Values | Notes |
|---|---|---|
| `--cuda-graph-bs <list>` | e.g. `8 12 24 36 48 …` | cover the batch sizes you actually serve; capture cost grows with the list |
| `--disable-cuda-graph` | flag | the "off" variant; also required on dedicated prefill nodes |
| `--cuda-graph-max-bs-decode N` | e.g. `4`–`64` | cap capture; align with observed serving batch distribution |

A/B variants (P0): `off` vs `on` with a bs list derived from the workload's
`max_concurrency` (e.g. include it and a few lower rungs).

Constraints:
- `--enable-torch-compile` and NPU Graph are **mutually exclusive** (set
  `--disable-cuda-graph` when compiling).
- `TASK_QUEUE_ENABLE=2` interferes with graph-launched tasks → use `0` for
  decode-with-graph, `2` for throughput-critical prefill without graph.
- Profiling with full Python-stack mapping needs `--disable-cuda-graph`;
  graph-capture profiling needs `--enable-profile-cuda-graph` (traces go to
  `SGLANG_TORCH_PROFILER_DIR/graph_capture_profile/`).

## 2. DeepEP (MoE all-to-all)

| Knob | Values | Notes |
|---|---|---|
| `--moe-a2a-backend` | `deepep` \| `ascend_fuseep` \| default | `deepep`: flexible prefill/decode paths; `ascend_fuseep`: general fused dispatch |
| `--deepep-mode` | `normal` \| `low_latency` \| `auto` | `normal`: prefill/large token counts; `low_latency`: decode, graph-compatible; `auto` when unsure |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | `0` (unbounded) … `1024` (hard cap) | must satisfy `max_running_requests * (1 + draft_tokens) <= value`; prefill: `0`, decode: match expected tokens/iter (e.g. `8`) |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT` | `1` \| unset | int8 dispatch quant: less comm volume, small accuracy trade-off; multi-node prefill mostly |

A/B variants (P0, MoE models only): `baseline(no deepep)` vs `deepep+auto`;
P1: `normal` vs `low_latency` under PD-mixed, and int8-quant on/off.

Constraints:
- EPLB (`--enable-eplb`) requires `deepep` (not `ascend_fuseep`).
- DeepEP is MoE-only; do not add to dense-model variants.

## 3. DP-Attention

Attention layers run data-parallel while FFN/MoE stay TP/EP — removes KV-cache
duplication and attention-side all-reduce during small-batch decode.

| Knob | Values | Notes |
|---|---|---|
| `--enable-dp-attention` | flag | the switch itself |
| `--dp-size` | 2, 4, … | must divide card count (remember A3: tp = 2 × cards); leftovers go to TP/EP |
| `--enable-dp-lm-head` | flag | effectively required with DP-attention (avoids logits all-gather) |
| `SGLANG_SCHEDULER_SKIP_ALL_GATHER=1` | env | safe under balanced load (`--load-balance-method round_robin`); drop if ranks look uneven |
| `--moe-dense-tp-size 1` | with DP-attn | avoids pointless all-reduce on dense MLP when each DP shard has full weights |

A/B variants (P0, tp ≥ 2): `off` vs `on (dp=2, lm-head, round_robin)`.

Constraints: partial (🟠) compatibility with EP / chunked prefill / graphs —
prefer changing one thing at a time; record the combination in the variant name.

## 4. MTP / Speculative decoding

| Knob | Values | Notes |
|---|---|---|
| `--speculative-algorithm` | `NEXTN` (DeepSeek-family MTP heads) \| `EAGLE3` (external draft; Qwen-family) | NEXTN needs no draft weights; EAGLE3 requires `--speculative-draft-model-path` |
| `--speculative-num-steps` | prefill `1`; decode `2–4` | more steps → longer acceptance chains, more latency |
| `--speculative-eagle-topk` | `1` (overlap scheduler) else `4–8` | `1` required with overlap scheduler (default on) |
| `--speculative-num-draft-tokens` | prefill `2`; decode `4–5` | balance against latency budget |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` | env | overlap draft with verification (EAGLE v2/v3) |

A/B variants (P0): `off` vs `on` with the model-family default (NEXTN steps=3
draft=4 / EAGLE3 steps=3 topk=1 draft=4 — mirror the model's best-practice page).
P1: steps ∈ {2,3,4} × draft ∈ {4,5}.

Constraints: 🟠 with NPU Graph and chunked prefill; acceptance rate decides
everything — if acceptance is low, MTP makes TPOT worse; report acceptance
length when available.

## Variant naming convention

`<switch>_<key-knob>[_<p1-tag>]`, e.g. `npugraph_bs64`, `deepep_auto`,
`dpattn_dp2`, `mtp_eagle3_s3d4`. The name must identify the single variable
that differs from baseline (P0) or the interaction being explored (P1).

## Sizing the matrix

P0 (acceptance minimum): one on/off pair per applicable switch — 2 variants for
dense models (Graph, MTP), 4 for MoE models on tp≥2. P1 only explores
interactions where P0 showed |Δ| > noise floor. Every variant reuses the
baseline workload verbatim.
