# NPU VLM input diagnostics

This temporary reproduction workflow runs B100 on `linux-aarch64-a3-2` using
the container's installed SGLang and the original VLM test assertions.

Six separate model processes run sequentially on the same allocation:

1. `baseline`: independent Qwen raw-image case without internal hooks.
2. `qwen`: the raw-image case with downstream observations.
3. `replay`: its captured vision embedding through the original precomputed helper.
4. `core-replay`: the same embedding replay with process resources limited to
   20 Cube / 40 Vector cores and BF16 LM-head projection retained.
5. `core-raw`: the original raw-image case with the same core-resource limit
   and BF16 LM-head projection retained.
6. `fp32-head`: the original raw-image case with SGLang's native
   `enable_fp32_lm_head=True`, using default process core resources.

The final phase changes only the vocabulary projection precision. The two core
phases apply their resource limit before the first model forward, including
warmup, and log effective worker limits and unchanged parameter dtypes. Each
phase preserves the original semantic assertions; all original failures keep
the final job red. This is a diagnostic comparison, not a claim that default CI
is fixed. R11's two core phases failed with identical 124-token responses even
though their first-layer QKV outputs matched host 88.
Earlier runs compared all nine cases, the three Qwen input paths, verified QKV
operator replays with FP64 references and ACL import order. R12 targets the next
observed difference: the first-layer prefill gate/up projection has matching
full input tensors but different full output tensors on core-limited CI and 88.
Generation does not force tokens or replace intermediate model activations,
apart from the explicitly selected captured-vision replay entry point.

In `core-replay`, `VLM_GATE_UP_PROBE=1` captures the first prefill gate/up input
and output on both ranks at their complete shapes: `[2526, 2048]` and
`[2526, 11008]` for this case. The subsequent standalone gate/up replay uses
20 Cube / 40 Vector resources, verifies captured tensor and checkpoint-weight
hashes, compares complete outputs, and profiles the actual kernel path. A last
token slice cannot substitute for this full-prefill matrix multiplication.
These diagnostic snapshots contain activations; model weights stay local.

`replay_qkv.py` remains available for standalone operator analysis with verified
checkpoint shards. Only activations are uploaded; checkpoint weights stay local.
`VLM_DIAGNOSTICS_TENSOR_SCOPE=qkv` retains vision output (needed for replay),
QKV X/Y, model logits and the first divergent LM-head projection snapshots.
Other stages retain their hashes/statistics, reducing duplicate artifact data.

Each phase records responses, stop reasons, output token IDs, software versions,
actual imported source hashes, model configuration/tokenizer hashes, weight
shard sizes, and the cached image files and their file/pixel hashes.
The workflow also captures driver/firmware/CANN versions, device state and plog,
and uploads `npu-vlm-diagnostics-<run>-<attempt>` even when a phase fails.
Any failed phase keeps the final job red; later successes never hide it.

The opt-in `sitecustomize` import hook observes the **installed** image package
in tokenizer/preprocessor child processes without rewriting its source.
For Qwen it records processor/device selection, raw processor tensors,
`image_grid_thw`, image-token spans, multimodal item offsets/features and mRoPE.
CPU tensor snapshots are stored as `.pt`; JSON records include shapes, dtypes,
devices and hashes, plus PID/parent PID, distributed rank and request ID.
Downstream hooks capture tokenizer send, scheduler receive/materialization,
per-item features/offsets/hash/pad values, vision input/output and embedding
scatter output on each rank. Vision blocks record per-layer hashes and statistics
without storing every layer tensor. Loaded vision parameter hashes are collected
once per rank, without storing parameter tensors. A coverage summary in
`result.json` checks all required stages on both ranks, so missing hooks cannot
silently count as a successful diagnostic run.
Final multimodal input IDs may already contain image padding/hash tokens; use
the earlier processor output and item offsets to count original image tokens.

Tensor snapshots copy device data to CPU and may synchronize preprocessing.
Compare the no-hook baseline before interpreting a diagnostic pass. Hooks are
restricted to this workflow by `PYTHONPATH` and `VLM_DIAGNOSTICS_DIR`. Observers
do not alter inference results. The named core and FP32-head phases make only
their explicit interventions; test thresholds and assertions remain unchanged.
Model checkpoint files are not uploaded or hashed in full; their sizes are recorded.
Loaded vision parameters are hashed individually.
Decoder parameter hashes are also recorded per rank. The prefill and first eight
decode forwards record actual position/sequence metadata and logits, per-layer
input/output hashes, and the first decoder layer's norm/QKV/RoPE/attention/MLP
boundaries. This distinguishes vision differences from later model computation.

For a local replay, invoke `run_case.py <original-test-file> --mode full|qwen|qwen-all
--output-dir <directory>`. Optional `--gemma-model`, `--kimi-model`,
`--qwen-model` and `--image-dir` select existing local fixtures. CI uses the
original cache paths and does not silently download or replace input data.

Binary fingerprints include torch_npu Python/shared-library files, torch shared
libraries, CANN lib64 shared libraries, OPP shared libraries and Matmul-related
source/config/object files, plus mounted driver libraries. This is a stated
selection, not a hash of every image file or every OPP kernel. The model workers
also record the actually loaded library hashes and effective JIT/HF32/precision
options after server initialization. Tensor records retain physical format,
strides, contiguity and storage offsets, including first-layer QKV parameters.
The observations themselves do not change these options.

`run_case.py --enable-fp32-lm-head` adds the native Engine flag to its explicitly
selected experiment. Projection observations record the actual flag, input/weight
dtypes and result dtype on both ranks; the new phase requires those observations
in its coverage check. The runtime fingerprint includes the actual chip model,
Cube/Vector core count and L2 cache size.

## Layer and operator comparison

Earlier frozen-input jobs reused immutable artifact
`npu-vlm-diagnostics-38525733697-1`, computed FP64 attention references on CPU,
exported compact probes, and replayed the deployed flash-qlens/paged and GEMM
kernels with default and 20 Cube / 40 Vector resources. Those experiments have
completed; R12 does not repeat the previous frozen-input job. Its operator work
uses the newly captured gate/up boundary described above. Profiler records must
verify actual kernel names and block counts: a successful resource-limit API
call alone does not establish which execution path an operator used.

`VLM_LAYER_PROBE=1` records the first prefill and three decode forwards for every
decoder layer. Hidden and residual snapshots retain the last prefill token;
attention inputs preserve full Q/K/V for exact-shape replay. This functionality
remains available, but R12 does not enable full 36-layer captures by default:
the earlier immutable R8 captures already support that comparison. Existing
per-layer and first-layer-component hashes still identify differences without
repeating the large tensor archive. Compare runs only after checking matching
visual embeddings, weights, and prior output token IDs. Observation hooks do
not change precision or sampling.

