# NPU VLM input diagnostics

This temporary reproduction workflow runs B100 on `linux-aarch64-a3-2` using
the container's installed SGLang and the original VLM test assertions.

Four separate processes run sequentially on the same allocation:

1. Independent Qwen raw-image baseline without internal hooks.
2. Independent Qwen raw-image case with downstream observations.
3. Replay its captured vision embedding through the original precomputed helper.
4. The original raw-image case with SGLang's native `enable_fp32_lm_head=True`.

The fourth phase changes only the vocabulary projection precision. Its original
semantic assertions remain unchanged; all original failures keep the final job
red. This is a diagnostic comparison, not a claim that default CI is fixed.
Earlier runs compared all nine cases, the three Qwen input paths, verified QKV
operator replays with FP64 references and ACL import order. This run tests the
near-tied generation logits without forcing tokens or model activations.

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
restricted to this workflow by `PYTHONPATH` and `VLM_DIAGNOSTICS_DIR`; no runtime
flags, inference operators, test thresholds or assertions are changed.
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
No option is changed by these observations.

`run_case.py --enable-fp32-lm-head` adds the native Engine flag to its explicitly
selected experiment. Projection observations record the actual flag, input/weight
dtypes and result dtype on both ranks; the new phase requires those observations
in its coverage check. The runtime fingerprint includes the actual chip model,
Cube/Vector core count and L2 cache size.
# Layer and operator comparison

The frozen-input job reuses artifact `npu-vlm-diagnostics-38525733697-1`, computes
FP64 attention references on CPU, and exports a compact probe archive. It also
replays the deployed flash-qlens/paged kernels at default process resources and
20 Cube / 40 Vector resources. The profiler must verify the actual block count;
a successful resource-limit API call alone does not establish a hardware-path
change. These operator experiments preserve the recorded failing case and do
not enable FP32 LM-head projection.

`VLM_LAYER_PROBE=1` records the first prefill and three decode forwards for every
decoder layer. Hidden and residual snapshots retain the last prefill token;
attention inputs preserve full Q/K/V for exact-shape replay. The raw and replay
CI phases enable these probes and package them as `layer-probes.tar.gz`.
Compare runs only after checking matching visual embeddings, weights, and prior
output token IDs. This instrumentation does not change precision or sampling.

