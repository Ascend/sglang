# NPU VLM input diagnostics

This temporary reproduction workflow runs B100 on `linux-aarch64-a3-2` using
the container's installed SGLang and the original VLM test assertions.

Four separate processes run sequentially on the same allocation:

1. Independent Qwen raw-image baseline without internal hooks.
2. All three original Qwen input paths with downstream observations.
3. Independent Qwen raw-image case with the same observations.
4. Replay that raw case's captured vision embedding through the original
   precomputed-input helper, with the same assertions.

The preceding run already compared the full nine cases with and without hooks,
and reproduced the same failure in independent Qwen. This follow-up focuses on
the failing raw-image path and its two passing processor/embedding controls.

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
