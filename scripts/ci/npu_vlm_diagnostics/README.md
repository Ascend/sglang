# NPU VLM input diagnostics

This temporary reproduction workflow runs B100 on `linux-aarch64-a3-2` using
the container's installed SGLang and the original VLM test assertions.

Three separate processes run sequentially on the same allocation:

1. Full nine-case baseline without internal hooks.
2. Full nine cases with Qwen2.5-VL input observations.
3. Independent Qwen raw-image case with the same observations.

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
devices and hashes, plus PID/parent PID and request ID where available.
Final multimodal input IDs may already contain image padding/hash tokens; use
the earlier processor output and item offsets to count original image tokens.

Tensor snapshots copy device data to CPU and may synchronize preprocessing.
Compare the no-hook baseline before interpreting a diagnostic pass. Hooks are
restricted to this workflow by `PYTHONPATH` and `VLM_DIAGNOSTICS_DIR`; no runtime
flags, inference operators, test thresholds or assertions are changed.
Model weights are not uploaded or hashed in full; their sizes are recorded.

For a local replay, invoke `run_case.py <original-test-file> --mode full|qwen
--output-dir <directory>`. Optional `--gemma-model`, `--kimi-model`,
`--qwen-model` and `--image-dir` select existing local fixtures. CI uses the
original cache paths and does not silently download or replace input data.
