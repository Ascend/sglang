"""NPU device-time measurement for operator performance tests.

Mirrors sgl-kernel-npu's ``tests/python/utils/performance.py``: compile/warm up
before collecting tasks, profile with a fixed warmup/active schedule, and read
the exported ``op_summary`` CSV instead of timing around
``torch.npu.synchronize()``. Wall-clock timing includes host-side launch,
dispatch and first-call initialisation, which dominates small kernels and makes
single-shot comparisons meaningless.
"""

import csv
import math
import tempfile
from pathlib import Path


def read_device_time_us(directory, active):
    """Sum task durations per call; ``active`` must be a positive integer."""
    if isinstance(active, bool) or not isinstance(active, int) or active <= 0:
        raise ValueError("active must be a positive integer")
    directory = Path(directory)
    files = sorted(directory.rglob("op_summary*.csv"))
    column = "Task Duration(us)"
    if not files:
        files = sorted(directory.rglob("op_statistic.csv"))
        column = "Total Time(us)"
    if not files:
        raise RuntimeError(f"No NPU profiler CSV found under {directory}")
    total = 0.0
    rows = 0
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if column not in (reader.fieldnames or []):
                raise RuntimeError(f"Missing {column!r} in {path}")
            for row in reader:
                duration = float(row[column])
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError(f"Invalid duration in {path}: {duration}")
                total += duration
                rows += 1
    if not rows or total <= 0:
        raise RuntimeError("Profiler returned no positive device task duration")
    return total / active


def measure_device_time_us(executor, *, warmup=5, active=5):
    """Return the average NPU task duration (us) per ``executor`` call."""
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup <= 0:
        raise ValueError("warmup must be a positive integer")
    if isinstance(active, bool) or not isinstance(active, int) or active <= 0:
        raise ValueError("active must be a positive integer")

    import torch
    import torch_npu

    with tempfile.TemporaryDirectory(prefix="sglang-npu-prof-") as directory:
        # Compile / first-call initialisation is excluded from the samples.
        executor()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=warmup,
                active=active,
                repeat=1,
                skip_first=0,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(directory),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            ),
        ) as profiler:
            for _ in range(warmup + active):
                torch.npu.synchronize()
                executor()
                torch.npu.synchronize()
                profiler.step()
        return read_device_time_us(directory, active)
