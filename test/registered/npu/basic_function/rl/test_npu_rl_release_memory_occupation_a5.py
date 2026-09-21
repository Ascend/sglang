"""
A5-only smoke for --enable-memory-saver and --enable-weights-cpu-backup.

The A3 test in test_npu_rl_release_memory_occupation.py maps npu-smi cards as
2 dies × 64 GB. A5 is 1 card = 1 die, 96 GB, so those HBM asserts fire before
the CPU-backup equality check. This file keeps the original A3 test unchanged.

Note:
  - the NPU-customized torch_memory_saver is required.
    pip install torch_memory_saver-xxx.whl
"""

import logging
import multiprocessing
import os
import re
import subprocess
import unittest

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase

_LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(_LOG_FMT))
logger.addHandler(_handler)
logger.propagate = False

# 1B weights ~2 GB plus the mem_fraction_static=0.6 KV pool on a 96 GB die.
_MIN_DELTA_SMI_RELEASE_ALL_MB = 10000
_A5_HBM_CAPACITY_MB = 96 * 1024


def _npu_smi_mem_mb_a5() -> float:
    """Sum HBM usage for A5: each ASCEND_RT_VISIBLE_DEVICES id is one die/card."""
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
        "ASCEND_VISIBLE_DEVICES"
    )
    if not visible:
        raise RuntimeError(
            "Neither ASCEND_RT_VISIBLE_DEVICES nor ASCEND_VISIBLE_DEVICES is set"
        )
    npu_ids = [int(x.strip()) for x in visible.split(",") if x.strip()]
    if not npu_ids:
        raise RuntimeError("No valid NPU IDs found in %s" % visible)

    logger.info("Tracking A5 dies: %s", npu_ids)
    total = 0.0

    for npu_id in npu_ids:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "usages", "-i", str(npu_id)],
            timeout=10,
            text=True,
        )
        cap_mb = 0.0
        rate_pct = 0.0
        used_mb = None
        for line in out.splitlines():
            m = re.match(r"^\s*HBM Capacity\(MB\)\s*:\s*(\d+)", line)
            if m:
                cap_mb = float(m.group(1))
                continue
            m = re.match(r"^\s*HBM Usage Rate\(%\)\s*:\s*(\d+)", line)
            if m:
                rate_pct = float(m.group(1))
                continue
            m = re.match(r"^\s*HBM-Usage\(MB\)\s*:\s*(\d+)", line)
            if m:
                used_mb = float(m.group(1))

        if used_mb is None:
            used_mb = cap_mb * rate_pct / 100.0
        logger.info(
            "A5 NPU %d: capacity=%.0f MB (expect ~%.0f), used=%.0f MB (%.0f%%)",
            npu_id,
            cap_mb,
            _A5_HBM_CAPACITY_MB,
            used_mb,
            rate_pct,
        )
        total += used_mb

    logger.info("A5 total HBM used: %.0f MB", total)
    return total


def _assert_mem_decreased(mem_before, mem_func, min_delta, tag):
    mem_after = mem_func()
    delta = mem_before - mem_after
    assert delta > min_delta, (
        f"[{tag}] Expected mem decrease > {min_delta} MB, "
        f"got {delta:.0f} MB ({mem_before:.0f} → {mem_after:.0f})"
    )
    return mem_after


def _assert_mem_increased(mem_before, mem_func, min_delta, tag):
    mem_after = mem_func()
    delta = mem_after - mem_before
    assert delta > min_delta, (
        f"[{tag}] Expected mem increase > {min_delta} MB, "
        f"got {delta:.0f} MB ({mem_before:.0f} → {mem_after:.0f})"
    )
    return mem_after


class TestReleaseMemoryOccupationNPUA5(CustomTestCase):
    """A5 TP=1 smoke: memory saver + weights CPU backup.

    [Test Category] Parameter
    [Test Target] --enable-memory-saver; --enable-weights-cpu-backup
    """

    @classmethod
    def setUpClass(cls):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        cls._saved_npu_alloc_conf = os.environ.pop("PYTORCH_NPU_ALLOC_CONF", None)
        cls._engine_model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        assert os.path.isdir(cls._engine_model), f"Model not found: {cls._engine_model}"

    @classmethod
    def tearDownClass(cls):
        if cls._saved_npu_alloc_conf is not None:
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = cls._saved_npu_alloc_conf

    def test_a5_memory_saver_and_weights_cpu_backup(self):
        import sglang as sgl

        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}
        engine = sgl.Engine(
            model_path=self._engine_model,
            random_seed=42,
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            mem_fraction_static=0.6,
            tp_size=1,
        )
        try:
            baseline = engine.generate(prompt, sampling_params)["text"]
            self.assertIsNotNone(baseline)
            self.assertGreater(len(baseline), 0)
            logger.info("[A5-CB] baseline: %s", baseline)

            mem_before = _npu_smi_mem_mb_a5()
            engine.release_memory_occupation()
            mem_release = _assert_mem_decreased(
                mem_before,
                _npu_smi_mem_mb_a5,
                _MIN_DELTA_SMI_RELEASE_ALL_MB,
                "a5-cpu-backup",
            )
            logger.info(
                "[A5-CB] release: %.0f→%.0f MB", mem_before, mem_release
            )

            engine.resume_memory_occupation()
            mem_resume = _assert_mem_increased(
                mem_release,
                _npu_smi_mem_mb_a5,
                _MIN_DELTA_SMI_RELEASE_ALL_MB,
                "a5-cb-resume",
            )
            logger.info("[A5-CB] resume: %.0f→%.0f MB", mem_release, mem_resume)

            result = engine.generate(prompt, sampling_params)["text"]
            self.assertEqual(
                baseline,
                result,
                "CPU backup must preserve weights; output unchanged",
            )
            logger.info("[A5-CB] after resume: %s", result)
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
