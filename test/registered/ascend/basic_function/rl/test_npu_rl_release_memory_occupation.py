"""
NPU RL Memory-Aware Sleep/Wake Tests

Note:
  - The ENV var ASCEND_RT_VISIBLE_DEVICES is needed.
  - the NPU-customized version of torch_memory_saver is needed.
"""

import logging
import multiprocessing
import os
import re
import subprocess
import time
import unittest

import torch

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_2_1B_WEIGHTS_PATH,
    QWEN3_5_9B_WEIGHTS_PATH,
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
    QWEN3_30B_A3B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(
    est_time=600,
    suite="full-2-npu-a3",
    disabled="Depends on the NPU-customized version of torch_memory_saver.",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger(__name__)

# Work around Python 3.11 forkserver × aarch64 × torch_npu signal handler
multiprocessing.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_DELTA_MB_SMALL = 50  # for ~1-2 GB dense models
_MIN_DELTA_MB_MOE = 10000  # for ~60 GB MoE
_MIN_DELTA_MB_GDN = 10000  # for ~9 GB GDN model
_MIN_DELTA_SMI_KV_MB = 1000  # npu-smi kv_cache release (1B model, 60% static pool)
_MIN_DELTA_SMI_W_MB = 500  # npu-smi weights release (~2 GB model)


# NPU memory
def _npu_mem_used_all_mb() -> float:
    """Sum of used memory across all visible NPU devices (for TP>1 tests)."""
    total = 0.0
    for d in range(torch.npu.device_count()):
        torch.npu.synchronize(d)
        free, t = torch.npu.mem_get_info(d)
        total += (t - free) / (1024**2)
    return total


def _npu_smi_mem_mb() -> float:
    """Sum of HBM-Usage(MB) for chips in ASCEND_RT_VISIBLE_DEVICES or ASCEND_VISIBLE_DEVICES only.

    Queries ``npu-smi info -t usages -i <npu_id>`` per NPU card, which
    provides HBM Capacity(MB) and HBM Usage Rate(%).  Only sums chips
    whose physical ID is listed in ASCEND_RT_VISIBLE_DEVICES or ASCEND_VISIBLE_DEVICES.

    Used for staged release tests where torch.npu.mem_get_info is blind
    to sglang's static memory pool.
    """
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
        "ASCEND_VISIBLE_DEVICES", ""
    )
    target_chips = set(int(x.strip()) for x in visible.split(",") if x.strip())
    if not target_chips:
        raise RuntimeError(
            "ASCEND_RT_VISIBLE_DEVICES or ASCEND_VISIBLE_DEVICES must be set for npu-smi memory tracking"
        )

    logger.info("Tracking chips: %s", target_chips)

    # Each NPU card exposes 2 chips → card_id = chip_phy_id // 2
    npu_ids = set(ch // 2 for ch in target_chips)
    total = 0.0

    for npu_id in sorted(npu_ids):
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "usages", "-i", str(npu_id)],
            timeout=10,
            text=True,
        )
        # Per-chip metrics appear before Chip ID; buffer them, flush on Chip ID.
        cap_mb = 0.0
        rate_pct = 0.0
        for line in out.splitlines():
            m = re.match(r"^\s*HBM Capacity\(MB\)\s*:\s*(\d+)", line)
            if m:
                cap_mb = float(m.group(1))
                continue
            m = re.match(r"^\s*HBM Usage Rate\(%\)\s*:\s*(\d+)", line)
            if m:
                rate_pct = float(m.group(1))
                continue
            m = re.match(r"^\s*Chip ID\s*:\s*(\d+)", line)
            if m:
                phy_id = npu_id * 2 + int(m.group(1))
                if phy_id in target_chips:
                    total += cap_mb * rate_pct / 100.0
                    logger.info(
                        "Chip ID %d: %.0f MB HBM, %.0f%% used",
                        phy_id,
                        cap_mb,
                        rate_pct,
                    )
                cap_mb = 0.0
                rate_pct = 0.0

        logger.info("NPU %d: %.0f MB HBM used", npu_id, total)

    return total


def _assert_mem_decreased(before, after, tag, min_delta):
    delta = before - after
    assert delta > min_delta, (
        f"[{tag}] Expected mem decrease > {min_delta} MB, "
        f"got {delta:.0f} MB ({before:.0f} → {after:.0f})"
    )


def _assert_mem_increased(before, after, tag, min_delta):
    delta = after - before
    assert delta > min_delta, (
        f"[{tag}] Expected mem increase > {min_delta} MB, "
        f"got {delta:.0f} MB ({before:.0f} → {after:.0f})"
    )


def _wait_mem_decreased(mem_before, mem_func, min_delta, max_retries=10, interval=5):
    """Loop-wait for memory to decrease by min_delta, checking every interval seconds.

    Memory may not be released immediately on NPU, so poll up to max_retries times.
    Returns the final memory reading.
    """
    for i in range(max_retries):
        mem_after = mem_func()
        if mem_before - mem_after > min_delta:
            return mem_after
        if i < max_retries - 1:
            time.sleep(interval)
    return mem_func()


def _wait_mem_increased(mem_before, mem_func, min_delta, max_retries=10, interval=5):
    """Loop-wait for memory to increase by min_delta, checking every interval seconds.

    Memory may not be re-allocated immediately on NPU, so poll up to max_retries times.
    Returns the final memory reading.
    """
    for i in range(max_retries):
        mem_after = mem_func()
        if mem_after - mem_before > min_delta:
            return mem_after
        if i < max_retries - 1:
            time.sleep(interval)
    return mem_func()


class TestReleaseMemoryOccupationNPU(CustomTestCase):
    """Test NPU release memory occupation.

    [Test Category] RL Memory Release/Resume
    [Test Target] POST /release_memory_occupation, POST /resume_memory_occupation,
                  POST /update_weights_from_tensor, /generate
    """

    @classmethod
    def setUpClass(cls):
        cls._engine_model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        assert os.path.isdir(cls._engine_model), f"Model not found: {cls._engine_model}"

    def _common_test_params(self):
        """Common test parameters."""
        return {
            "prompt": "Today is a sunny day and I like",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            "expect_output_before_update_weights": " to spend it outdoors. I decided to",
            "expect_output_after_update_weights": " to go for a walk. I like",
            "prompt_moe": "The weather is nice today, and I want to",
            "sampling_params_moe": {"temperature": 0, "max_new_tokens": 16},
            "expect_output_before_update_weights_moe": " go out for a walk. But I have to study for an exam. What",
            "expect_output_after_update_weights_moe": " go to the park. I have a picnic basket with sandwiches, fruit, and",
        }

    def _setup_engine(
        self,
        *,
        model=None,
        mem_fraction_static=0.6,
        tp_size=1,
        enable_weights_cpu_backup=False,
    ):
        import sglang as sgl

        return sgl.Engine(
            model_path=model or self._engine_model,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=mem_fraction_static,
            tp_size=tp_size,
            enable_weights_cpu_backup=enable_weights_cpu_backup,
        )

    def _make_hf_model(self, model_path):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("npu:0")

    def test_npu_rl_release_and_resume_occupation(self):
        """TP=1,2: release all → resume → tensor update → generate."""
        params = self._common_test_params()
        self.assertTrue(
            os.path.isdir(LLAMA_3_2_1B_WEIGHTS_PATH),
            f"Base model not found: {LLAMA_3_2_1B_WEIGHTS_PATH}",
        )

        for tp_size, mem_frac in [(1, 0.6), (2, 0.5)]:
            tag = f"R{tp_size}"
            engine = self._setup_engine(mem_fraction_static=mem_frac, tp_size=tp_size)
            try:
                out = engine.generate(params["prompt"], params["sampling_params"])[
                    "text"
                ]
                self.assertEqual(out, params["expect_output_before_update_weights"])
                logger.info(f"[{tag}] baseline: {out}")

                mem_before = _npu_mem_used_all_mb()
                t0 = time.perf_counter()
                engine.release_memory_occupation()
                mem_after = _wait_mem_decreased(
                    mem_before, _npu_mem_used_all_mb, _MIN_DELTA_MB_SMALL
                )
                _assert_mem_decreased(
                    mem_before, mem_after, f"{tag}-release", _MIN_DELTA_MB_SMALL
                )
                logger.info(
                    f"[{tag}] release: {time.perf_counter()-t0:.1f}s, {mem_before:.0f}→{mem_after:.0f} MB"
                )

                engine.resume_memory_occupation()
                logger.info(
                    f"[{tag}] resume: {time.perf_counter()-t0:.1f}s, mem={_npu_mem_used_all_mb():.0f} MB"
                )

                hf = self._make_hf_model(LLAMA_3_2_1B_WEIGHTS_PATH)
                engine.update_weights_from_tensor(list(hf.named_parameters()))
                del hf
                torch.npu.empty_cache()

                out2 = engine.generate(params["prompt"], params["sampling_params"])[
                    "text"
                ]
                self.assertEqual(out2, params["expect_output_after_update_weights"])
                logger.info(f"[{tag}] after update: {out2}")
            finally:
                engine.shutdown()

    def test_npu_rl_release_and_resume_occupation_with_weights_cpu_backup(self):
        """TP=1: CPU backup preserves output after release+resume (no update)."""
        params = self._common_test_params()
        engine = self._setup_engine(
            mem_fraction_static=0.6, enable_weights_cpu_backup=True
        )
        try:
            baseline = engine.generate(params["prompt"], params["sampling_params"])[
                "text"
            ]
            self.assertEqual(baseline, params["expect_output_before_update_weights"])
            logger.info(f"[CB] baseline: {baseline}")

            mem_before = _npu_mem_used_all_mb()
            engine.release_memory_occupation()
            mem_after = _wait_mem_decreased(
                mem_before, _npu_mem_used_all_mb, _MIN_DELTA_MB_SMALL
            )
            _assert_mem_decreased(
                mem_before, mem_after, "cpu-backup", _MIN_DELTA_MB_SMALL
            )
            logger.info(f"[CB] release: {mem_before:.0f}→{mem_after:.0f} MB")

            engine.resume_memory_occupation()
            result = engine.generate(params["prompt"], params["sampling_params"])[
                "text"
            ]
            self.assertEqual(
                baseline, result, "CPU backup must preserve weights; output unchanged"
            )
            logger.info(f"[CB] after resume: {result}")
        finally:
            engine.shutdown()

    def test_npu_rl_multi_stage_release_and_resume(self):
        """TP=1,2: staged release kv→w, resume w→update→resume kv.

        Uses npu-smi for memory tracking because torch.npu.mem_get_info
        cannot see into sglang's static memory pool on NPU.
        """
        params = self._common_test_params()
        self.assertTrue(
            os.path.isdir(LLAMA_3_2_1B_WEIGHTS_PATH),
            f"Base model not found: {LLAMA_3_2_1B_WEIGHTS_PATH}",
        )

        for tp_size, mem_frac in [(1, 0.6), (2, 0.45)]:
            tag = f"MS-TP{tp_size}"
            engine = self._setup_engine(
                model=self._engine_model, mem_fraction_static=mem_frac, tp_size=tp_size
            )
            try:
                baseline = engine.generate(params["prompt"], params["sampling_params"])[
                    "text"
                ]
                self.assertEqual(
                    baseline, params["expect_output_before_update_weights"]
                )
                logger.info(f"[{tag}] baseline (instruct): {baseline}")

                t0 = time.perf_counter()
                mem0 = _npu_smi_mem_mb()

                # Stage 1: release kv_cache
                engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
                mem1 = _wait_mem_decreased(mem0, _npu_smi_mem_mb, _MIN_DELTA_SMI_KV_MB)
                _assert_mem_decreased(mem0, mem1, f"{tag}-kv", _MIN_DELTA_SMI_KV_MB)

                # Stage 2: release weights
                engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                mem2 = _wait_mem_decreased(mem1, _npu_smi_mem_mb, _MIN_DELTA_SMI_W_MB)
                _assert_mem_decreased(mem1, mem2, f"{tag}-w", _MIN_DELTA_SMI_W_MB)

                logger.info(
                    f"[{tag}] release: {mem0:.0f}→{mem1:.0f}→{mem2:.0f} MB, {time.perf_counter()-t0:.1f}s"
                )

                # Resume weights
                t0 = time.perf_counter()
                engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                mem3 = _wait_mem_increased(mem2, _npu_smi_mem_mb, _MIN_DELTA_SMI_W_MB)
                _assert_mem_increased(
                    mem2, mem3, f"{tag}-resume-w", _MIN_DELTA_SMI_W_MB
                )

                # Load training model, update weights, then destroy it
                hf = self._make_hf_model(LLAMA_3_2_1B_WEIGHTS_PATH)
                engine.update_weights_from_tensor(list(hf.named_parameters()))
                del hf
                torch.npu.empty_cache()

                # Resume kv_cache
                engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
                mem4 = _wait_mem_increased(mem3, _npu_smi_mem_mb, _MIN_DELTA_SMI_KV_MB)
                _assert_mem_increased(
                    mem3, mem4, f"{tag}-resume-kv", _MIN_DELTA_SMI_KV_MB
                )

                logger.info(
                    f"[{tag}] resume+update: {mem2:.0f}→{mem3:.0f}→{mem4:.0f} MB, {time.perf_counter()-t0:.1f}s"
                )

                out = engine.generate(params["prompt"], params["sampling_params"])[
                    "text"
                ]
                self.assertEqual(out, params["expect_output_after_update_weights"])
                logger.info(f"[{tag}] after update: {out}")
            finally:
                engine.shutdown()

    def test_npu_rl_moe_model_release_and_resume(self):
        """MoE (TP=2): release all → resume → tensor update."""
        params = self._common_test_params()
        self.assertTrue(
            os.path.isdir(QWEN3_30B_A3B_WEIGHTS_PATH),
            f"MoE model not found: {QWEN3_30B_A3B_WEIGHTS_PATH}",
        )

        os.environ.setdefault("SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT", "1")
        import sglang as sgl

        engine = sgl.Engine(
            model_path=QWEN3_30B_A3B_WEIGHTS_PATH,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=0.5,
            tp_size=2,
            disable_cuda_graph=True,
        )
        try:
            baseline = engine.generate(
                params["prompt_moe"], params["sampling_params_moe"]
            )["text"]
            self.assertIsNotNone(baseline)
            self.assertGreater(len(baseline), 0)
            logger.info(f"[MoE] baseline: {baseline}")

            # Wait for the scheduler to become fully idle
            time.sleep(3)

            mem_before = _npu_mem_used_all_mb()
            engine.release_memory_occupation()
            mem_after = _wait_mem_decreased(
                mem_before, _npu_mem_used_all_mb, _MIN_DELTA_MB_MOE
            )
            _assert_mem_decreased(
                mem_before, mem_after, "moe-release", _MIN_DELTA_MB_MOE
            )
            logger.info(f"[MoE] release: {mem_before:.0f}→{mem_after:.0f} MB")

            engine.resume_memory_occupation()
            mem_resume = _wait_mem_increased(
                mem_after, _npu_mem_used_all_mb, _MIN_DELTA_MB_MOE
            )
            _assert_mem_increased(
                mem_after, mem_resume, "moe-resume", _MIN_DELTA_MB_MOE
            )
            logger.info(f"[MoE] resume: {mem_resume:.0f} MB")

            # update to instruct variant via disk (avoids ForkingPickler
            # shm exhaustion with 60GB model).
            engine.update_weights_from_disk(QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH)
            torch.npu.empty_cache()

            out = engine.generate(params["prompt_moe"], params["sampling_params_moe"])[
                "text"
            ]
            self.assertIsNotNone(out)
            self.assertGreater(len(out), 0)
            self.assertNotEqual(
                baseline, out, "update_weights_from_disk must change output"
            )
            logger.info(f"[MoE] after update: {out}")
        finally:
            engine.shutdown()

    def test_npu_rl_gdn_model_release_and_resume(self):
        """GDN TP=1: release → resume → update_weights_from_disk → generate."""

        params = self._common_test_params()

        import sglang as sgl

        engine = sgl.Engine(
            model_path=QWEN3_5_9B_WEIGHTS_PATH,
            random_seed=42,
            enable_memory_saver=True,
        )
        try:
            baseline = engine.generate(
                params["prompt_moe"], params["sampling_params_moe"]
            )["text"]
            self.assertIsNotNone(baseline)
            self.assertGreater(len(baseline), 0)
            logger.info(f"[GDN] baseline: {baseline!r}")

            # Wait for the scheduler to become fully idle after generate().
            time.sleep(2)

            mem_before = _npu_mem_used_all_mb()
            engine.release_memory_occupation()
            mem_after = _wait_mem_decreased(
                mem_before, _npu_mem_used_all_mb, _MIN_DELTA_MB_GDN
            )
            _assert_mem_decreased(
                mem_before, mem_after, "disk-release", _MIN_DELTA_MB_GDN
            )
            logger.info(f"[GDN] release: {mem_before:.0f}→{mem_after:.0f} MB")

            engine.resume_memory_occupation()
            logger.info(f"[GDN] resume: {_npu_mem_used_all_mb():.0f} MB")

            engine.update_weights_from_disk(QWEN3_5_9B_WEIGHTS_PATH)
            torch.npu.empty_cache()

            out = engine.generate(params["prompt_moe"], params["sampling_params_moe"])[
                "text"
            ]
            self.assertEqual(
                baseline, out, "update_weights_from_disk must preserve output"
            )
            logger.info(f"[GDN] after update: {out!r}")
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
