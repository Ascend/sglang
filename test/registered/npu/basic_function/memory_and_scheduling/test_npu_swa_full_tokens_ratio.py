"""Tests for --swa-full-tokens-ratio parameter.

The parameter controls: SWA pool tokens = Full pool tokens * ratio.
Only effective on Hybrid SWA models (DeepSeek V4, MiMo, Inkling, etc.).

Two test strategies:
- Unit test: test/registered/unit/model_executor/test_pool_configurator.py (CPU only)
- Server test: launch a real Hybrid SWA model, verify inference and print pool sizes
"""

import os
import re
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MIMO_V2_FLASH_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

_MIMO_BASE_ARGS = [
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--mem-fraction-static",
    0.85,
    "--reasoning-parser",
    "mimo",
    "--attention-backend",
    "ascend",
    "--disable-piecewise-cuda-graph",
    "--base-gpu-id",
    0,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--quantization",
    "modelslim",
    "--skip-server-warmup",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--swa-full-tokens-ratio",
    0.3,
    "--enable-multi-layer-eagle",
    "--speculative-draft-model-quantization",
    "unquant",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]

_MIMO_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_USE_FIA": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "800",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_BF16_DISPATCH": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
}

_POOL_LOG_PATTERN = re.compile(
    r"Use sliding window memory pool. full_layer_tokens=(\d+).*swa_layer_tokens=(\d+)"
)


class TestSwaFullTokensRatioServer(TestNpuAccuracyTestCaseBase):
    """Verify --swa-full-tokens-ratio on a real Hybrid SWA model (MiMo V2 Flash).

    Launches the server, sends an inference request, and prints the
    Full and SWA pool sizes from server logs.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    [Scenario] S2: parameter accepted on Hybrid SWA model, pool sizes printed
    """

    model = MIMO_V2_FLASH_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    other_args = _MIMO_BASE_ARGS
    envs = _MIMO_ENVS
    accuracy = 0.70
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 2048,
        "temperature": 1.0,
    }
    max_concurrency = 64
    output_len = 2048

    @classmethod
    def setUpClass(cls):
        cls.out_log_file_name = "./tmp_out_log.txt"
        cls.err_log_file_name = "./tmp_err_log.txt"
        cls.out_log_file = open(cls.out_log_file_name, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.err_log_file_name, "w+", encoding="utf-8")

        import sglang.test.ascend.e2e.test_npu_accuracy_utils as base_module
        original_popen = base_module.popen_launch_server

        def _patched_popen(*args, **kwargs):
            kwargs["return_stdout_stderr"] = (cls.out_log_file, cls.err_log_file)
            return original_popen(*args, **kwargs)

        base_module.popen_launch_server = _patched_popen
        try:
            super().setUpClass()
        finally:
            base_module.popen_launch_server = original_popen

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove(cls.out_log_file_name)
        os.remove(cls.err_log_file_name)

    def _capture_pool_sizes(self, stdout):
        """Extract full/swa pool sizes from server stdout."""
        for line in stdout.splitlines():
            m = _POOL_LOG_PATTERN.search(line)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None, None

    def test_launch_and_print_pool_sizes(self):
        """S2: Launch MiMo V2 Flash, infer, and print Full/SWA pool sizes."""
        self.run_accuracy()

        self.out_log_file.seek(0)
        stdout = self.out_log_file.read()
        self.assertTrue(len(stdout) > 0)
        full, swa = self._capture_pool_sizes(stdout)

        self.assertIsNotNone(
            full,
            "Pool size log not found in server stdout. "
            "Look for 'Use sliding window memory pool' in server logs.",
        )
        self.assertIsNotNone(
            swa,
            "Pool size log not found in server stdout. "
            "Look for 'Use sliding window memory pool' in server logs.",
        )
        ratio = swa / full
        print(
            f"\n  [SWA Pool Info] full={full}, swa={swa}, "
            f"ratio={ratio:.4f} (config=0.95)"
        )
        self.assertAlmostEqual(
            ratio,
            0.95,
            delta=0.01,
            msg=f"SWA/Full ratio {ratio:.4f} deviates from config 0.95",
        )


if __name__ == "__main__":
    unittest.main()
