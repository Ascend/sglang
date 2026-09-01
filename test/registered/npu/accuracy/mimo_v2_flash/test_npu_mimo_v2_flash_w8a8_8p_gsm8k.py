import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import MIMO_V2_FLASH_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-16-npu-a3-test-debug",
    nightly=True,
    disabled="accuracy testcase",
)

MIMO_V2_FLASH_W8A8_8P_ENVS = {
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

MIMO_V2_FLASH_W8A8_8P_OTHER_ARGS = [
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--mem-fraction-static",
    0.85,
    "--swa-full-tokens-ratio",
    0.95,
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


class TestNPUMiMoV2FlashW8A8_8P_GSM8K(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for MiMo-V2-Flash W8A8 8p single-node on GSM8K.

    [Test Category] Accuracy
    [Test Target] MiMo-V2-Flash W8A8
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MIMO_V2_FLASH_MODEL_PATH
    other_args = MIMO_V2_FLASH_W8A8_8P_OTHER_ARGS
    envs = MIMO_V2_FLASH_W8A8_8P_ENVS
    accuracy = 0.70
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 2048,
        "temperature": 1.0,
    }
    max_concurrency = 64
    output_len = 2048

    def test_npu_mimo_v2_flash_w8a8_8p_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
