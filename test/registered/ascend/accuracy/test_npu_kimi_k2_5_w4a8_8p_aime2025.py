import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    KIMI_K2_5_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-16-npu-a3",
    nightly=True,
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_BUFFSIZE": "1200",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

OTHER_ARGS = [
    "--skip-server-warmup",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.8,
    "--max-running-requests",
    128,
    "--context-length",
    260000,
    "--chunked-prefill-size",
    16384,
    "--max-prefill-tokens",
    16384,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--cuda-graph-bs",
    4,
    8,
    16,
    32,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
]


class TestNPUKimiK2_5AIME25(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for Kimi-K2.5-w4a8 on AIME 2025"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = KIMI_K2_5_W4A8_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    accuracy = 0.3
    dataset_type = "aime2025"
    dataset_name = "aime2025_gen"
    batch_size = 64
    generation_kwargs = "dict(temperature=1.0, top_p=0.95)"
    max_out_len = 256000

    def test_npu_kimi_k2_5_aime25(self):
        """Run NPU accuracy test for Kimi-K2.5 on AIME 2025"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
