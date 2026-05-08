import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="npu-accuracy",
    nightly=True,
)

MINIMAX_M2_5_W8A8_4P_GPQA_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "0",
    "HCCL_BUFFSIZE": "1600",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "640",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "64",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_NPU_DEEPEP_USE_FUSED_MOE_DECODE": "1",
    "SGLANG_NPU_FUSEEP_DECODE_ONLY": "1",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "PYTHONPATH": "/root/.cache/modelscope/hub/models/Eco-Tech/MiniMax-M2.5-eagel-model-0318",
    "ENABLE_PROFILING": "0",
    "PROFILING_BS": "28",
    "PROFILING_STAGE": "decode",
    "PROFILING_step": "10",
}

MINIMAX_M2_5_W8A8_4P_GPQA_OTHER_ARGS = [
    "--tp-size",
    8,
    "--enable-dp-attention",
    "--prefill-delayer-max-delay-passes",
    500,
    "--enable-prefill-delayer",
    "--mem-fraction-static",
    0.65,
    "--max-running-requests",
    36,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-token",
    150000,
    "--cuda-graph-bs",
    8,
    16,
    24,
    32,
    40,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--tokenizer-worker-num",
    8,
]


class TestNPUMiniMaxM2_5W8A8_4P_GPQA(TestAscendAccuracyTestCaseBase):
    """MiniMax-M2.5-w8a8 4p (4 cards) GPQA accuracy test"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_W8A8_4P_GPQA_OTHER_ARGS
    envs = MINIMAX_M2_5_W8A8_4P_GPQA_ENVS
    accuracy = 0.5
    dataset_type = "gpqa"
    dataset_name = "gpqa"
    output_len = 1024
    max_concurrency = 16
    num_prompts = 198

    def test_npu_minimax_m2_5_w8a8_4p_gpqa(self):
        """Run MiniMax-M2.5-w8a8 4p GPQA accuracy test"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()