import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_5_122B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-acc-8-npu-a3",
    nightly=True,
)

QWEN3_5_122B_4P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "2400",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
    "TEST_FIV_BACKEND_TRITON": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "ENABLE_PROFILING": "0",
}

QWEN3_5_122B_4P_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    32768,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.87,
    "--max-running-requests",
    112,
    "--cuda-graph-bs-decode",
    1,
    4,
    8,
    14,
    16,
    "--dp",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--ep-size",
    8,
    "--stream-interval",
    64,
    "--schedule-conservativeness",
    0.4,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
]


class TestNPUQwen3_5_122B_4P_AIME2025(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3.5-122B 4p on AIME2025"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_5_122B_MODEL_PATH
    other_args = QWEN3_5_122B_4P_OTHER_ARGS
    envs = QWEN3_5_122B_4P_ENVS
    accuracy = 0.9
    datasets = ["aime25"]
    few_shot_num = 0
    generation_config = {
        "max_tokens": 200000,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "timeout": 60000,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    eval_batch_size = 128
    seed = 1

    def test_npu_qwen3_5_122b_4p_aime2025(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
