import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_6_27B_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

QWEN3_6_27B_1P_ACC_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "130",
}

QWEN3_6_27B_1P_ACC_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    60000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    54,
    "--max-mamba-cache-size",
    64,
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs",
    2,
    8,
    16,
    32,
    45,
    54,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUQwen3_6_27B_1P_GPQA(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3.6-27B-W8A8 1p on GPQA"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    other_args = QWEN3_6_27B_1P_ACC_OTHER_ARGS
    envs = QWEN3_6_27B_1P_ACC_ENVS
    accuracy = 87.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    output_len = 81920
    max_concurrency = 8
    generation_kwargs = (
        "dict(temperature=0.7, top_p=0.95, seed=None, repetition_penalty=1.0)"
    )

    def test_npu_qwen3_6_27b_1p_gpqa(self):
        """Run NPU accuracy test for Qwen3.6-27B-W8A8 on GPQA"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
