import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_5_397B_W8A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

QWEN3_5_397B_W8A8_1P_HIGH_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
}

QWEN3_5_397B_W8A8_1P_HIGH_OTHER_ARGS = [
    "--model-path",
    QWEN3_5_397B_W8A8_MODEL_PATH,
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    60000,
    "--max-total-tokens",
    80000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    4,
    "--max-mamba-cache-size",
    60,
    "--mem-fraction-static",
    0.75,
    "--cuda-graph-bs",
    2,
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


class TestNPUQwen3_5_397B_W8A8_1P_In3k5_Out1k5_High_GPQA(
    TestAscendAccuracyTestCaseBase
):
    """Test NPU accuracy for Qwen3.5-397B-W8A8 1p in3k5 out1k5 high throughput on GPQA"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_5_397B_W8A8_MODEL_PATH
    other_args = QWEN3_5_397B_W8A8_1P_HIGH_OTHER_ARGS
    envs = QWEN3_5_397B_W8A8_1P_HIGH_ENVS
    accuracy = 0.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    output_len = 1500
    max_concurrency = 4
    num_prompts = 448
    generation_kwargs = '{"temperature": 0}'

    def test_npu_qwen3_5_397b_w8a8_1p_in3k5_out1k5_high_gpqa(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
