import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    KIMI_K2_5_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

KIMI_K2_5_TWO_NODE_MIX_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "3072",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "88",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
}

KIMI_K2_5_TWO_NODE_MIX_OTHER_ARGS = [
    "--skip-server-warmup",
    "--trust-remote-code",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--mem-fraction-static",
    0.72,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    512,
    "--context-length",
    160000,
    "--chunked-prefill-size",
    132000,
    "--max-prefill-tokens",
    32768,
    "--dp-size",
    32,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-dp-lm-head",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--disable-shared-experts-fusion",
    "--cuda-graph-bs",
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    "--tokenizer-worker-num",
    4,
    "--dtype",
    "bfloat16",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--quantization",
    "modelslim",
]

KIMI_K2_5_TWO_NODE_MIX_MODEL_CONFIG = {
    "model_path": KIMI_K2_5_W4A8_MODEL_PATH,
    "other_args": KIMI_K2_5_TWO_NODE_MIX_OTHER_ARGS,
    "node_envs": KIMI_K2_5_TWO_NODE_MIX_ENVS,
}


class TestNPUKimiK2_5_W4A8_16P_GPQA(TestAscendAccuracyMultiNodePdMixTestCaseBase):
    """Test NPU accuracy for Kimi-K2.5-w4a8 16p PD Mix on GPQA"""

    model_config = KIMI_K2_5_TWO_NODE_MIX_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    accuracy = 0.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    output_len = 8192
    max_concurrency = 128
    num_prompts = 100000

    def test_npu_kimi_k2_5_w4a8_16p_gpqa(self):
        """Run NPU accuracy test for Kimi-K2.5-w4a8 16p PD Mix on GPQA"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
