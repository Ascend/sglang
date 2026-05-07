import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "STREAMS_PER_DEVICE": "32",
    "ENABLE_PROFILING": "1",
    "PROFILING_BS": "30",
    "PROFILING_step": "8",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "2500",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "64",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "640",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--mem-fraction-static",
    0.43,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    128,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    58000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--tokenizer-worker-num",
    16,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
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
    "--disable-radix-cache",
]

DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--cuda-graph-bs",
    8,
    16,
    24,
    "--mem-fraction-static",
    0.6,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    96,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    65536,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--deepep-mode",
    "low_latency",
    "--tokenizer-worker-num",
    16,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
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
    "--disable-radix-cache",
]

MODEL_CONFIG = {
    "model_path": MINIMAX_M2_5_W8A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ["--policy", "round_robin", "--mini-lb"],
    "router_envs": {},
}


class TestNPUMiniMaxM2_5_W8A8_1P1D_24P_GPQA(
    TestAscendAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for MiniMax-M2.5-w8a8 1p1d_24p on GPQA"""

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    accuracy = 0.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    max_concurrency = 128
    output_len = 1024

    def test_npu_minimax_m2_5_w8a8_1p1d_24p_gpqa(self):
        """Run NPU accuracy test for MiniMax-M2.5-w8a8 1p1d_24p on GPQA"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
