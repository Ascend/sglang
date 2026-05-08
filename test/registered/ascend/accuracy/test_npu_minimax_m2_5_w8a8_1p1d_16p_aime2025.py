import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MINIMAX_M2_5_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="npu-accuracy",
    nightly=True,
)

PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "2500",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "64",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
}

DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "640",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS": "96",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
}

PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--trust-remote-code",
    "--tp-size",
    16,
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
    130000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
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
    "/root/.cache/modelscope/hub/models/Eco-Tech/MiniMax-M2.5-eagle3",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
    "--speculative-draft-model-quantization",
    "unquant",
    "--skip-server-warmup",
]

DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--trust-remote-code",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.76,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    80,
    "--chunked-prefill-size",
    -1,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--deepep-mode",
    "low_latency",
    "--tokenizer-worker-num",
    8,
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
    "/root/.cache/modelscope/hub/models/Eco-Tech/MiniMax-M2.5-eagle3",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
    "--speculative-draft-model-quantization",
    "unquant",
    "--disaggregation-enable-decode-radix-cache",
    "--skip-server-warmup",
    "--cuda-graph-bs",
    2,
    4,
    8,
]

ROUTER_ARGS = [
    "--pd-disaggregation",
    "--policy",
    "round_robin",
    "--mini-lb",
]

ROUTER_ENVS = {}

MODEL_CONFIG = {
    "model_path": MINIMAX_M2_5_W8A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ROUTER_ARGS,
    "router_envs": ROUTER_ENVS,
}


class TestNPUMiniMaxM2_5W8A8_1P1D_16P_AIME2025(TestAscendAccuracyMultiNodePdSepTestCaseBase):
    """MiniMax-M2.5-w8a8 PD Sep 1p1d 16p AIME2025 accuracy test"""

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    accuracy = 0.8
    dataset_type = "aime2025"
    dataset_name = "aime2025"
    output_len = 8192
    max_concurrency = 16
    num_prompts = 30

    def test_npu_minimax_m2_5_w8a8_1p1d_16p_aime2025(self):
        """Run MiniMax-M2.5-w8a8 PD Sep 1p1d 16p AIME2025 accuracy test"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()