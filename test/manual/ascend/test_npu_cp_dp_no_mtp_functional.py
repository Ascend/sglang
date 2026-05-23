import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHSERVING,
    DEEPSEEK_V32_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

MODEL_CONFIG = {
    "model_path": DEEPSEEK_V32_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_BUFFSIZE": "1200",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "TASK_QUEUE_ENABLE": "0",
        "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
        "HCCL_BUFFSIZE": "400",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8",
    },
    "prefill_args": [
        "--nnodes",
        2,
        "--tp",
        32,
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.73,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        1,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
        "--disaggregation-mode",
        "prefill",
        "--disable-cuda-graph",
        "--moe-dense-tp-size",
        1,
        "--enable-nsa-prefill-context-parallel",
        "--nsa-prefill-cp-mode",
        "in-seq-split",
        "--attn-cp-size",
        16,
        "--dp-size",
        2,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
    ],
    "decode_args": [
        "--nnodes",
        2,
        "--tp",
        32,
        "--dp",
        8,
        "--ep",
        32,
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.79,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        32,
        "--cuda-graph-max-bs",
        4,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
        "--disaggregation-mode",
        "decode",
    ],
    "router_args": [
        "--mini-lb",
    ],
}


class TestDeepSeekV32W8A8CpDpNoMtpFunctional(TestAscendPerfMultiNodePdSepTestCaseBase):
    """Verify functional correctness with CP and DP enabled, MTP disabled

    [Test Category] Functional
    [Test Target] Inference Correctness (CP + DP, No MTP)
    --enable-nsa-prefill-context-parallel; --nsa-prefill-cp-mode
    """

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHSERVING
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 65536
    output_len = 1024
    random_range_ratio = 1
    output_token_throughput = 0

    def test_long_context_inference_with_cp_dp_enabled(self):
        """Verify 64K long-context inference runs correctly with CP and DP enabled, MTP disabled."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()