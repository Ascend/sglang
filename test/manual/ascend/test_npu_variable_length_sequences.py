import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)

MODEL_CONFIG = {
    "model_path": DEEPSEEK_R1_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "HCCL_BUFFSIZE": "2800",
        "HAS_INDEX_K": "1",
        "SGLANG_DEEPEP_BF16_DISPATCH": "0",
        "SGLANG_NPU_USE_MLAPO": "0",
        "SGLANG_USE_AG_AFTER_QLORA": "0",
        "USE_MULTI_STREAM": "1",
        "ENABLE_MOE_NZ": "1",
        "PROFILING_MODE": "dynamic",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "HCCL_BUFFSIZE": "1024",
        "HAS_INDEX_K": "1",
        "SGLANG_DEEPEP_BF16_DISPATCH": "0",
        "SGLANG_NPU_USE_MLAPO": "0",
        "SGLANG_NPU_USE_MLAPROLOG": "0",
        "USE_MULTI_STREAM": "1",
        "ENABLE_FUSED_MOE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "TASK_QUEUE_ENABLE": "0",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "router_envs": {
        "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "prefill_args": [
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.8,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        16,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        327680,
        "--max-prefill-tokens",
        68000,
        "--max-total-tokens",
        68000,
        "--context-length",
        68000,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--disable-cuda-graph",
        "--disaggregation-transfer-backend",
        "ascend",
    ],
    "decode_args": [
        "--nnodes",
        "2",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.8,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        128,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        262144,
        "--max-prefill-tokens",
        68000,
        "--context-length",
        68000,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--disaggregation-transfer-backend",
        "ascend",
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--cuda-graph-max-bs",
        32,
        "--prefill-round-robin-balance",
        "--load-balance-method",
        ROUND_ROBIN,
    ],
    "router_args": [
        "--pd-disaggregation",
        "--prefill-policy",
        "bucket",
        "--balance-rel-threshold",
        1.0001,
        "--balance-abs-threshold",
        32,
        "--bucket-adjust-interval-secs",
        5,
    ],
}


class TestManualDeploy(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 40
    max_concurrency = 2048
    num_prompts = 2048
    input_len = 300
    output_len = 20
    random_range_ratio = 1
    host = "192.168.0.60"
    port = 6699

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
