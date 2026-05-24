import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME, check_role
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    QWEN3_235B_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
    run_aisbench,
)

# ====================== Base Configuration ======================
BASE_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "188416",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DP_ROUND_ROBIN": "1",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "HCCL_BUFFSIZE": "4300",
    "TASK_QUEUE_ENABLE": "2",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "TRANSFORMERS_VERBOSITY": "error",
}

BASE_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DP_ROUND_ROBIN": "1",
    "DP_ROUND_ROBIN": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "65536",
    "HCCL_BUFFSIZE": "800",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "TRANSFORMERS_VERBOSITY": "error",
}

BASE_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--mem-fraction-static",
    0.6,
    "--disable-radix-cache",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--max-running-requests",
    128,
    "--chunked-prefill-size",
    94208,
    "--max-prefill-tokens",
    262144,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--dtype",
    "bfloat16",
]

BASE_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    "2",
    "--tp-size",
    32,
    "--dp-size",
    32,
    "--mem-fraction-static",
    0.83,
    "--max-running-requests",
    768,
    "--quantization",
    "modelslim",
    "--enable-dp-attention",
    "--cuda-graph-bs",
    6,
    8,
    12,
    15,
    18,
    20,
    22,
    24,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--watchdog-timeout",
    9000,
    "--context-length",
    8192,
    "--prefill-round-robin-balance",
    "--enable-dp-lm-head",
    "--tokenizer-worker-num",
    4,
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
]

# ====================== Configurations ======================
MODEL_CONFIG_FUSION_DISABLED = {
    "model_path": QWEN3_235B_W8A8_MODEL_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "router_envs": {"SGLANG_DP_ROUND_ROBIN": "1", "TRANSFORMERS_VERBOSITY": "error"},
    "prefill_args": BASE_PREFILL_ARGS,
    "decode_args": BASE_DECODE_ARGS
    + [
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
    ],
    "router_args": ["--mini-lb"],
}

MODEL_CONFIG_FUSION_ENABLED = {
    "model_path": QWEN3_235B_W8A8_MODEL_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "router_envs": {"SGLANG_DP_ROUND_ROBIN": "1"},
    "prefill_args": BASE_PREFILL_ARGS,
    "decode_args": BASE_DECODE_ARGS
    + [
        "--moe-a2a-backend",
        "ascend_fuseep",
    ],
    "router_args": ["--mini-lb"],
}


class TestQwen235bFusionOperator(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    max_attempts = 3
    model_config = MODEL_CONFIG_FUSION_ENABLED
    backend = "sglang-oai"
    dataset_name = "random"
    max_concurrency = 860
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 35.8
    output_token_throughput = 11570

    def test_throughput(self):
        self.run_throughput()

    # benchmark_tool = BENCHMARK_TOOL_DEFAULT
    # aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    # max_attempts = 3
    # backend = "sglang-oai"
    # dataset_name = "random"
    # max_concurrency = 128
    # num_prompts = int(max_concurrency) * 4
    # input_len = 3500
    # output_len = 1500
    # random_range_ratio = 1

    # # Configurable parameters for test
    # num_test_runs = 3  # Number of runs to average for each configuration
    # model_layers = 94  # Number of layers in QWEN3 235B model
    # expected_per_layer_reduction_ms = 0.05  # 50 microseconds = 0.05 milliseconds

    # # Initialize model_config to avoid None reference issues
    # model_config = MODEL_CONFIG_FUSION_DISABLED

    # @check_role(allowed_roles=["router"])
    # def run_throughput(self):
    #     # Use class attribute explicitly to ensure consistency
    #     model_config = TestQwen235bFusionOperator.model_config
    #     metrics = run_aisbench(
    #         host=self.host,
    #         port=str(self.port),
    #         model_path=model_config.get("model_path"),
    #         dataset_type=self.aisbench_dataset_type,
    #         dataset_path=self.aisbench_dataset_path,
    #         input_len=self.input_len,
    #         output_len=self.output_len,
    #         max_concurrency=self.max_concurrency,
    #         num_prompts=self.num_prompts,
    #         image_resolution=self.image_resolution,
    #         random_range_ratio=self.random_range_ratio,
    #         prefix_hit_rate=self.prefix_hit_rate,
    #         aisbench_request_rate=self.aisbench_request_rate,
    #         aisbench_repeat_rate=self.aisbench_repeat_rate,
    #         dp=self.dp,
    #         generation_kwargs=self.generation_kwargs,
    #     )
    #     return metrics

    # def run_test_with_config(self, config):
    #     """
    #     Run performance test multiple times with the given configuration and return average TPOT.

    #     Args:
    #         config: Model configuration dictionary

    #     Returns:
    #         Average TPOT (time per output token) in milliseconds
    #     """
    #     # Set class attribute directly to ensure router thread can access it
    #     TestQwen235bFusionOperator.model_config = config
    #     tpot_values = []

    #     try:
    #         self.start_pd_server()
    #         self.start_router_server()

    #         for run_idx in range(self.num_test_runs):
    #             metrics = self.run_throughput()
    #             tpot_ms = metrics.get("tpot", 0.0)
    #             tpot_values.append(tpot_ms)
    #             print(f"Run {run_idx + 1}/{self.num_test_runs} - TPOT: {tpot_ms}ms")

    #         if not tpot_values:
    #             return 0.0
    #         return sum(tpot_values) / len(tpot_values)

    #     finally:
    #         self.stop_sglang_thread()

    # def test_fusion_operator_latency_reduction(self):
    #     """
    #     Test that enabling fusion operator reduces per-layer computation latency by at least 50us.

    #     TPOT (Time Per Output Token) is measured in milliseconds.
    #     Per-layer latency reduction = (TPOT_disabled - TPOT_enabled) / number_of_layers
    #     Target: per-layer reduction >= 50us = 0.05ms
    #     """

    #     # # Test without fusion operator (average over num_test_runs)
    #     # print("Testing WITHOUT fusion operator...")
    #     # tpot_disabled_avg = self.run_test_with_config(MODEL_CONFIG_FUSION_DISABLED)
    #     # print(f"Average TPOT (disabled): {tpot_disabled_avg}ms")

    #     # Test with fusion operator (average over num_test_runs)
    #     print("\nTesting WITH fusion operator...")
    #     tpot_enabled_avg = self.run_test_with_config(MODEL_CONFIG_FUSION_ENABLED)
    #     print(f"Average TPOT (enabled): {tpot_enabled_avg}ms")

    #     # # Calculate per-layer latency reduction
    #     # total_latency_reduction_ms = tpot_disabled_avg - tpot_enabled_avg
    #     # per_layer_reduction_ms = total_latency_reduction_ms / self.model_layers
    #     #
    #     # print(f"\nTotal TPOT reduction: {total_latency_reduction_ms}ms")
    #     # print(
    #     #     f"Per-layer reduction: {per_layer_reduction_ms}ms (target: {self.expected_per_layer_reduction_ms}ms)"
    #     # )
    #     #
    #     # # Verify per-layer latency reduction meets the requirement
    #     # self.assertGreaterEqual(
    #     #     per_layer_reduction_ms,
    #     #     self.expected_per_layer_reduction_ms,
    #     #     msg=f"Per-layer latency reduction {per_layer_reduction_ms}ms is less than expected {self.expected_per_layer_reduction_ms}ms. "
    #     #     f"TPOT (disabled avg): {tpot_disabled_avg}ms, TPOT (enabled avg): {tpot_enabled_avg}ms, "
    #     #     f"Total reduction: {total_latency_reduction_ms}ms, Layers: {self.model_layers}",
    #     # )


if __name__ == "__main__":
    unittest.main()
