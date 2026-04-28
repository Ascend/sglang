import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    QWEN3_VL_235B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_VL_235B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "1600",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_BS": "2",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "SGLANG_NPU_PROFILING_STEP": "50",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_DEEPEP_BF16_DISPATCH": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
}

QWEN3_VL_235B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--max-prefill-tokens",
    30720,
    "--chunked-prefill-size",
    30720,
    "--mem-fraction-static",
    0.89,
    "--mm-attention-backend",
    "ascend_attn",
    "--max-running-requests",
    512,
    "--disable-radix-cache",
    "--cuda-graph-bs",
    32,
    "--enable-multimodal",
    "--mm-enable-dp-encoder",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--sampling-backend",
    "ascend",
]


class TestQwenVl235B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = AISBENCHMARK
    aisbench_dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    # aisbench_dataset_path = (
    #     "/root/.cache/modelscope/hub/datasets/sglang_test/1024x1024_0.jsonl"
    # )
    model = QWEN3_VL_235B_MODEL_PATH
    other_args = QWEN3_VL_235B_OTHER_ARGS
    envs = QWEN3_VL_235B_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    image_resolution = "1024x1024"
    image_count = 1
    request_rate = 512
    max_concurrency = 512
    num_prompts = 2048
    input_len = 1
    output_len = 1024
    random_range_ratio = 1
    seed = 1000
    tpot = 91
    output_token_throughput = 4500

    def test_qwen3_vl_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
