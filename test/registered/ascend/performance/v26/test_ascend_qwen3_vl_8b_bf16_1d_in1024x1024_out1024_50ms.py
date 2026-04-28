import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    QWEN3_VL_8B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_VL_8B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "1",
}

QWEN3_VL_8B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    1,
    "--max-prefill-tokens",
    40960,
    "--chunked-prefill-size",
    40960,
    "--mem-fraction-static",
    0.86,
    "--max-running-requests",
    512,
    "--disable-radix-cache",
    "--cuda-graph-bs",
    10,
    20,
    40,
    80,
    85,
    88,
    90,
    96,
    100,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--context-length",
    32768,
]


class TestQwenVl8B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = AISBENCHMARK
    aisbench_dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    # aisbench_dataset_path = (
    #     "/root/.cache/modelscope/hub/datasets/sglang_test/1024x1024_0.jsonl"
    # )
    model = QWEN3_VL_8B_MODEL_PATH
    other_args = QWEN3_VL_8B_OTHER_ARGS
    envs = QWEN3_VL_8B_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    dataset_path = None
    image_resolution = "1024x1024"
    image_count = 1
    max_concurrency = 96
    num_prompts = 384
    input_len = 1
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 1520

    def test_qwen3_vl_8b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
