import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    QWEN2_5_VL_72B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN2_5_VL_72B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "200",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "1",
}

QWEN2_5_VL_72B_OTHER_ARGS = [
    "--trust-remote-code",
    "--device",
    "npu",
    "--tp-size",
    4,
    "--max-prefill-tokens",
    20480,
    "--chunked-prefill-size",
    20480,
    "--attention-backend",
    "ascend",
    "--mem-fraction-static",
    0.82,
    "--quantization",
    "modelslim",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--max-running-requests",
    96,
    "--cuda-graph-bs",
    96,
    "--tokenizer-worker-num",
    4,
    "--skip-server-warmup",
]


class TestQwen25Vl72B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = AISBENCHMARK
    aisbench_dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    # aisbench_dataset_path = (
    #     "/root/.cache/modelscope/hub/datasets/sglang_test/1920x1080_256.jsonl"
    # )
    model = QWEN2_5_VL_72B_MODEL_PATH
    other_args = QWEN2_5_VL_72B_OTHER_ARGS
    envs = QWEN2_5_VL_72B_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    image_resolution = "1920x1080"
    image_count = 1
    request_rate = 16
    max_concurrency = 96
    num_prompts = 384
    input_len = 256
    output_len = 512
    random_range_ratio = 1
    seed = 1000
    tpot = 140
    output_token_throughput = 450

    def test_qwen25_vl_72b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
