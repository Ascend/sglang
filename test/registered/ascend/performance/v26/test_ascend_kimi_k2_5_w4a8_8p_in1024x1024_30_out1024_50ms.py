import unittest

from python.sglang.test.ascend.e2e.test_npu_performance_utils import AISBENCHMARK
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    KIMI_K2_5_W4A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

KIMI_K2_5_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "2100",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

KIMI_K2_5_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.82,
    "--max-running-requests",
    256,
    "--chunked-prefill-size",
    65536,
    "--context-length",
    8192,
    "--max-prefill-tokens",
    16384,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]


class TestKimiK25W4A8(TestAscendPerformanceTestCaseBase):
    benchmark_tool = AISBENCHMARK
    aisbench_dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    # aisbench_dataset_path = (
    #     "/root/.cache/modelscope/hub/datasets/sglang_test/1024x1024_30.jsonl"
    # )
    model = KIMI_K2_5_W4A8_MODEL_PATH
    other_args = KIMI_K2_5_OTHER_ARGS
    envs = KIMI_K2_5_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    image_resolution = "1920x1080"
    image_count = 1
    max_concurrency = 16
    num_prompts = 16
    input_len = 30
    output_len = 1024
    random_range_ratio = 1
    # warmup_requests = 16
    tpot = 50
    output_token_throughput = 200

    def test_kimi_k2_5_w4a8(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
