import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_MM_CUSTOM_GEN,
    QWEN3_VL_30B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_VL_30B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "200",
    "USE_TRITON_MATMUL": "1",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_VIT_ENABLE_CUDA_GRAPH": "1",
    # "ASCEND_LAUNCH_BLOCKING": "1",  # 临时定位问题
}

QWEN3_VL_30B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.8,
    "--max-running-requests",
    368,
    "--prefill-max-requests",
    37,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    102400,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--cuda-graph-bs",
    92,
    "--tokenizer-worker-num",
    4,
    "--dtype",
    "bfloat16",
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
]


class TestQwen3Vl30B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = AISBENCHMARK
    aisbench_dataset_type = AISBENCHMARK_DATASET_MM_CUSTOM_GEN
    # aisbench_dataset_path = (
    #     "/root/.cache/modelscope/hub/datasets/sglang_test/1024x1024_0.jsonl"
    # )
    model = QWEN3_VL_30B_MODEL_PATH
    other_args = QWEN3_VL_30B_OTHER_ARGS
    envs = QWEN3_VL_30B_ENVS
    backend = "sglang-oai-chat"
    dataset_name = "image"
    image_resolution = "1024x1024"
    image_count = 1
    max_concurrency = 368
    num_prompts = 1472
    input_len = 1
    output_len = 1024
    random_range_ratio = 1
    seed = 1000
    tpot = 50
    output_token_throughput = 6000

    def test_qwen3_vl_30b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
