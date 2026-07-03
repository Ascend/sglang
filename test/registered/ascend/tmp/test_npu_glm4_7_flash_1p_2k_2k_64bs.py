import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_4_7_FLASH_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-2-npu-a3",
    nightly=True,
    disabled="performance testcase",
)

GLM_4_7_FLASH_2K_2K_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_SPEC_V2": "1",
}

GLM_4_7_FLASH_2K_2K_OTHER_ARGS = [
    "--tp-size",
    2,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.8,
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    150000,
    "--max-running-requests",
    64,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUGLM_4_7_FLASH_2K_2K_64BS(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for GLM-4.7-Flash 1p in2K out2K 64BS"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = GLM_4_7_FLASH_MODEL_PATH
    other_args = GLM_4_7_FLASH_2K_2K_OTHER_ARGS
    envs = GLM_4_7_FLASH_2K_2K_ENVS
    dataset_name = "random"
    max_concurrency = 64
    num_prompts = 65
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    output_token_throughput = 1288.1

    def test_npu_glm_4_7_flash_2k_2k_64bs(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
