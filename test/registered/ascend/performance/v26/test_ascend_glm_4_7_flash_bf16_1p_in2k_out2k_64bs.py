import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_4_7_FLASH_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

GLM_4_7_FLASH_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
}

GLM_4_7_FLASH_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    2,
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
]


class TestGlm47Flash(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = GLM_4_7_FLASH_MODEL_PATH
    other_args = GLM_4_7_FLASH_OTHER_ARGS
    envs = GLM_4_7_FLASH_ENVS
    max_attempts = 5
    dataset_name = "random"
    max_concurrency = 64
    num_prompts = 64
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    output_token_throughput = 1021

    def test_glm_4_7_flash(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
