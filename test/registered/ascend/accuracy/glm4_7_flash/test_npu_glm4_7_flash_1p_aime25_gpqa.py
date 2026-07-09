import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import GLM_4_7_FLASH_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=7200,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_SET_CPU_AFFINITY": "1",
}

OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    2,
    "--chunked-prefill-size",
    16384,
    "--max-prefill-tokens",
    150000,
    "--dtype",
    "bfloat16",
    "--max-running-requests",
    32,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.75,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    32,
    "--watchdog-timeout",
    9000,
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
]


class TestNPUGLM4_7Flash_2P_AIME2025(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for GLM-4.7-Flash on AIME2025 dataset.

    [Test Category] Accuracy
    [Test Target] GLM-4.7-Flash / AIME2025
    """

    model = GLM_4_7_FLASH_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.906
    datasets = ["aime25"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_aime2025(self):
        self.run_accuracy()


class TestNPUGLM4_7Flash_2P_GPQA(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for GLM-4.7-Flash on GPQA dataset.

    [Test Category] Accuracy
    [Test Target] GLM-4.7-Flash / GPQA
    """

    model = GLM_4_7_FLASH_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.752
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_gpqa(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
