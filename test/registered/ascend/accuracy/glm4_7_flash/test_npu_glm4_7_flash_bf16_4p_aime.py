import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import GLM_4_7_FLASH_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-4-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)
register_npu_ci(
    est_time=3600,
    suite="stage-b-test-4-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "1000",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--tp-size",
    4,
    "--ep-size",
    4,
    "--mem-fraction-static",
    0.8,
    "--disable-radix-cache",
    "--max-running-requests",
    1,
    "--chunked-prefill-size",
    "32768",
    "--tool-call-parser",
    "glm47",
    "--reasoning-parser",
    "glm45",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-attention-mode",
    "decode",
    "--speculative-moe-a2a-backend",
    "deepep",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]


class TestNPUGLM4_7_FLASH_AIME2025(TestAscendAccuracyTestCaseBase):

    model = GLM_4_7_FLASH_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.6
    datasets = ["aime25"]
    generation_config = {
        "max_tokens": 40000,
        "temperature": 0,
        "top-p": 1,
        "stream": True,
        "retries": 2,
        "seed": 1234,
    }
    eval_batch_size = 8

    def test_aime2025(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
