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
    "--tool-call-parser",
    "glm47",
    "--reasoning-parser",
    "glm45",
    "--disable-radix-cache",
    "--mem-fraction-static",
    0.8,
    "--device",
    "npu",
    "--tp-size",
    4,
    "--ep-size",
    4,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--chunked-prefill-size",
    "32768",
    "--max-running-requests",
    24,
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-moe-a2a-backend",
    "deepep",
]


class TestNPUGLM4_7_FLASH_AIME2025(TestAscendAccuracyTestCaseBase):

    model = GLM_4_7_FLASH_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.6629
    datasets = ["gpqa_diamond"]
    generation_config = {"max_tokens": 131072, "temperature": 1.0, "top-p": 0.95}
    eval_batch_size = 30

    def test_aime2025(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
