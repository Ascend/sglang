import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import QWEN3_5_35B_A3B_MODEL_PATH
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
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "40",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
}

OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    2,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    10000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs",
    1,
    2,
    4,
    6,
    8,
    16,
    24,
    28,
    30,
    "--mm-attention-backend",
    "ascend_attn",
]


class TestNPUQwen_3_5_35B_A3B_gpqa(TestAscendAccuracyTestCaseBase):
    model = QWEN3_5_35B_A3B_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.823
    datasets = ["gpqa_diamond"]
    eval_batch_size = 30
    generation_config = {
        "max_tokens": 40000,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }

    def test_gpqa(self):
        self.run_accuracy()

if __name__ == "__main__":
    unittest.main()
