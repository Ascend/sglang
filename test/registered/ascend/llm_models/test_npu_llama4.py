import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_npu_ci(est_time=1800, suite="full-8-npu-a3", nightly=True)

from sglang.test.ascend.test_ascend_utils import (
    LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,
)


class TestLlama4(unittest.TestCase):
    """Testcase: Verify that the inference accuracy of the meta-llama/Llama-4-Scout-17B-16E-Instruct model on the GSM8K dataset is no less than 0.9.

    [Test Category] Model
    [Test Target] meta-llama/Llama-4-Scout-17B-16E-Instruct
    """

    def test_llama4(self):
        """Run accuracy test for Llama-4-Scout."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
            "--chat-template=llama-4",
            "--mem-fraction-static=0.8",
            "--context-length=8192",
            "--disable-cuda-graph",
            "--disable-radix-cache",
        ]

        variants = [
            ModelLaunchSettings(
                LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Llama-4-Scout",
            accuracy_params=AccuracyTestParams(dataset="gsm8k", baseline_accuracy=0.9),
        )


if __name__ == "__main__":
    unittest.main()
