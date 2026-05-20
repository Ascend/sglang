import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_35B_A3B_MODEL_PATH,
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)


class TestNPUQwen3_6_35BA3B_1P_GPQA(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3.6-35B-A3B 1p GPQA"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_6_35B_A3B_MODEL_PATH
    dataset_name = "gpqa"
    accuracy = 86

    other_args = [
        "--tp-size",
        2,
        "--nnodes",
        1,
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
    ]

    def test_npu_qwen3_6_35b_a3b_1p_gpqa(self):
        """Run NPU accuracy test for Qwen3.6-35B-A3B GPQA"""
        self.run_accuracy()


class TestNPUQwen3_6_35BA3B_1P_AIME2025(TestAscendAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3.6-35B-A3B 1p AIME2025"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_6_35B_A3B_MODEL_PATH
    dataset_name = "aime2025"
    accuracy = 92.7

    other_args = [
        "--tp-size",
        2,
        "--nnodes",
        1,
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
    ]

    def test_npu_qwen3_6_35b_a3b_1p_aime2025(self):
        """Run NPU accuracy test for Qwen3.6-35B-A3B AIME2025"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()