import unittest

from sglang.test.ascend.test_ascend_utils import QWEN2_5_MATH_RM_72B_WEIGHTS_PATH
from sglang.test.ascend.test_no_hf_reward_base import BaseNoHFRewardModelTest
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="full-4-npu-a3",
    nightly=True,
)


class TestQwen25MathReward(BaseNoHFRewardModelTest, CustomTestCase):
    """Testcase: This test case verifies that the Qwen/Qwen2.5-Math-RM-72B model can successfully generate reward scores
    for different conversational responses using the SGLang framework, without comparing to a reference implementation.

    [Test Category] Model
    [Test Target] Qwen/Qwen2.5-Math-RM-72B
    """

    model_path = QWEN2_5_MATH_RM_72B_WEIGHTS_PATH


if __name__ == "__main__":
    unittest.main()
