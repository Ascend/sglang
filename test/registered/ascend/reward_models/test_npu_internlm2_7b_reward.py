import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import unittest

from sglang.test.ascend.base_no_hf_reward_test import BaseNoHFRewardModelTest
from sglang.test.ascend.test_ascend_utils import INTERNLM2_7B_REWARD_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="full-4-npu-a3",
    nightly=True,
)


class TestInternlm2Reward(BaseNoHFRewardModelTest, CustomTestCase):
    """Testcase: This test case verifies that the Shanghai_AI_Laboratory/internlm2-7b-reward model can successfully generate reward
    scores for different conversational responses using the SGLang framework, without comparing to a reference implementation.

    [Test Category] Model
    [Test Target] Shanghai_AI_Laboratory/internlm2-7b-reward
    """

    model_path = INTERNLM2_7B_REWARD_WEIGHTS_PATH


if __name__ == "__main__":
    unittest.main()
