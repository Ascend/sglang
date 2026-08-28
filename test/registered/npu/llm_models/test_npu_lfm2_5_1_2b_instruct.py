import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LFM2_5_1_2B_Instruct_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestLFM2512BInstruct(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the mistralai/Mistral-7B-Instruct-v0.2 model on the GSM8K dataset is no less than 0.6452.

    [Test Category] Model
    [Test Target] LiquidAI/LFM2.5-1.2B-Instruct
    """

    model = LFM2_5_1_2B_Instruct_WEIGHTS_PATH
    accuracy = 0.6452
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--tp-size",
        "1",
    ]


if __name__ == "__main__":
    unittest.main()
