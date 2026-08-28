import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GRANITE_4_0_H_MICRO_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestGranite40HMicro(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the mistralai/Mistral-7B-Instruct-v0.2 model on the GSM8K dataset is no less than 0.8135.

    [Test Category] Model
    [Test Target] ibm-granite/granite-4.0-h-micro
    """

    model = GRANITE_4_0_H_MICRO_WEIGHTS_PATH
    accuracy = 0.8135
    gsm8k_num_shots = 8


if __name__ == "__main__":
    unittest.main()
