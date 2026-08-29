import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import GRANITE_4_0_MICRO_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestGranite40Micro(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the ibm-granite/granite-4.0-micro model on the GSM8K dataset is no less than 0.8545.

    [Test Category] Model
    [Test Target] ibm-granite/granite-4.0-micro
    """

    model = GRANITE_4_0_MICRO_WEIGHTS_PATH
    accuracy = 0
    gsm8k_num_shots = 8
    num_questions = 1319
    max_tokens = 1024
    env = {
        **os.environ,
    }


class TestGranite40Micro_200(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the ibm-granite/granite-4.0-micro model on the GSM8K dataset is no less than 0.8545.

    [Test Category] Model
    [Test Target] ibm-granite/granite-4.0-micro
    """

    model = GRANITE_4_0_MICRO_WEIGHTS_PATH
    accuracy = 0
    gsm8k_num_shots = 8
    max_tokens = 1024
    env = {
        **os.environ,
    }


if __name__ == "__main__":
    unittest.main()
