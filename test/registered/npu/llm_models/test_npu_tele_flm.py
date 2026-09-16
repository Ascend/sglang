import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import TELE_FLM_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the FLM/Tele-FLM model on the GSM8K dataset is no less than 0.41.

    [Test Category] Model
    [Test Target] FLM/Tele-FLM
    """

    model = TELE_FLM_WEIGHTS_PATH
    accuracy = 0.41
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        16,
        "--watchdog-timeout",
        300000,
    ]
    env = {
        **os.environ,
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    }


if __name__ == "__main__":
    unittest.main()
