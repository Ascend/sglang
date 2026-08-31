import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import HY3_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1200, suite="full-16-npu-a3", nightly=True)


class TestHy3(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Tencent-Hunyuan/Hy3 model on the GSM8K dataset is no less than 0.95.

    [Test Category] Model
    [Test Target] Tencent-Hunyuan/Hy3
    """

    model = HY3_WEIGHTS_PATH
    accuracy = 0
    timeout_for_server_launch = 1200
    other_args = [
        "--attention-backend",
        "ascend",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--tp-size",
        "16",
        "--mem-fraction-static",
        "0.84",
        "--dtype",
        "bfloat16",
        "--prefill-max-requests",
        40,
        "--max-running-requests",
        40,
        "--cuda-graph-bs",
        4,
        8,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
    ]
    env = {
        **os.environ,
        "SGLANG_SET_CPU_AFFINITY": "1",
        "ASCEND_USE_FIA": "1",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_BUFFSIZE": "3000",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    }


if __name__ == "__main__":
    unittest.main()
