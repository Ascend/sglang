import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLADA2_0_FLASH_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestAFM(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the inclusionAI/LLaDA2.0-flash model on the GSM8K dataset is no less than 0.95.

    [Test Category] Model
    [Test Target] inclusionAI/LLaDA2.0-flash
    """

    model = LLADA2_0_FLASH_WEIGHTS_PATH
    accuracy = 0.95
    timeout_for_server_launch = 3000
    other_args = [
        "--attention-backend",
        "flashinfer",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--dllm-algorithm",
        "LowConfidence",
        "--tp-size",
        8,
        "--max-running-requests",
        4,
        "--enable-tokenizer-batch-encode",
        "--trust-remote-code",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--mem-fraction-static",
        0.8,
        "--cuda-graph-bs",
        1,
        2,
        3,
        4,
    ]


if __name__ == "__main__":
    unittest.main()
