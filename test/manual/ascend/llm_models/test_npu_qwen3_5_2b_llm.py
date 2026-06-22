import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_5_2B_WEIGHTS_PATH
from sglang.test.test_utils import CustomTestCase


class TestQwen3_5_2BGraphWithMTP(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify the inference accuracy of Qwen3.5-2B on GSM8K
    with cuda graph and NEXTN speculative decoding.

    [Test Category] Model
    [Test Target] Qwen/Qwen3.5-2B
    [Test Config] Prefill+Decode, cuda graph enabled, NEXTN speculative decoding
    """

    model = QWEN3_5_2B_WEIGHTS_PATH
    accuracy = 0.55
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.7",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "1",
        "--enable-multimodal",
        "--mm-attention-backend",
        "ascend_attn",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
