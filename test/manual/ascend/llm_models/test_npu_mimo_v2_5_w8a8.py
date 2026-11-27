import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MIMO_V2_5_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

# TEMP (validation only): register the manual MiMo MultiLayer case so the
# validation workflow can enumerate it in the full-acc-8-npu-a3 suite.
register_npu_ci(est_time=3600, suite="full-acc-8-npu-a3", nightly=True)


class TestMiMoV25W8A8GraphWithMTP(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify the inference accuracy of MiMo-V2.5-W8A8 on GSM8K with cuda graph and MTP (speculative decoding).

    [Test Category] Model
    [Test Target] XiaomiMiMo/MiMo-V2.5-W8A8
    [Test Config] Prefill+Decode, cuda graph enabled, EAGLE speculative decoding, modelslim quantization
    """

    model = MIMO_V2_5_W8A8_WEIGHTS_PATH
    accuracy = 0.9
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.75",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--reasoning-parser",
        "mimo",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-multi-layer-eagle",
        "--quantization",
        "modelslim",
        "--speculative-draft-model-quantization",
        "unquant",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
    ]


if __name__ == "__main__":
    unittest.main()
