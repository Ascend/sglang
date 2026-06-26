import unittest

from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import MIMO_V2_5_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)

MIMO_V2_OTHER_ARGS = [
    "--tp",
    "8",
    "--dp",
    "2",
    "--enable-dp-attention",
    "--mm-enable-dp-encoder",
    "--attention-backend",
    "ascend",
    "--mm-attention-backend",
    "ascend_attn",
    "--reasoning-parser",
    "mimo",
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "1.5",
    "--hicache-mem-layout",
    "page_first_direct",
    "--hicache-io-backend",
    "direct",
]
MIMO_V2_MTP_OTHER_ARGS = MIMO_V2_OTHER_ARGS + [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-multi-layer-eagle",
]


class TestMiMoV2(GSM8KMixin, MMMUServerBase):
    gsm8k_accuracy_thres = 0.75
    gsm8k_accept_length_thres = 2.5
    model = MIMO_V2_5_WEIGHTS_PATH
    mem_fraction_static = 0.65
    server_api_key = None
    other_args = MIMO_V2_MTP_OTHER_ARGS

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(True):
            super().setUpClass()


if __name__ == "__main__":
    unittest.main()
