import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_npu_ci(est_time=600, suite="full-16-npu-a3", nightly=True)


class TestQwen3Next(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "1024",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "2",
        "--page-size",
        "1",
        "--attention-backend",
        "ascend",
    ]


if __name__ == "__main__":
    unittest.main()
