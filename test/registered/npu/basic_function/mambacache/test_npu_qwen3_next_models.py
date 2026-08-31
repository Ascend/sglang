import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import (
    DefaultServerBase,
    openai_api_env,
)
from sglang.test.test_utils import popen_launch_server

register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)

QWEN3_NEXT_MODEL = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path


class TestQwen3Next(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 128
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "1024",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "16",
        "--page-size",
        "16",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mamba-track-interval",
        "128",
        "--page-size",
        "128",
    ]

    @classmethod
    def setUpClass(cls):
        assert cls.model is not None, "Please set cls.model in subclass"
        with openai_api_env(cls.api_key):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=cls.other_args,
                env={
                    "ASCEND_USE_FIA": "1",
                    "GDN_USE_MEGA_GDN": "1",
                },
            )


if __name__ == "__main__":
    unittest.main()
