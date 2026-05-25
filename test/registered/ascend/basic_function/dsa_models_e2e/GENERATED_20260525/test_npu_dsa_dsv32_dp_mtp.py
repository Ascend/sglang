import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="nightly-8-npu-a3", nightly=True)


class TestNPUDsv32DPMTP(CustomTestCase, GSM8KMixin, SpecDecodingMixin):
    """Test DeepSeek-V3.2 with EAGLE speculative decoding and DP attention on NPU.

    [Test Category] E2E Speculative Decoding
    [Test Target] DeepSeek-V3.2, EAGLE3, TP=8, DP=8, DP attention, GSM8K accuracy, speed benchmark
    """

    model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    mem_fraction_static = 0.7
    enable_dp_attention = True
    bs_1_speed_thres = 90

    gsm8k_accuracy_thres = 0.85
    gsm8k_accept_length_thres = 2.5
    gsm8k_num_questions = 500
    gsm8k_num_threads = 500
    gsm8k_num_shots = 20

    accept_length_thres = 2.5
    bs_1_speed_attempts = 3

    speculative_algorithm = "EAGLE3"
    speculative_num_steps = 3
    speculative_eagle_topk = 1
    speculative_num_draft_tokens = 4

    @classmethod
    def setUpClass(cls):
        cls.env = os.environ.copy()
        cls.env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )

        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--tp-size",
            "8",
            "--dp-size",
            "8",
            "--enable-dp-attention",
            "--speculative-algorithm",
            cls.speculative_algorithm,
            "--speculative-num-steps",
            str(cls.speculative_num_steps),
            "--speculative-eagle-topk",
            str(cls.speculative_eagle_topk),
            "--speculative-num-draft-tokens",
            str(cls.speculative_num_draft_tokens),
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
