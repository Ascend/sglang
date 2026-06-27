import unittest
from urllib.parse import urlparse

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MINIMAX_M2_5_W8A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    write_github_step_summary,
)

register_npu_ci(est_time=1500, suite="full-8-npu-a3", nightly=True)


class TestMiniMaxM25(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify MiniMax-M2.5 (W8A8) end-to-end on Ascend NPU.

    Ported from sgl-project/sglang/test/registered/models_e2e/test_minimax_m25_basic.py.

    [Test Category] Model
    [Test Target] Eco-Tech/MiniMax-M2.5-w8a8-QuaRot
    [Observation Points]
      1. GSM8K accuracy >= 0.90
      2. bs=1 single-request throughput > 90 token/s
      3. (implicit) W8A8 quantization has no accuracy regression
    """

    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    accuracy = 0.90
    timeout_for_server_launch = 3000

    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--ep-size",
        "8",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--reasoning-parser",
        "minimax-append-think",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "-1",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
        "--weight-loader-prefetch-checkpoints",
    ]

    def test_bs_1_speed(self):
        """Port of GPU test_bs_1_speed: bs=1 latency/throughput.

        Asserts single-request throughput > 90 token/s, matching the GPU
        baseline (kept tight per requirement; will revisit based on CI data).
        """
        url = urlparse(DEFAULT_URL_FOR_TEST)
        args = BenchArgs(
            host=url.hostname,
            port=int(url.port),
            max_new_tokens=2048,
        )
        acc_length, speed = send_one_prompt(args, print_output=False)

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (minimax-m25-w8a8)\n"
                f"- speed: {speed:.2f} token/s\n"
                f"- accept_length: {acc_length:.2f}\n"
            )

        self.assertGreater(speed, 90, f"bs=1 speed {speed:.2f} below 90 token/s")


if __name__ == "__main__":
    unittest.main()
