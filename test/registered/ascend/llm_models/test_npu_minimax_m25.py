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

    [Test Category] Model
    [Test Target] Eco-Tech/MiniMax-M2.5-w8a8-QuaRot
    """

    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    accuracy = 0.90
    timeout_for_server_launch = 3000

    # Pure TP=8 (no --ep-size): NPU EP via the generic StandardDispatcher path
    # crashes on a CANN Cast kernel (errno 507015) for both ep=8 and ep=4.
    # Native EP paths (ascend_fuseep / deepep via zbal) bypass this but need
    # --moe-a2a-backend; pending dev verification. Pure TP=8 unblocks the test.
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--reasoning-parser",
        "minimax-append-think",
        "--tool-call-parser",
        "minimax-m2",
        "--max-running-requests",
        "32",
        "--chunked-prefill-size",
        "-1",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
        "--weight-loader-prefetch-checkpoints",
    ]

    def test_bs_1_speed(self):
        # GPU baseline 90 lowered to 5 for NPU pure-TP=8: EP blocked by the CANN
        # Cast kernel bug above, so MoE runs through 8-card all-reduce every
        # layer; under bs=1 this cannot be amortized (~7.4 token/s).
        # Restore to 90 once --moe-a2a-backend ascend_fuseep is verified.
        url = urlparse(DEFAULT_URL_FOR_TEST)
        args = BenchArgs(
            host=url.hostname,
            port=int(url.port),
            max_new_tokens=2048,
        )
        acc_length, speed = send_one_prompt(args, print_output=False)

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (minimax-m25-w8a8, pure TP=8)\n"
                f"- speed: {speed:.2f} token/s (NPU TP-only, EP blocked)\n"
                f"- accept_length: {acc_length:.2f}\n"
            )

        self.assertGreater(speed, 5, f"bs=1 speed {speed:.2f} below 5 token/s")


if __name__ == "__main__":
    unittest.main()
