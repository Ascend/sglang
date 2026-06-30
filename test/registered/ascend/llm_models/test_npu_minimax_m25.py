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

    # NPU EP only supports deepep backend (see Ascend reference:
    #   test_npu_deepep.py, test_npu_deepep_auto_qwen3_30b_a3b_w8a8.py).
    # Use dp-size instead of ep-size; moe-a2a-backend=deepep replaces the
    # crashing generic EP path (moe_a2a_backend='none' -> StandardDispatcher
    # -> CANN Cast kernel errno 507015).
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--dp-size",
        "1",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
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

    # Deepep needs larger HCCL buffer and dispatch token budget
    # (see test_npu_deepep.py: HCCL_BUFFSIZE=1000, dispatch=32).
    env = {
        **GSM8KAscendMixin.env,
        "HCCL_BUFFSIZE": "1000",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
        "SGLANG_WARMUP_TIMEOUT": "3600",
        "TRANSFORMERS_VERBOSITY": "error",
    }

    def test_bs_1_speed(self):
        # NPU TP=8 + deepep: bs=1 MoE all-to-all not amortized (~7.4 token/s baseline).
        url = urlparse(DEFAULT_URL_FOR_TEST)
        args = BenchArgs(
            host=url.hostname,
            port=int(url.port),
            max_new_tokens=2048,
        )
        acc_length, speed = send_one_prompt(args, print_output=False)

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (minimax-m25-w8a8, TP=8 + deepep)\n"
                f"- speed: {speed:.2f} token/s\n"
                f"- accept_length: {acc_length:.2f}\n"
            )

        self.assertGreater(speed, 5, f"bs=1 speed {speed:.2f} below 5 token/s")


if __name__ == "__main__":
    unittest.main()
