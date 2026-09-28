import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLaDA2_0_MINI_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    CustomTestCase,
    is_in_ci,
    write_github_step_summary,
)

register_npu_ci(est_time=800, suite="base-b-test-4-npu-a3")
register_npu_ci(est_time=800, suite="nightly-4-npu-a3", nightly=True)

_DLLM_ALGO_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "llada2_lowconf.yaml"
)

_LLADA2_BASE_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.9",
    "--max-running-requests",
    "1",
    "--attention-backend",
    "ascend",
    "--dllm-algorithm",
    "LowConfidence",
    "--dllm-algorithm-config",
    _DLLM_ALGO_CONFIG,
]


class TestLLaDA2Mini(GSM8KAscendMixin, CustomTestCase):
    """LLaDA2-mini GSM8K with LowConfidence in synchronous dLLM mode.

    [Test Category] Parameter
    [Test Target] --dllm-algorithm; --dllm-algorithm-config; --no-dllm-fdfo
    """

    model = LLaDA2_0_MINI_WEIGHTS_PATH
    other_args = _LLADA2_BASE_ARGS + [
        "--no-dllm-fdfo",  # FDFO (PR #27551) halves single-batch speed on NPU; use sync mode
    ]
    accuracy = 0.88
    output_throughput = 70
    speed_threshold = 130
    speed_summary_name = "llada2-mini"

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed ({self.speed_summary_name}) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, self.speed_threshold)


class TestLLaDA2MiniFdfo(TestLLaDA2Mini):
    """LLaDA2-mini GSM8K with LowConfidence and FDFO scheduling.

    [Test Category] Parameter
    [Test Target] --dllm-algorithm; --dllm-algorithm-config; --dllm-fdfo
    """

    other_args = _LLADA2_BASE_ARGS + [
        "--dllm-fdfo",
    ]
    speed_threshold = 50
    speed_summary_name = "llada2-mini fdfo"
    # FDFO is slower than sync mode; do not reuse the sync throughput bar.
    output_throughput = 0


if __name__ == "__main__":
    unittest.main()
