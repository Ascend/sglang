import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.npu_eval_accuracy_kit import _is_pr_pipeline, run_npu_pr_smoke
from sglang.test.ascend.test_ascend_utils import LLaDA2_0_MINI_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
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


class _LLaDA2MiniBase(CustomTestCase):
    extra_args = []
    accuracy = 0.88
    speed_threshold = 130
    speed_summary_name = "llada2-mini"

    @classmethod
    def setUpClass(cls):
        cls.model = LLaDA2_0_MINI_WEIGHTS_PATH
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_LLADA2_BASE_ARGS + cls.extra_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if _is_pr_pipeline:
            run_npu_pr_smoke(DEFAULT_URL_FOR_TEST)
            return
        args = SimpleNamespace(
            max_tokens=512,
            base_url=DEFAULT_URL_FOR_TEST,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], self.accuracy)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(DEFAULT_URL_FOR_TEST.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed ({self.speed_summary_name}) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, self.speed_threshold)


class TestLLaDA2Mini(_LLaDA2MiniBase):
    """Testcase: LLaDA2-mini GSM8K with LowConfidence in synchronous dLLM mode.

    [Test Category] Parameter
    [Test Target] --dllm-algorithm; --dllm-algorithm-config; --no-dllm-fdfo
    """

    extra_args = [
        "--no-dllm-fdfo",  # FDFO (PR #27551) halves single-batch speed on NPU; use sync mode
    ]


class TestLLaDA2MiniFdfo(_LLaDA2MiniBase):
    """Testcase: LLaDA2-mini GSM8K with LowConfidence and FDFO scheduling.

    [Test Category] Parameter
    [Test Target] --dllm-algorithm; --dllm-algorithm-config; --dllm-fdfo
    """

    extra_args = [
        "--dllm-fdfo",
    ]
    speed_threshold = 50
    speed_summary_name = "llada2-mini fdfo"


if __name__ == "__main__":
    unittest.main()
