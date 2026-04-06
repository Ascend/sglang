import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(est_time=800, suite="nightly-1-npu-a3", nightly=True)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH


class TestSchedulerRecvIntervalConsistency(unittest.TestCase):
    """
    Test Case: Verify that when --scheduler-recv-interval > 1, inference latency increases but model accuracy is not affected.

    [Test Category] Parameter Validation
    [Test Target] --scheduler-recv-interval
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.server_process = None

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.server_process.pid)

    def _run_gsm8k_evaluation(self, scheduler_recv_interval: int):
        self.server_process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--scheduler-recv-interval",
                str(scheduler_recv_interval),
                "--attention-backend",
                "ascend",
                "--disable-radix-cache",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{self.url.hostname}",
                port=int(self.url.port),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            return metrics
        finally:
            kill_process_tree(self.server_process.pid)

    def test_scheduler_recv_interval_consistency(self):
        baseline_metrics = self._run_gsm8k_evaluation(scheduler_recv_interval=1)
        test_metrics = self._run_gsm8k_evaluation(scheduler_recv_interval=50000)

        self.assertGreaterEqual(
            baseline_metrics["accuracy"], 0.38, msg="Baseline accuracy is too low."
        )

        self.assertGreaterEqual(
            test_metrics["accuracy"],
            baseline_metrics["accuracy"] - 0.02,
            msg=f"Accuracy dropped by more than 2%! Baseline: {baseline_metrics['accuracy']:.2%}, Test: {test_metrics['accuracy']:.2%}",
        )

        # When --scheduler-recv-interval is set to 50000, the inference latency increases compared to 1
        self.assertGreaterEqual(
            test_metrics["latency"],
            baseline_metrics["latency"],
            msg="Test latency should be >= baseline latency.",
        )


if __name__ == "__main__":
    unittest.main()
