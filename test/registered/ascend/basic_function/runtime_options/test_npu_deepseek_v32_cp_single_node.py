import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    write_github_step_summary,
)
# from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="debug-full-16-npu-a3", nightly=True)


class TestDeepseekV32CPInSeqSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--attn-cp-size",
            "4",
            "--enable-dsa-prefill-context-parallel",
            "--dsa-prefill-cp-mode",
            "in-seq-split",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-frac",
            "0.7",
            "--cuda-graph-max-bs",
            "32",
            "--max-running-requests",
            "32",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=32,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        write_github_step_summary(
            f"### test_a_gsm8k (deepseek-v32-cp-in-seq-split)\n"
            f'{metrics["score"]=:.3f}\n'
        )
        self.assertGreater(metrics["score"], 0.935)


class TestDeepseekV32CPRoundRobinSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--attn-cp-size",
            "8",
            "--enable-dsa-prefill-context-parallel",
            "--dsa-prefill-cp-mode",
            "round-robin-split",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-frac",
            "0.7",
            "--cuda-graph-max-bs",
            "32",
            "--max-running-requests",
            "32",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=32,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        write_github_step_summary(
            f"### test_a_gsm8k (deepseek-v32-cp-in-seq-split)\n"
            f'{metrics["score"]=:.3f}\n'
        )
        self.assertGreater(metrics["score"], 0.935)


if __name__ == "__main__":
    unittest.main()
