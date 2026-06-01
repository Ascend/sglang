import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNpuEagle3(CustomTestCase):
    """Testcase: Verify GSM8K inference accuracy ≥0.81 for model with specified EAGLE3 speculative inference parameters.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-draft-model-quantization; --speculative-algorithm; --speculative-draft-model-path; --speculative-num-steps; --speculative-eagle-topk; --speculative-num-draft-tokens; --speculative-attention-mode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.accuracy = 0.81
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-radix-cache",
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_8B_EAGLE3_WEIGHTS_PATH,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "decode",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]

        cls.extra_envs = {
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
            "SGLANG_SPEC_NAN_DETECTION": "1",
            "SGLANG_SPEC_OOB_DETECTION": "1",
        }
        os.environ.update(cls.extra_envs)

    def test_gsm8k(self):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=1500,
            other_args=self.common_args,
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                eval_name="gsm8k",
                api="completion",
                num_examples=1319,
                num_threads=128,
                max_tokens=512,
                num_shots=5,
                temperature=0.0,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(
                metrics["score"],
                self.accuracy,
                f"GSM8K score {metrics['score']} below threshold {self.accuracy}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
