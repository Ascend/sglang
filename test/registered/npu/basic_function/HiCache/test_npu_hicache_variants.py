"""NPU adaptation of the EAGLE3 variant from test_hicache_variants.py."""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=600,
    suite="stage-b-test-4-npu-a3",
    nightly=False,
)


class TestHiCacheEagle(CustomTestCase):
    """HiCache + EAGLE3 speculative decoding: verify they coexist without
    regressing MMLU accuracy.

    [Test Category] Functional
    [Test Target] --enable-hierarchical-cache + --speculative-algorithm EAGLE3
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    mmlu_score_threshold = 0.65

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
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
                "4",
                "--mem-fraction-static",
                "0.7",
                "--disable-cuda-graph",
                "--dtype",
                "bfloat16",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "1.2",
                "--page-size",
                "128",
            ],
            env={
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "TRANSFORMERS_VERBOSITY": "error",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        """MMLU score must clear the threshold with HiCache + EAGLE3 both on."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        score = float(metrics["score"])
        self.assertGreaterEqual(
            score,
            self.mmlu_score_threshold,
            f"MMLU score {score} below threshold {self.mmlu_score_threshold}",
        )


if __name__ == "__main__":
    unittest.main()
