import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestDeepEpDeepseek(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "8",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--disable-cuda-graph",
                "--dp-size",
                8,
                "--enable-dp-attention",
                "--chunked-prefill-size",
                1024,
                "--mem-fraction-static",
                0.7,
            ],
            env={
                "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
                "HCCL_BUFFSIZE": "2048",
                "MOE_ENABLE_TOPK_NEG_ONE": "1",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        expect_score = 0.58
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=32,
        )
        print("Starting mmlu test...")
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], expect_score)

    def test_gsm8k(self):
        expect_accuracy = 0.34
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        print("Starting gsm8k test...")
        metrics = run_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )


if __name__ == "__main__":
    unittest.main()
