import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-8-npu-a3", nightly=True)


class TestDeepSeekV32(CustomTestCase):
    """Testcase: Verify that the inference accuracy of the vllm-ascend/DeepSeek-V3.2-W8A8 model on the GSM8K dataset is no less than 0.95.

    [Test Category] Model
    [Test Target] vllm-ascend/DeepSeek-V3.2-W8A8
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=6000,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.6",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tp-size",
                "8",
                "--quantization",
                "modelslim",
                "--disable-radix-cache",
                "--enable-deepep-waterfill",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=128,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.95)


if __name__ == "__main__":
    unittest.main()
