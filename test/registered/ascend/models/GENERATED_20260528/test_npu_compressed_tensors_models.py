import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWQ_32B_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNPUCompressedTensorsQWQ32BW8A8(CustomTestCase):
    """Test W8A8 quantized model (QWQ-32B-W8A8) on NPU.

    [Test Category] Quantization Model
    [Test Target] QWQ-32B-W8A8, W8A8 quantization, GSM8K accuracy
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWQ_32B_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--quantization",
                "modelslim",
                "--tp-size",
                "2",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.7",
            ],
        )
        cls.gsm8k_lower_bound = 0.55

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
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], self.gsm8k_lower_bound)


if __name__ == "__main__":
    unittest.main()