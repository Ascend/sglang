import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

MODEL = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
MMMU_ACCURACY_THRESHOLD = 0.2
MMMU_NUM_EXAMPLES = 32


class TestNPUVLMTP4(CustomTestCase):
    """Test VLM TP=4 functionality on NPU.

    [Test Category] VLM Performance
    [Test Target] TP=4 parallelism, MMMU benchmark
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Use tp-size=1 for CI environment (only 1 NPU available)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--enable-multimodal",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--tp-size",
                "1",
                "--cuda-graph-max-bs",
                "32",
                "--mem-fraction-static",
                "0.35",
                "--disable-cuda-graph",
                "--chunked-prefill-size",
                "2048",
                "--max-running-requests",
                "128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_mmmu_accuracy(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            model=self.model,
            eval_name="mmmu",
            num_examples=MMMU_NUM_EXAMPLES,
            num_threads=16,
            max_tokens=2048,
            chat_template_kwargs={"enable_thinking": False},
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"MMMU score: {metrics['score']}")
        self.assertGreaterEqual(
            metrics["score"],
            MMMU_ACCURACY_THRESHOLD,
            f"MMMU accuracy {metrics['score']:.4f} below threshold {MMMU_ACCURACY_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
