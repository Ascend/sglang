import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMultiTokenizer(CustomTestCase):
    """Test multi-tokenizer worker performance on NPU.

    [Test Category] Performance
    [Test Target] --tokenizer-worker-num; TTFT latency
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                8,
                "--mem-fraction-static",
                0.8,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multi_tokenizer_ttft(self):
        res = run_bench_serving(
            model=self.model,
            num_prompts=100,
            request_rate=1,
            other_server_args=[
                "--tokenizer-worker-num",
                8,
                "--mem-fraction-static",
                0.8,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
            dataset_name="random",
            random_input_len=1024,
            random_output_len=128,
        )
        self.assertLess(res["median_ttft_ms"], 200)
        self.assertLess(res["median_itl_ms"], 20)


if __name__ == "__main__":
    unittest.main()
