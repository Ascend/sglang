import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


class TestOffloadGroupSize(CustomTestCase):
    """Testcase: Test the --offload-group-size parameter the parameter
    takes effect and the inference request succeeds.

    [Test Category] Parameter
    [Test Target] --offload-group-size
    """

    OTHER_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--offload-group-size",
        "2",
        "--tp-size",
        "2",
    ]

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.OTHER_ARGS,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")
        kill_process_tree(cls.process.pid)

    def _send_request(self):
        return requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

    def _check_offload_message(self):
        self.err_log_file.seek(0)
        self.assertIn("[offloader]", self.err_log_file.read())

    def test_inference(self):
        response = self._send_request()
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        self._check_offload_message()


class TestOffloadMeta(TestOffloadGroupSize):
    """Testcase: Test the --offload-mode = meta parameter, the parameter
    takes effect and the inference request succeeds.

    [Test Category] Parameter
    [Test Target] --offload-mode, --offload-num-in-group, --offload-prefetch-step
    """

    OTHER_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--offload-group-size",
        "4",
        "--offload-num-in-group",
        "2",
        "--offload-prefetch-step",
        "2",
        "--offload-mode",
        "meta",
    ]

    # When --offload-mode=meta, it is in debugging mode, creating empty tensors
    # and resulting in incorrect inference results
    def test_inference(self):
        response = self._send_request()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Paris", response.text)
        self._check_offload_message()


class TestOffloadShardedGpu(TestOffloadGroupSize):
    """Testcase: Test the --offload-mode=sharded_gpu parameter, the parameter
    takes effect and the inference request succeeds.

    [Test Category] Parameter
    [Test Target] --offload-mode
    """

    OTHER_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--dp",  # When --offload-mode = sharded_gpu, must set --dp > 1 and --tp = 1.
        "2",
        "--offload-group-size",
        "4",
        "--offload-num-in-group",
        "2",
        "--offload-prefetch-step",
        "2",
        "--offload-mode",
        "sharded_gpu",
    ]

    def _check_offload_message(self):
        self.err_log_file.seek(0)
        self.assertIn("[offloader] post_init", self.err_log_file.read())


if __name__ == "__main__":
    unittest.main()
