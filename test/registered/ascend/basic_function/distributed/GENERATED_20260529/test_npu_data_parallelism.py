import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=91,
    suite="nightly-2-npu-a3",
    nightly=True,
)


class TestNPUDataParallelism(CustomTestCase):
    """Test data parallelism on NPU.

    [Test Category] Distributed
    [Test Target] Data parallelism (DP=2)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--disable-cuda-graph",
            "--dp",
            "2",
            "--mem-fraction-static",
            "0.3",
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

    def test_update_weight(self):
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": self.model},
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(1)

        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": self.model},
        )
        self.assertEqual(response.status_code, 200)

    def test_server_info(self):
        response = requests.get(self.base_url + "/server_info")
        self.assertEqual(response.status_code, 200)

        time.sleep(1)

        response = requests.get(self.base_url + "/server_info")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
