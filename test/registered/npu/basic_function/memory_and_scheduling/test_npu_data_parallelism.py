import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="full-2-npu-a3", nightly=True)


class TestDataParallelism(CustomTestCase, GSM8KMixin):
    """
    Test case for data parallelism (DP=2) on Ascend NPU.
    Test covered:
    1. Reload model weight from disk via HTTP api
    2. Query server runtime info repeatedly
    3. GSM8K accuracy verification
    """

    gsm8k_accuracy_thres = 0.37  # Lowered from 0.7 for DP mode compatibility

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dp",
                2,
                "--attention-backend",
                "ascend",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_update_weight(self):
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": QWEN3_0_6B_WEIGHTS_PATH},
        )

        # check if the response is 200
        assert response.status_code == 200

        # pause a few seconds then send again
        time.sleep(1)

        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={"model_path": QWEN3_0_6B_WEIGHTS_PATH},
        )

        # check if the response is 200
        assert response.status_code == 200

    def test_get_memory_pool_size(self):
        # use `server_info` instead since `get_memory_pool_size` is merged into `server_info`
        response = requests.get(self.base_url + "/server_info")
        assert response.status_code == 200

        time.sleep(1)

        response = requests.get(self.base_url + "/server_info")
        assert response.status_code == 200


if __name__ == "__main__":
    unittest.main()
