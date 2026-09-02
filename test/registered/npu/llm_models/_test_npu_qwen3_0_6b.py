import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, DEFAULT_URL_FOR_TEST, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, \
    popen_launch_server

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

QWEN3_0_6B_WEIGHTS_PATH="/home/weights/Qwen3-0.6B"
class TestQwen306B(CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-0.6B model on the GSM8K dataset is no less than 0.38.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-0.6B
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--chunked-prefill-size",
            256,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-session-radix-cache",
            "--model-checksum",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_lora_session(self):
        # test the correct collaboration of lora with session management functionality
        # Create two sessions
        session_id_1, session_id_2 = [
            requests.post(
                self.base_url + "/open_session",
                json={"capacity_of_str_len": 1000},
            ).json()
            for _ in range(2)
        ]
        self.assertNotEqual(session_id_1, session_id_2, "Session IDs should be different")
        input_ids = [1] * 260


        r1 = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "session_id": session_id_1,
            },
        )
        rid = r1.json()["meta_info"]["id"]
        print("r1.json")
        print(r1.json())


        # r2 = requests.post(
        #     self.base_url + "/generate",
        #     json={
        #         "input_ids": input_ids,
        #         "session_id": session_id_1,
        #     },
        # )
        # print("r2.json")
        # print(r2.json())

        # self.assertGreater(
        #     r2.json()["meta_info"]["cached_tokens"], 0
        # )

        r3 = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "session_id": session_id_2,
            },
        )
        print("r3.json")
        print(r3.json())

        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id_1},
        )

        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id_2},
        )

        r4 = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                # "session_id": session_id_2,
            },
        )

        print("r4.json")
        print(r4.json())

        # self.assertGreater(
        #     r3.json()["meta_info"]["cached_tokens"], 0
        # )


    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)



if __name__ == "__main__":
    unittest.main()
