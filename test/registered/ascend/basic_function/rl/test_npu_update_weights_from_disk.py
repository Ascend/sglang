import unittest

import requests

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_2_1B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="nightly-2-npu-a3", nightly=True)


class TestEngineUpdateWeightsFromDisk(CustomTestCase):
    """Test update weights from disk on NPU.

    [Test Category] RL Weight Update
    [Test Target] Engine.update_weights_from_disk
    """

    def setUp(self):
        self.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        self.engine = sgl.Engine(
            model_path=self.model,
            trust_remote_code=True,
        )

    def tearDown(self):
        if hasattr(self, "engine") and self.engine:
            try:
                self.engine.shutdown()
            except Exception:
                pass  # Ignore shutdown errors

    def run_decode(self):
        prompts = ["The capital of France is"]
        sampling_params = {"temperature": 0, "max_new_tokens": 32}
        outputs = self.engine.generate(prompts, sampling_params)
        return outputs[0]["text"]

    def run_update_weights(self, model_path):
        ret = self.engine.update_weights_from_disk(model_path)
        return ret

    def test_update_weights(self):
        origin_response = self.run_decode()
        new_model_path = LLAMA_3_2_1B_WEIGHTS_PATH
        ret = self.run_update_weights(new_model_path)
        self.assertTrue(ret[0])

        updated_response = self.run_decode()
        self.assertNotEqual(origin_response[:32], updated_response[:32])

        ret = self.run_update_weights(self.model)
        self.assertTrue(ret[0])
        reverted_response = self.run_decode()
        self.assertEqual(origin_response[:32], reverted_response[:32])

    def test_update_weights_unexist_model(self):
        origin_response = self.run_decode()
        new_model_path = self.model.replace("-Instruct", "wrong")
        ret = self.run_update_weights(new_model_path)
        self.assertFalse(ret[0])
        updated_response = self.run_decode()
        self.assertEqual(origin_response[:32], updated_response[:32])


class TestServerUpdateWeightsFromDisk(CustomTestCase):
    """Test update weights from disk via HTTP server on NPU.

    [Test Category] RL Weight Update
    [Test Target] /update_weights_from_disk endpoint
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = "http://127.0.0.1:30002"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        try:
            kill_process_tree(cls.process.pid)
        except Exception:
            pass  # Ignore cleanup errors

    def run_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        return response.json()["text"]

    def get_model_info(self):
        response = requests.get(self.base_url + "/get_model_info")
        model_path = response.json()["model_path"]
        return model_path

    def run_update_weights(self, model_path, flush_cache=True):
        response = requests.post(
            self.base_url + "/update_weights_from_disk",
            json={
                "model_path": model_path,
                "flush_cache": flush_cache,
            },
        )
        return response.json()

    def test_update_weights(self):
        origin_model_path = self.get_model_info()
        origin_response = self.run_decode()

        new_model_path = LLAMA_3_2_1B_WEIGHTS_PATH
        ret = self.run_update_weights(new_model_path)
        self.assertTrue(ret["success"])

        updated_model_path = self.get_model_info()
        self.assertEqual(updated_model_path, new_model_path)
        self.assertNotEqual(updated_model_path, origin_model_path)

        updated_response = self.run_decode()
        self.assertNotEqual(origin_response[:32], updated_response[:32])

        ret = self.run_update_weights(origin_model_path)
        self.assertTrue(ret["success"])
        updated_model_path = self.get_model_info()
        self.assertEqual(updated_model_path, origin_model_path)

        updated_response = self.run_decode()
        self.assertEqual(origin_response[:32], updated_response[:32])

    def test_update_weights_unexist_model(self):
        origin_model_path = self.get_model_info()
        origin_response = self.run_decode()

        new_model_path = self.model.replace("-Instruct", "wrong")
        ret = self.run_update_weights(new_model_path)
        self.assertFalse(ret["success"])

        updated_model_path = self.get_model_info()
        self.assertEqual(updated_model_path, origin_model_path)

        updated_response = self.run_decode()
        self.assertEqual(origin_response[:32], updated_response[:32])


if __name__ == "__main__":
    unittest.main()
