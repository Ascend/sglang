import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestOpenAIV1Score(CustomTestCase):
    """Test Score API on NPU.

    [Test Category] Interface
    [Test Target] /v1/score
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )
        cls.base_url += "/v1/score"
        cls.tokenizer = get_tokenizer(QWEN3_0_6B_WEIGHTS_PATH)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_score(
        self, query, items, label_token_ids, apply_softmax=False, item_first=False
    ):
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "items": items,
                "label_token_ids": label_token_ids,
                "apply_softmax": apply_softmax,
                "item_first": item_first,
            },
        )
        return response.json()

    def test_score_text_input(self):
        """Test scoring with text input"""
        query = "The capital of France is"
        items = ["Paris", "London", "Berlin"]

        label_token_ids = []
        for item in items:
            token_ids = self.tokenizer.encode(item, add_special_tokens=False)
            if not token_ids:
                self.fail(f"Failed to encode item: {item}")
            label_token_ids.append(token_ids[0])

        response = self.run_score(query, items, label_token_ids, apply_softmax=True)

        if response.get("type") == "BadRequestError":
            self.fail(f"Score request failed with error: {response['message']}")

        self.assertIn("scores", response, "Response should have a 'scores' field")
        self.assertIsInstance(response["scores"], list, "scores should be a list")
        self.assertEqual(
            len(response["scores"]),
            len(items),
            "Number of scores should match number of items",
        )

        for i, score_list in enumerate(response["scores"]):
            self.assertIsInstance(score_list, list, f"Score {i} should be a list")
            self.assertEqual(
                len(score_list),
                len(label_token_ids),
                f"Score {i} length should match label_token_ids",
            )
            self.assertTrue(
                all(isinstance(v, float) for v in score_list),
                f"Score {i} values should be floats",
            )
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=6,
                msg=f"Score {i} probabilities should sum to 1",
            )

        self.assertIn("usage", response, "Response should have a 'usage' field")
        self.assertGreater(response["usage"]["prompt_tokens"], 0)
        self.assertEqual(
            response["usage"]["prompt_tokens"], response["usage"]["total_tokens"]
        )
        self.assertEqual(
            response["usage"]["completion_tokens"],
            0,
            "completion_tokens should be 0 for /v1/score",
        )

    def test_score_token_input(self):
        """Test scoring with token IDs input"""
        query = "The capital of France is"
        items = ["Paris", "London", "Berlin"]

        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        item_ids = [
            self.tokenizer.encode(item, add_special_tokens=False) for item in items
        ]
        label_token_ids = [ids[0] for ids in item_ids if ids]

        response = self.run_score(
            query_ids, item_ids, label_token_ids, apply_softmax=True
        )

        if response.get("type") == "BadRequestError":
            self.fail(f"Score request failed with error: {response['message']}")

        self.assertIn("scores", response, "Response should have a 'scores' field")
        self.assertIsInstance(response["scores"], list, "scores should be a list")
        self.assertEqual(
            len(response["scores"]),
            len(items),
            "Number of scores should match number of items",
        )

        for i, score_list in enumerate(response["scores"]):
            self.assertIsInstance(score_list, list, f"Score {i} should be a list")
            self.assertEqual(
                len(score_list),
                len(label_token_ids),
                f"Score {i} length should match label_token_ids",
            )
            self.assertTrue(
                all(isinstance(v, float) for v in score_list),
                f"Score {i} values should be floats",
            )
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=6,
                msg=f"Score {i} probabilities should sum to 1",
            )

        self.assertIn("usage", response, "Response should have a 'usage' field")
        self.assertGreater(response["usage"]["prompt_tokens"], 0)
        self.assertEqual(
            response["usage"]["prompt_tokens"], response["usage"]["total_tokens"]
        )
        self.assertEqual(
            response["usage"]["completion_tokens"],
            0,
            "completion_tokens should be 0 for /v1/score",
        )

    def test_score_error_handling(self):
        """Test error handling for invalid inputs"""
        query = "The capital of France is"
        items = ["Paris", "London", "Berlin"]

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "items": items,
                "label_token_ids": [999999],
                "apply_softmax": True,
            },
        )
        self.assertEqual(response.status_code, 400)
        error_response = response.json()
        self.assertEqual(error_response["type"], "BadRequestError")
        self.assertIn("Token ID 999999 is out of vocabulary", error_response["message"])


if __name__ == "__main__":
    unittest.main()
