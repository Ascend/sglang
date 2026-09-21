import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestRequestLengthValidation(CustomTestCase):
    """Testcase：Verify set --max-total-tokens and --context-length, can correctly reject inference requests
    that exceed the limits and throw the specified exceptions.

    [Test Category] Parameter
    [Test Target] --max-total-tokens, --context-length
    """

    model = QWEN3_0_6B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start server with auto truncate disabled
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--max-total-tokens",
                "1000",
                "--context-length",
                "1000",
                "--attention-backend",
                "ascend",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def create_openai_client(self):
        return openai.Client(
            api_key=self.api_key, base_url=f"{DEFAULT_URL_FOR_TEST}/v1"
        )

    def test_input_length_no_longer_than_context_length_success(self):
        """
        Verify that the interface can be requested successfully when the length of the
        text input by the user does not exceed the model context length.
        """

        client = self.create_openai_client()
        long_text = "hello " * 500
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": long_text},
            ],
            temperature=0,
        )
        completions_tokens = response.usage.completion_tokens
        self.assertGreater(completions_tokens, 0)

    def test_input_length_longer_than_context_length(self):
        """
        Verify whether the interface normally throws an exception when the number of tokens of the
        input text exceeds the model context length.
        """

        client = self.create_openai_client()

        long_text = "hello " * 1200  # Will tokenize to more than context length

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_input_length_longer_than_maximum_allowed_length(self):
        """
        Verify whether the interface can throw exceptions correctly when the length of the
        incoming text exceeds the maximum context limit of the model.
        """

        client = self.create_openai_client()

        long_text = "hello " * 999  # the maximum allowed length is 994 tokens

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_input_length_longer_than_context_length_streaming(self):
        """
        In the streaming invocation scenario, verify whether an error is thrown normally when the length of the
        input text exceeds the model's context window.
        """

        client = self.create_openai_client()

        long_text = "hello " * 1200

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                stream=True,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_not_longer_max_tokens_validation_success(self):
        """
        When the verification request does not exceed the maximum token limit,
        the interface call verification logic is executed normally.
        """

        client = self.create_openai_client()
        long_text = "hello "
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": long_text},
            ],
            temperature=0,
            max_tokens=800,
        )
        completions_tokens = response.usage.completion_tokens
        self.assertGreater(completions_tokens, 0)

    def test_max_tokens_validation(self):
        """Exception verification for exceeding the limit of the max_tokens parameter"""

        client = self.create_openai_client()

        long_text = "hello "

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                max_tokens=1200,
            )

        self.assertIn(
            "max_completion_tokens is too large",
            str(cm.exception),
        )

    def test_token_ids_logprob_out_of_vocabulary(self):
        """
        Verify whether the interface can correctly return an error when checking the
        token_ids_logprob parameter outside the range of the incoming vocabulary list.
        """

        headers = {"Authorization": f"Bearer {self.api_key}"}
        for token_ids_logprob in ([-1], [2_000_000_000]):
            response = requests.post(
                f"{self.base_url}/generate",
                headers=headers,
                json={
                    "text": "hi",
                    "sampling_params": {"max_new_tokens": 1},
                    "return_logprob": True,
                    "token_ids_logprob": token_ids_logprob,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("out-of-vocabulary", response.text)

    def test_token_ids_logprob_rejects_nested_list(self):
        # Nested lists are a batch-level wire format; a single request must
        # pass a flat list of ints. A ragged nested list with in-vocab ids
        # would otherwise crash the scheduler in the sampler gather.
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for token_ids_logprob in ([[0]], [[0], [1, 2]]):
            response = requests.post(
                f"{self.base_url}/generate",
                headers=headers,
                json={
                    "text": "hi",
                    "sampling_params": {"max_new_tokens": 1},
                    "return_logprob": True,
                    "token_ids_logprob": token_ids_logprob,
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("flat list of integers", response.text)

    def test_token_ids_logprob_batch_with_one_oov(self):
        """
        Verify When sending batch requests, if token IDs outside the vocabulary are included in token_ids_logprob,
        the API return error messages correctly
        """

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/generate",
            headers=headers,
            json={
                "text": ["hi", "hi"],
                "sampling_params": {"max_new_tokens": 1},
                "return_logprob": True,
                "token_ids_logprob": [[0], [2_000_000_000]],
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("out-of-vocabulary", response.text)

    def test_token_ids_logprob_valid(self):
        """
        Verify that a valid token_ids_logprob request with in-vocabulary
        token IDs is accepted and returns 200.
        """

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/generate",
            headers=headers,
            json={
                "text": "hi",
                "sampling_params": {"max_new_tokens": 1},
                "return_logprob": True,
                "token_ids_logprob": [0],
            },
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
