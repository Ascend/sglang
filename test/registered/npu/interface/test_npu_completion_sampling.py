import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestCompletionSampling(CustomTestCase):
    """Testcase: sampling parameter interactions on /v1/completions.

    Completions differs from chat: temperature/top_p/top_k/min_p use fixed
    defaults (no get_param() / model generation_config), max_tokens defaults
    to 16, logprobs is an int (top-k count), and seed rides the same
    --enable-deterministic-inference gate.

    [Test Category] Interface
    [Test Target] temperature / top_p / top_k / min_p / max_tokens / n / logprobs
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
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
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _complete(self, **kwargs):
        kwargs.setdefault("prompt", "The capital of France is")
        return self.client.completions.create(model=self.model, **kwargs)

    # --- max_tokens default 16 ---

    def test_max_tokens_default_16(self):
        response = self._complete(temperature=0)
        self.assertEqual(
            response.usage.completion_tokens,
            16,
            "completions max_tokens defaults to 16",
        )
        self.assertEqual(response.choices[0].finish_reason, "length")

    # --- temperature interactions ---

    def test_temperature_n2_divergent(self):
        response = self._complete(temperature=2, n=2, max_tokens=32)
        texts = [choice.text for choice in response.choices]
        self.assertEqual(len(texts), 2)
        self.assertNotEqual(texts[0], texts[1], "t=2 + n=2 choices should differ")

    def test_temperature_n2_greedy_identical(self):
        response = self._complete(temperature=0, n=2, max_tokens=32)
        texts = [choice.text for choice in response.choices]
        self.assertEqual(texts[0], texts[1], "t=0 + n=2 choices should be identical")

    def test_temperature_stream_consistent(self):
        non_stream = self._complete(temperature=0, max_tokens=32).choices[0].text
        stream = self._complete(temperature=0, max_tokens=32, stream=True)
        stream_text = "".join(
            chunk.choices[0].text for chunk in stream if chunk.choices
        )
        self.assertEqual(non_stream, stream_text)

    # --- top_p / top_k / min_p interactions ---

    def test_top_k_n2_greedy_identical(self):
        response = self._complete(
            n=2, temperature=1.5, max_tokens=32, extra_body={"top_k": 1}
        )
        texts = [choice.text for choice in response.choices]
        self.assertEqual(texts[0], texts[1], "top_k=1 is greedy: n choices identical")

    def test_min_p_n2(self):
        response = self._complete(n=2, max_tokens=32, extra_body={"min_p": 0.2})
        self.assertEqual(len(response.choices), 2)
        for choice in response.choices:
            self.assertTrue(choice.text)

    # --- logprobs (int type: top-k count) ---

    def test_logprobs_value_1_3_5(self):
        for k in (1, 3, 5):
            with self.subTest(logprobs=k):
                response = self._complete(logprobs=k, max_tokens=32)
                logprobs = response.choices[0].logprobs
                n = len(logprobs.tokens)
                self.assertEqual(len(logprobs.token_logprobs), n)
                self.assertEqual(len(logprobs.top_logprobs), n)
                for top in logprobs.top_logprobs:
                    self.assertGreater(len(top), 0)
                    self.assertLessEqual(
                        len(top), k, "top_logprobs must not exceed the requested k"
                    )

    def test_logprobs_zero(self):
        response = self._complete(logprobs=0, max_tokens=32)
        logprobs = response.choices[0].logprobs
        self.assertTrue(logprobs.tokens)
        self.assertTrue(logprobs.token_logprobs)
        # top-k gated off when k=0: top_logprobs is an empty list
        self.assertEqual(logprobs.top_logprobs, [])

    # --- n x batch ordering ---

    def test_n_batch_batch_major_order(self):
        """prompt list × n: batch-major ordering — choice i belongs to prompt i//n."""
        prompts = ["The capital of France is", "The capital of Germany is"]
        response = self._complete(prompt=prompts, n=2, temperature=0, max_tokens=16)
        self.assertEqual(len(response.choices), 4)
        # Batch-major ordering: choices 0,1 belong to prompt 0; 2,3 to prompt 1.
        self.assertEqual(response.choices[0].text, response.choices[1].text)
        self.assertEqual(response.choices[2].text, response.choices[3].text)
        self.assertNotEqual(response.choices[0].text, response.choices[2].text)


@unittest.skip(
    "The batch_invariant_ops package (registers torch.ops.batch_invariant_ops."
    "npu_* kernels) is an RL-requirement operator that is not open-sourced "
    "and unavailable in the public CI image; the seed path requires manual "
    "testing. Re-enable once the package is available in CI."
)
class TestCompletionSeedDeterministic(CustomTestCase):
    """Testcase: seed on /v1/completions under --enable-deterministic-inference.

    [Test Category] Interface
    [Test Target] seed
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
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
                "--enable-deterministic-inference",
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _complete(self, **kwargs):
        return self.client.completions.create(
            model=self.model,
            prompt="Write a short poem about autumn.",
            temperature=2.0,
            max_tokens=64,
            **kwargs,
        )

    def test_seed_same_identical(self):
        first = self._complete(seed=42).choices[0].text
        second = self._complete(seed=42).choices[0].text
        self.assertEqual(first, second)

    def test_seed_different_differs(self):
        first = self._complete(seed=42).choices[0].text
        second = self._complete(seed=43).choices[0].text
        self.assertNotEqual(first, second)

    def test_seed_stream_consistent(self):
        non_stream = self._complete(seed=42).choices[0].text
        stream = self._complete(seed=42, stream=True)
        stream_text = "".join(
            chunk.choices[0].text for chunk in stream if chunk.choices
        )
        self.assertEqual(non_stream, stream_text)

    def test_top_p_stream_consistent(self):
        non_stream = (
            self._complete(temperature=0.8, top_p=0.1, max_tokens=32, seed=42)
            .choices[0]
            .text
        )
        stream = self._complete(
            temperature=0.8, top_p=0.1, max_tokens=32, seed=42, stream=True
        )
        stream_text = "".join(
            chunk.choices[0].text for chunk in stream if chunk.choices
        )
        self.assertEqual(non_stream, stream_text)


if __name__ == "__main__":
    unittest.main()
