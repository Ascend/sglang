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

register_npu_ci(est_time=600, suite="debug-full-1-npu-a3", nightly=True)

STORY_PROMPT = (
    "Please write a very long fantasy story with many paragraphs and dialogues. "
    "Make every paragraph as long as possible."
)


class TestCompletionStopFamily(CustomTestCase):
    """Testcase: stop-family parameters on /v1/completions.

    Completions has no chat template stop_str: without user stop only EOS
    stops generation. ignore_eos short-circuits stop_token_ids but not stop
    strings/regex; min_tokens does not protect stop strings.

    [Test Category] Interface
    [Test Target] stop / stop_token_ids / stop_regex / no_stop_trim / ignore_eos / min_tokens
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
        return self.client.completions.create(
            model=self.model,
            prompt=STORY_PROMPT,
            temperature=0,
            **kwargs,
        )

    # --- stop ---

    def test_stop_list(self):
        response = self._complete(stop=["END", "."], max_tokens=200)
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIn(choice.matched_stop, ("END", "."))

    def test_stop_stream(self):
        stream = self._complete(stop="\n", max_tokens=200, stream=True)
        finish_reason = None
        matched = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                matched = chunk.choices[0].matched_stop
        self.assertEqual(finish_reason, "stop")
        self.assertEqual(matched, "\n")

    def test_stop_n2(self):
        response = self._complete(stop="\n", max_tokens=200, n=2)
        self.assertEqual(len(response.choices), 2)
        for choice in response.choices:
            self.assertEqual(choice.finish_reason, "stop")
            self.assertEqual(choice.matched_stop, "\n")

    # --- stop_token_ids ---

    def test_stop_token_ids_stream(self):
        stream = self._complete(
            max_tokens=200, stream=True, extra_body={"stop_token_ids": [13]}
        )
        finish_reason = None
        matched = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                matched = chunk.choices[0].matched_stop
        self.assertEqual(finish_reason, "stop")
        self.assertEqual(matched, 13)

    def test_stop_token_ids_n2(self):
        response = self._complete(
            max_tokens=200, n=2, extra_body={"stop_token_ids": [13]}
        )
        for choice in response.choices:
            self.assertEqual(choice.finish_reason, "stop")
            self.assertEqual(choice.matched_stop, 13)

    # --- stop_regex ---

    def test_stop_regex_list(self):
        response = self._complete(
            max_tokens=200, extra_body={"stop_regex": ["\n", "FINISHED"]}
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        # matched_stop echoes the pattern, not the matched text
        self.assertIn(choice.matched_stop, ("\n", "FINISHED"))

    def test_stop_regex_stream(self):
        stream = self._complete(
            max_tokens=200, stream=True, extra_body={"stop_regex": "and|or"}
        )
        finish_reason = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        self.assertEqual(finish_reason, "stop")

    # --- no_stop_trim ---

    def test_no_stop_trim_false(self):
        response = self._complete(
            stop="\n", max_tokens=200, extra_body={"no_stop_trim": False}
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.matched_stop, "\n")
        self.assertFalse(choice.text.endswith("\n"))

    def test_no_stop_trim_true(self):
        response = self._complete(
            stop="\n", max_tokens=200, extra_body={"no_stop_trim": True}
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.matched_stop, "\n")
        self.assertTrue(choice.text.endswith("\n"))

    def test_no_stop_trim_stream(self):
        stream = self._complete(
            stop="\n", max_tokens=200, stream=True, extra_body={"no_stop_trim": True}
        )
        text = "".join(chunk.choices[0].text for chunk in stream if chunk.choices)
        self.assertTrue(text.endswith("\n"))

    # --- ignore_eos ---

    def test_ignore_eos_exact_length(self):
        response = self._complete(max_tokens=50, extra_body={"ignore_eos": True})
        self.assertEqual(response.usage.completion_tokens, 50)
        self.assertEqual(response.choices[0].finish_reason, "length")

    def test_ignore_eos_false_natural_stop(self):
        response = self.client.completions.create(
            model=self.model,
            prompt="Count from 1 to 20.",
            temperature=0,
            max_tokens=200,
        )
        self.assertLess(response.usage.completion_tokens, 200)
        self.assertEqual(response.choices[0].finish_reason, "stop")

    def test_ignore_eos_stop_string_still_works(self):
        response = self._complete(
            stop="\n", max_tokens=200, extra_body={"ignore_eos": True}
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.matched_stop, "\n")

    def test_ignore_eos_stop_token_ids_ignored(self):
        response = self._complete(
            max_tokens=50,
            extra_body={"ignore_eos": True, "stop_token_ids": [13]},
        )
        self.assertEqual(response.usage.completion_tokens, 50)
        self.assertEqual(response.choices[0].finish_reason, "length")

    # --- min_tokens ---

    def test_min_tokens_basic(self):
        response = self._complete(max_tokens=32, extra_body={"min_tokens": 5})
        self.assertGreaterEqual(response.usage.completion_tokens, 5)

    def test_min_tokens_stop_not_protected(self):
        response = self._complete(
            stop="\n", max_tokens=200, extra_body={"min_tokens": 100}
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertLess(
            response.usage.completion_tokens,
            100,
            "stop string terminates before min_tokens is satisfied",
        )

    def test_min_tokens_stream(self):
        stream = self._complete(
            max_tokens=32,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"min_tokens": 5},
        )
        # usage chunk carries completion_tokens
        total = None
        for chunk in stream:
            if chunk.usage:
                total = chunk.usage.completion_tokens
        self.assertIsNotNone(total)
        self.assertGreaterEqual(total, 5)


if __name__ == "__main__":
    unittest.main()
