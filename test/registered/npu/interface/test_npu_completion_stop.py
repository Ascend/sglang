import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
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

        # Token ids for the strings the stop tests force into the output:
        # temperature-0 greedy trajectories differ across platforms, so a
        # test that waits for the model to emit the stop string by chance
        # cannot distinguish a product bug from a missed trajectory.
        cls.tokenizer = get_tokenizer(cls.model)
        cls.newline_id = cls.tokenizer.encode("\n", add_special_tokens=False)[-1]
        cls.period_id = cls.tokenizer.encode(".", add_special_tokens=False)[-1]
        cls.and_id = cls.tokenizer.encode(" and", add_special_tokens=False)[-1]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _complete(self, **kwargs):
        return self.client.completions.create(
            model=self.model,
            prompt=kwargs.pop("prompt", STORY_PROMPT),
            temperature=0,
            **kwargs,
        )

    # --- stop ---

    def test_stop_list(self):
        response = self._complete(
            stop=["END", "."],
            max_tokens=200,
            logit_bias={str(self.period_id): 100},
        )
        choice = response.choices[0]
        echo = response.model_dump_json()
        self.assertEqual(choice.finish_reason, "stop", msg=echo)
        self.assertIn(choice.matched_stop, ("END", "."), msg=echo)

    def test_stop_stream(self):
        stream = self._complete(
            stop="\n",
            max_tokens=200,
            stream=True,
            logit_bias={str(self.newline_id): 100},
        )
        text = ""
        finish_reason = None
        matched = None
        for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].text
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                    matched = chunk.choices[0].matched_stop
        self.assertEqual(finish_reason, "stop", msg=text)
        self.assertEqual(matched, "\n", msg=text)

    def test_stop_n2(self):
        response = self._complete(
            stop="\n",
            max_tokens=200,
            n=2,
            logit_bias={str(self.newline_id): 100},
        )
        self.assertEqual(len(response.choices), 2, msg=response.model_dump_json())
        for choice in response.choices:
            self.assertEqual(choice.finish_reason, "stop", msg=response.model_dump_json())
            self.assertEqual(choice.matched_stop, "\n", msg=response.model_dump_json())

    # --- stop_token_ids ---

    def test_stop_token_ids_stream(self):
        # The token must be emitted for the id-based finish to fire: force
        # it instead of relying on the greedy trajectory to produce it.
        stream = self._complete(
            max_tokens=200,
            stream=True,
            extra_body={"stop_token_ids": [self.newline_id]},
            logit_bias={str(self.newline_id): 100},
        )
        text = ""
        finish_reason = None
        matched = None
        for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].text
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                    matched = chunk.choices[0].matched_stop
        self.assertEqual(finish_reason, "stop", msg=text)
        self.assertEqual(matched, self.newline_id, msg=text)

    def test_stop_token_ids_n2(self):
        response = self._complete(
            max_tokens=200,
            n=2,
            extra_body={"stop_token_ids": [self.newline_id]},
            logit_bias={str(self.newline_id): 100},
        )
        for choice in response.choices:
            self.assertEqual(choice.finish_reason, "stop", msg=response.model_dump_json())
            self.assertEqual(choice.matched_stop, self.newline_id, msg=response.model_dump_json())

    # --- stop_regex ---

    def test_stop_regex_list(self):
        response = self._complete(
            max_tokens=200,
            extra_body={"stop_regex": ["\n", "FINISHED"]},
            logit_bias={str(self.newline_id): 100},
        )
        choice = response.choices[0]
        echo = response.model_dump_json()
        self.assertEqual(choice.finish_reason, "stop", msg=echo)
        # matched_stop echoes the pattern, not the matched text
        self.assertIn(choice.matched_stop, ("\n", "FINISHED"), msg=echo)

    def test_stop_regex_stream(self):
        stream = self._complete(
            max_tokens=200,
            stream=True,
            extra_body={"stop_regex": "and|or"},
            logit_bias={str(self.and_id): 100},
        )
        text = ""
        finish_reason = None
        for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].text
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
        self.assertEqual(finish_reason, "stop", msg=text)

    # --- no_stop_trim ---

    def test_no_stop_trim_false(self):
        response = self._complete(
            stop="\n",
            max_tokens=200,
            extra_body={"no_stop_trim": False},
            logit_bias={str(self.newline_id): 100},
        )
        choice = response.choices[0]
        echo = response.model_dump_json()
        self.assertEqual(choice.finish_reason, "stop", msg=echo)
        self.assertEqual(choice.matched_stop, "\n", msg=echo)
        self.assertFalse(choice.text.endswith("\n"), msg=echo)

    def test_no_stop_trim_true(self):
        response = self._complete(
            stop="\n",
            max_tokens=200,
            extra_body={"no_stop_trim": True},
            logit_bias={str(self.newline_id): 100},
        )
        choice = response.choices[0]
        echo = response.model_dump_json()
        self.assertEqual(choice.finish_reason, "stop", msg=echo)
        self.assertEqual(choice.matched_stop, "\n", msg=echo)
        self.assertTrue(choice.text.endswith("\n"), msg=echo)

    def test_no_stop_trim_stream(self):
        stream = self._complete(
            stop="\n",
            max_tokens=200,
            stream=True,
            extra_body={"no_stop_trim": True},
            logit_bias={str(self.newline_id): 100},
        )
        text = "".join(chunk.choices[0].text for chunk in stream if chunk.choices)
        self.assertTrue(text.endswith("\n"), msg=text)

    # --- ignore_eos ---

    def test_ignore_eos_exact_length(self):
        response = self._complete(max_tokens=50, extra_body={"ignore_eos": True})
        echo = response.model_dump_json()
        self.assertEqual(response.usage.completion_tokens, 50, msg=echo)
        self.assertEqual(response.choices[0].finish_reason, "length", msg=echo)

    def test_ignore_eos_false_natural_stop(self):
        """Without ignore_eos the request stops on EOS or max_tokens — the
        instruct model without a chat template does not guarantee an early EOS."""
        response = self.client.completions.create(
            model=self.model,
            prompt="Count from 1 to 20.",
            temperature=0,
            max_tokens=200,
        )
        self.assertIn(response.choices[0].finish_reason, ("stop", "length"))

    def test_ignore_eos_stop_string_still_works(self):
        # The structured prompt constrains the newline to appear inside
        # normal output; under ignore_eos only the string matcher can
        # stop the request — the assertion probes that path directly.
        response = self._complete(
            prompt="List three colors, one per line.",
            stop="\n",
            max_tokens=600,
            extra_body={"ignore_eos": True},
        )
        choice = response.choices[0]
        echo = response.model_dump_json()
        self.assertEqual(choice.finish_reason, "stop", msg=echo)
        self.assertEqual(choice.matched_stop, "\n", msg=echo)

    def test_ignore_eos_stop_token_ids_ignored(self):
        response = self._complete(
            max_tokens=50,
            extra_body={"ignore_eos": True, "stop_token_ids": [self.newline_id]},
        )
        echo = response.model_dump_json()
        self.assertEqual(response.usage.completion_tokens, 50, msg=echo)
        self.assertEqual(response.choices[0].finish_reason, "length", msg=echo)

    # --- min_tokens ---

    def test_min_tokens_basic(self):
        response = self._complete(max_tokens=32, extra_body={"min_tokens": 5})
        self.assertGreaterEqual(response.usage.completion_tokens, 5)

    def test_min_tokens_stop_not_protected(self):
        # Forced newline at token 1: if min_tokens gated the string matcher,
        # the request would run to >= 100 tokens; stopping at once proves
        # stop strings are not gated.
        response = self._complete(
            stop="\n",
            max_tokens=200,
            extra_body={"min_tokens": 100},
            logit_bias={str(self.newline_id): 100},
        )
        choice = response.choices[0]
        echo = response.model_dump_json()
        self.assertEqual(choice.finish_reason, "stop", msg=echo)
        self.assertLess(
            response.usage.completion_tokens,
            100,
            f"stop string terminates before min_tokens is satisfied; {echo}",
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
