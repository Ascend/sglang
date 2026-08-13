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

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)

MANY_NEW_TOKENS_PROMPT = """
Please write an extremely detailed and vivid fantasy story, set in a world full of intricate magic systems, political intrigue, and complex characters.
Ensure that you thoroughly describe every scene, character's motivations, and the environment. Include long, engaging dialogues and elaborate on the inner thoughts of the characters.
Each section should be as comprehensive as possible to create a rich and immersive experience for the reader.
The story should span multiple events, challenges, and character developments over time. Aim to make the story at least 3,000 words long.
"""


class TestChatStopInteractions(CustomTestCase):
    """Testcase: stop-family parameter interactions on /v1/chat/completions.

    [Test Category] Interface
    [Test Target] no_stop_trim / min_tokens x stop / ignore_eos x stop_token_ids / stream x stop / n x stop
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
                "--max-running-requests",
                "10",
                "--attention-backend",
                "ascend",
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _chat(self, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": MANY_NEW_TOKENS_PROMPT},
            ],
            temperature=0,
            **kwargs,
        )

    def test_no_stop_trim_false(self):
        """no_stop_trim=False (default): matched stop string is trimmed from output."""
        response = self._chat(
            max_tokens=200, stop="\n", extra_body={"no_stop_trim": False}
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.matched_stop, "\n")
        self.assertFalse(
            choice.message.content.endswith("\n"),
            "stop string should be trimmed when no_stop_trim=False",
        )

    def test_no_stop_trim_true(self):
        """no_stop_trim=True: matched stop string is kept in output."""
        response = self._chat(
            max_tokens=200, stop="\n", extra_body={"no_stop_trim": True}
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.matched_stop, "\n")
        self.assertTrue(
            choice.message.content.endswith("\n"),
            "stop string should be kept when no_stop_trim=True",
        )

    def test_no_stop_trim_stream(self):
        """Stream path: concatenated output keeps the stop string when no_stop_trim=True."""
        stream = self._chat(
            max_tokens=200,
            stop="\n",
            stream=True,
            extra_body={"no_stop_trim": True},
        )
        text = ""
        finish_reason = None
        for chunk in stream:
            if chunk.choices:
                if chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
        self.assertEqual(finish_reason, "stop")
        self.assertTrue(text.endswith("\n"))

    def test_min_tokens_stop_not_protected(self):
        """stop strings are NOT gated by min_tokens: the request can stop before min is reached."""
        response = self._chat(
            max_tokens=200,
            stop="\n",
            extra_body={"min_tokens": 100},
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.matched_stop, "\n")
        self.assertLess(
            response.usage.completion_tokens,
            100,
            "stop string should terminate before min_tokens is satisfied",
        )

    def test_ignore_eos_stop_token_ids_shortcircuit(self):
        """ignore_eos=True short-circuits token-based finish: stop_token_ids are silently ignored."""
        response = self._chat(
            max_tokens=50,
            extra_body={
                "ignore_eos": True,
                "stop_token_ids": [13],
            },
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "length")
        self.assertEqual(response.usage.completion_tokens, 50)

    def test_stream_stop(self):
        """Stream + stop: last chunk carries finish_reason and matched_stop."""
        stream = self._chat(max_tokens=200, stop="\n", stream=True)
        finish_reason = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        self.assertEqual(finish_reason, "stop")

    def test_n2_stop(self):
        """n=2 + stop: each choice independently matches the stop string."""
        response = self._chat(max_tokens=200, stop="\n", n=2)
        self.assertEqual(len(response.choices), 2)
        for choice in response.choices:
            self.assertEqual(choice.finish_reason, "stop")
            self.assertEqual(choice.matched_stop, "\n")

    def test_stop_never_matching(self):
        """A stop string that never appears: generation ends via EOS or
        max_tokens, never via the stop string."""
        response = self._chat(max_tokens=200, stop="ZZZ999ZZZ")
        choice = response.choices[0]
        self.assertIn(
            choice.finish_reason,
            ("stop", "length"),
            "generation should end naturally, not via the stop string",
        )
        self.assertNotEqual(choice.matched_stop, "ZZZ999ZZZ")

    def test_min_tokens_400(self):
        """min_tokens > max_tokens: engine verify() rejects with 400."""
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(
                max_tokens=20,
                extra_body={"min_tokens": 50},
            )
        self.assertIn(
            "min_new_tokens must be in [0, max_new_tokens", str(ctx.exception)
        )

    def test_stop_and_stop_token_ids_combined(self):
        """stop string and stop_token_ids are checked independently — whichever
        hits first terminates the request."""
        response = self._chat(
            max_tokens=200,
            stop="\n",
            extra_body={"stop_token_ids": [13]},
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        # matched_stop is either the string or the token id depending on
        # which condition matched first.
        self.assertIn(choice.matched_stop, ("\n", 13))

    def test_stop_regex_list(self):
        """A list of regex patterns: the first matching pattern in list order wins."""
        response = self._chat(
            max_tokens=200,
            extra_body={"stop_regex": ["\n", "FINISHED"]},
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        # matched_stop echoes the PATTERN itself, not the matched text.
        self.assertIn(choice.matched_stop, ("\n", "FINISHED"))

    def test_ignore_eos_stop_regex_still_works(self):
        """stop_regex is NOT gated by ignore_eos (unlike stop_token_ids which is
        short-circuited) — the regex still terminates the request."""
        response = self._chat(
            max_tokens=200,
            extra_body={"ignore_eos": True, "stop_regex": "\n"},
        )
        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.matched_stop, "\n")


if __name__ == "__main__":
    unittest.main()
