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

register_npu_ci(est_time=500, suite="debug-full-1-npu-a3", nightly=True)


class TestCompletionFim(CustomTestCase):
    """Testcase: echo / suffix on /v1/completions with --completion-template.

    suffix is FIM INPUT — it modifies the prompt via the completion template
    and never appears in the output. Known quirk (code_completion_parser.py:85):
    `if request.suffix == ""` does not catch suffix=None, so an omitted suffix
    interpolates the literal string "None" into the FIM prompt.

    [Test Category] Interface
    [Test Target] echo / suffix
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
                "--completion-template",
                "deepseek_coder",
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    PROMPT = "def add(a, b):\n    return a +"

    def _complete(self, **kwargs):
        return self.client.completions.create(
            model=self.model,
            prompt=self.PROMPT,
            temperature=0,
            max_tokens=32,
            **kwargs,
        )

    # --- suffix ---

    def test_suffix_changes_behavior(self):
        without = self._complete().choices[0].text
        with_suffix = self._complete(suffix="b").choices[0].text
        self.assertNotEqual(
            without,
            with_suffix,
            "suffix must change the FIM prompt and thus the generated content",
        )

    def test_suffix_omitted_vs_empty(self):
        """Pin the known quirk: omitted suffix interpolates the literal 'None'
        into the FIM prompt, while suffix='' passes the prompt through —
        the two must produce different outputs."""
        omitted = self._complete().choices[0].text
        empty = self._complete(suffix="").choices[0].text
        self.assertNotEqual(
            omitted,
            empty,
            "omitted suffix injects literal 'None' (code_completion_parser.py:85), "
            "empty suffix passes through",
        )

    def test_suffix_stream(self):
        non_stream = self._complete(suffix="b").choices[0].text
        stream = self._complete(suffix="b", stream=True)
        stream_text = "".join(
            chunk.choices[0].text for chunk in stream if chunk.choices
        )
        self.assertEqual(non_stream, stream_text)

    # --- echo ---

    def test_echo_false(self):
        response = self._complete(echo=False)
        self.assertFalse(
            response.choices[0].text.startswith(self.PROMPT),
            "echo=False must not prepend the prompt",
        )

    def test_echo_true_prepends(self):
        response = self._complete(echo=True)
        self.assertTrue(response.choices[0].text.startswith(self.PROMPT))

    def test_echo_with_suffix(self):
        """echo prepends the original prompt; suffix only shapes generation
        (it is FIM input and never appears in the output)."""
        response = self._complete(echo=True, suffix="b")
        text = response.choices[0].text
        # echo prepends the original prompt; suffix only shapes generation
        # (the model's natural FIM middle for suffix "b" is itself "b").
        self.assertTrue(text.startswith(self.PROMPT))


if __name__ == "__main__":
    unittest.main()
