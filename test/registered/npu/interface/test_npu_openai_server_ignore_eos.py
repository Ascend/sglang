import unittest

import openai

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

register_npu_ci(est_time=800, suite="nightly-2-npu-a3", nightly=True)


class TestOpenAIServerIgnoreEOS(CustomTestCase):
    """Testcase: Test 'ignore_eos' is True, the EOS is ignored and continue reasoning

    [Test Category] Interface
    [Test Target] ignore_eos
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.other_args = [
            "--attention-backend",
            "ascend",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=cls.other_args,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ignore_eos(self):
        """
        Test that ignore_eos=True allows generation to continue beyond EOS token
        and reach the max_tokens limit.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        max_tokens = 200

        response_default = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": False},
        )

        response_ignore_eos = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Count from 1 to 20."},
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"ignore_eos": True},
        )

        default_tokens = response_default.usage.completion_tokens
        ignore_eos_tokens = response_ignore_eos.usage.completion_tokens

        # ignore_eos=True forces generation to exactly max_tokens (length finish
        # is checked before EOS, and EOS is never a stop condition)
        self.assertEqual(
            ignore_eos_tokens,
            max_tokens,
            f"ignore_eos=True should generate exactly {max_tokens} tokens, got {ignore_eos_tokens}",
        )
        self.assertEqual(
            response_ignore_eos.choices[0].finish_reason,
            "length",
            f"Expected finish_reason='length' for ignore_eos=True, got {response_ignore_eos.choices[0].finish_reason}",
        )

        # ignore_eos=False stops naturally at EOS before max_tokens
        self.assertLess(
            default_tokens,
            max_tokens,
            f"ignore_eos=False should stop before {max_tokens} tokens, got {default_tokens}",
        )
        self.assertEqual(
            response_default.choices[0].finish_reason,
            "stop",
            f"Expected finish_reason='stop' for ignore_eos=False, got {response_default.choices[0].finish_reason}",
        )


if __name__ == "__main__":
    unittest.main()
