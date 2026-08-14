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

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestChatGrammarMatrix(CustomTestCase):
    """Testcase: grammar constraint mutual exclusion and dispatch behavior.

    [Test Category] Interface
    [Test Target] regex / ebnf / response_format(json_schema, structural_tag) interactions
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

    def _chat(self, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            temperature=0,
            max_tokens=50,
            **kwargs,
        )

    def _json_schema_format(self):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "integer"}},
                    "required": ["result"],
                },
            },
        }

    def test_regex_ebnf_mutual_exclusion_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(
                extra_body={
                    "regex": r"\d+",
                    "ebnf": 'root ::= "1" | "2"',
                }
            )
        self.assertIn(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set.",
            str(ctx.exception),
        )

    def test_json_schema_ebnf_mutual_exclusion_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(
                response_format=self._json_schema_format(),
                extra_body={"ebnf": 'root ::= "1" | "2"'},
            )
        self.assertIn(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set.",
            str(ctx.exception),
        )

    def test_json_schema_regex_mutual_exclusion_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(
                response_format=self._json_schema_format(),
                extra_body={"regex": r"\d+"},
            )
        self.assertIn(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set.",
            str(ctx.exception),
        )

    def test_invalid_ebnf_abort(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(extra_body={"ebnf": 'root ::= "unclosed'})
        self.assertIn("Failed to compile ebnf grammar", str(ctx.exception))

    def test_invalid_regex_abort(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(extra_body={"regex": "("})
        self.assertIn("Failed to compile regex grammar", str(ctx.exception))

    def test_structural_tag_regex_mutual_exclusion_400(self):
        """structural_tag IS in the mutual exclusion list (latest business
        code): regex + structural_tag is rejected with 400."""
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._chat(
                extra_body={"regex": r"\d+"},
                response_format={
                    "type": "structural_tag",
                    "format": {"type": "const_string", "value": "<answer>"},
                },
            )
        self.assertIn(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set.",
            str(ctx.exception),
        )


class TestChatGrammarBackendNone(CustomTestCase):
    """Testcase: grammar constraints are rejected when launched with --grammar-backend none.

    [Test Category] Interface
    [Test Target] --grammar-backend none negative path
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
                "--grammar-backend",
                "none",
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_ebnf_rejected_with_backend_none(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Count from 1 to 5."}],
                extra_body={"ebnf": 'root ::= "1" | "2"'},
            )
        self.assertIn(
            "not supported when the server is launched with --grammar-backend none",
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
