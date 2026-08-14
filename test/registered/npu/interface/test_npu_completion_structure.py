import json
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

register_npu_ci(est_time=500, suite="full-1-npu-a3", nightly=True)


class TestCompletionStructure(CustomTestCase):
    """Testcase: json_schema / regex / ebnf constraints on /v1/completions.

    json_schema is a completions-UNIQUE top-level parameter (chat uses
    response_format instead). All three are mutually exclusive via
    SamplingParams.verify(); response_format silently overwrites the
    top-level json_schema.

    [Test Category] Interface
    [Test Target] json_schema / regex / ebnf
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
        kwargs.setdefault(
            "prompt", "Extract the name and age from: Zhang San is 35 years old."
        )
        return self.client.completions.create(
            model=self.model, temperature=0, max_tokens=64, **kwargs
        )

    # --- json_schema (completions unique) ---

    def test_json_schema_basic(self):
        response = self._complete(
            extra_body={
                "json_schema": (
                    '{"type":"object","properties":{"name":{"type":"string"},'
                    '"age":{"type":"integer"}},"required":["name","age"]}'
                )
            }
        )
        data = json.loads(response.choices[0].text)
        self.assertIn("name", data)
        self.assertIn("age", data)
        self.assertIsInstance(data["age"], int)

    def test_json_schema_ebnf_mutual_exclusion_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._complete(
                extra_body={
                    "json_schema": '{"type":"object"}',
                    "ebnf": 'root ::= "1" | "2"',
                }
            )
        self.assertIn(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set.",
            str(ctx.exception),
        )

    def test_json_schema_regex_mutual_exclusion_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._complete(
                extra_body={"json_schema": '{"type":"object"}', "regex": r"\d+"}
            )
        self.assertIn(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set.",
            str(ctx.exception),
        )

    def test_json_schema_response_format_override(self):
        """response_format silently overwrites the top-level json_schema
        (serving_completions.py:169-178)."""
        response = self._complete(
            extra_body={
                "json_schema": (
                    '{"type":"object","properties":{"name":{"type":"string"}},'
                    '"required":["name"]}'
                ),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "schema": {
                            "type": "object",
                            "properties": {"result": {"type": "integer"}},
                            "required": ["result"],
                        },
                    },
                },
            }
        )
        data = json.loads(response.choices[0].text)
        self.assertIn(
            "result",
            data,
            "response_format schema should win over the top-level json_schema",
        )

    # --- regex ---

    def test_regex_basic(self):
        response = self._complete(
            prompt="Count from 1 to 5.", extra_body={"regex": r"\d+"}
        )
        self.assertTrue(response.choices[0].text.replace(" ", "").isdigit())

    def test_regex_ebnf_mutual_exclusion_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._complete(extra_body={"regex": r"\d+", "ebnf": 'root ::= "1"'})
        self.assertIn(
            "Only one of json_schema, regex, ebnf, or structural_tag can be set.",
            str(ctx.exception),
        )

    def test_regex_stream(self):
        stream = self._complete(
            prompt="Count from 1 to 5.",
            stream=True,
            extra_body={"regex": r"\d+"},
        )
        text = "".join(chunk.choices[0].text for chunk in stream if chunk.choices)
        self.assertTrue(text.replace(" ", "").isdigit())

    # --- ebnf ---

    def test_ebnf_basic(self):
        response = self._complete(extra_body={"ebnf": 'root ::= "1" | "2" | "3"'})
        self.assertIn(response.choices[0].text.strip(), ("1", "2", "3"))

    def test_ebnf_invalid_abort(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._complete(extra_body={"ebnf": 'root ::= "unclosed'})
        self.assertIn("Failed to compile ebnf grammar", str(ctx.exception))


class TestCompletionGrammarBackendNone(CustomTestCase):
    """Testcase: grammar constraints rejected with --grammar-backend none.

    [Test Category] Interface
    [Test Target] --grammar-backend none
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

    def test_json_schema_rejected_with_backend_none(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.completions.create(
                model=self.model,
                prompt="Extract the name.",
                extra_body={"json_schema": '{"type":"object"}'},
            )
        self.assertIn(
            "not supported when the server is launched with --grammar-backend none",
            str(ctx.exception),
        )


if __name__ == "__main__":
    unittest.main()
