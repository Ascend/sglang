import json
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_EAGLE3_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUConstrainedDecodingSpecReasoning(CustomTestCase):
    """Test EAGLE3 speculative reasoning with constrained JSON decoding on NPU.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-algorithm=EAGLE3; --speculative-draft-model-path;
    --speculative-num-steps; --speculative-eagle-topk; --speculative-num-draft-tokens;
    --reasoning-parser; --response-format=json_schema
    """

    json_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[\\w]+$"},
                "population": {"type": "integer"},
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "has_held_olympics": {"type": "boolean"},
            },
            "required": ["name", "population", "languages", "has_held_olympics"],
            "additionalProperties": False,
        }
    )

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.7",
            "--reasoning-parser",
            "qwen3",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "8",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_json_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Introduce the capital of France. Return in a JSON format. "
                    "The JSON Schema is: " + json.dumps(self.json_schema),
                },
            ],
            temperature=0,
            max_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": json.loads(self.json_schema)},
            },
        )
        text = response.choices[0].message.content

        self.assertIsNotNone(text)
        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            self.fail(f"Failed to parse JSON: {text}")

        self.assertIsInstance(js_obj["name"], str)
        self.assertIsInstance(js_obj["population"], int)


if __name__ == "__main__":
    unittest.main()
