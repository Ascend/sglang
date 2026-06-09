import json
import unittest

import openai
import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
# from sglang.test.ascend.test_ascend_utils import (
#     QWEN3_8B_WEIGHTS_PATH,
# )
QWEN3_8B_WEIGHTS_PATH="/home/weights/Qwen3-8B"
register_cuda_ci(est_time=509, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationAccuracy(PauseResumeInPlaceMixin, PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.pause_generate_url = cls.lb_url
        cls.pause_target_urls = [cls.prefill_url, cls.decode_url]
        cls.launch_all()


    def test_logprob(self):
        prompt = "The capital of france is "
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
        )

        j = response.json()
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        assert (
                len(output_logprobs) == completion_tokens
        ), f"output_logprobs and completion_tokens should have the same length, but got {len(output_logprobs)} and {completion_tokens}"
        assert (
                len(input_logprobs) > 0
        ), f"input_logprobs should have at least one token, but got {len(input_logprobs)}"

    def test_chat_completion_top_logprobs(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        response = client.chat.completions.create(
            model="dummy",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            max_tokens=8,
            logprobs=True,
            top_logprobs=5,
        )

        self.assertIsNotNone(response.choices[0].logprobs)
        content_logprobs = response.choices[0].logprobs.content
        self.assertGreater(len(content_logprobs), 0)

        first_top_logprobs = next(
            (item.top_logprobs for item in content_logprobs if item.top_logprobs),
            None,
        )
        self.assertIsNotNone(first_top_logprobs)
        self.assertGreater(len(first_top_logprobs), 0)
        self.assertIsInstance(first_top_logprobs[0].token, str)

    def test_structured_output(self):
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )

        # JSON
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "Here is the information of the capital of France in the JSON format.\n",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "json_schema": json_schema,
                },
            },
        )
        output = response.json()["text"]
        # ensure the output is a valid JSON
        json.loads(output)

    def test_first_token_finish(self):
        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        eos_token = tokenizer.eos_token_id
        prompt = "The best programming language for AI is"

        # First token EOS
        res = client.completions.create(
            model="dummy", prompt=prompt, logit_bias={eos_token: 42}
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is EOS, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token EOS with ignore_eos
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={eos_token: 42},
            extra_body={"ignore_eos": True},
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] > 1, (
            "Expected completion_tokens to be greater than 1 when ignore_eos is True, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token with specified stop token
        stop_token_id = tokenizer.encode(" hello", add_special_tokens=False)[0]
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={stop_token_id: 42},
            stop=[" hello"],
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is stop token, "
            f"but got {res['usage']['completion_tokens']}"
        )


if __name__ == "__main__":
    unittest.main()
