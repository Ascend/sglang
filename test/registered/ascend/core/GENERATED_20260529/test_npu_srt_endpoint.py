import json
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestSRTEndpoint(CustomTestCase):
    """Test SRT endpoint functionality on NPU.

    [Test Category] Core
    [Test Target] endpoint API, logprobs, custom logit processor
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--enable-custom-logit-processor",
                "--mem-fraction-static",
                "0.3",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(
        self,
        return_logprob=False,
        top_logprobs_num=0,
        return_text=False,
        n=1,
        stream=False,
        batch=False,
    ):
        if batch:
            text = ["The capital of France is"]
        else:
            text = "The capital of France is"

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 16,
                    "n": n,
                },
                "stream": stream,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "return_text_in_logprobs": return_text,
                "logprob_start_len": 0,
            },
        )
        if not stream:
            response_json = response.json()
        else:
            response_json = []
            for line in response.iter_lines():
                if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                    response_json.append(json.loads(line[6:]))

    def test_simple_decode(self):
        self.run_decode()

    def test_simple_decode_batch(self):
        self.run_decode(batch=True)

    def test_parallel_sample(self):
        self.run_decode(n=3)

    def test_parallel_sample_stream(self):
        self.run_decode(n=3, stream=True)

    def test_logprob(self):
        self.run_decode(
            return_logprob=True,
            top_logprobs_num=5,
            return_text=True,
        )

    def test_logprob_start_len(self):
        logprob_start_len = 4
        new_tokens = 4
        prompts = [
            "I have a very good idea on",
            "Today is a sunndy day and",
        ]

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": new_tokens,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "return_text_in_logprobs": True,
                "logprob_start_len": logprob_start_len,
            },
        )
        response_json = response.json()

        for i, res in enumerate(response_json):
            self.assertEqual(
                res["meta_info"]["prompt_tokens"],
                logprob_start_len + len(res["meta_info"]["input_token_logprobs"]),
            )
            assert prompts[i].endswith(
                "".join([x[-1] for x in res["meta_info"]["input_token_logprobs"]])
            )

            self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)
            self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)
            self.assertEqual(
                res["text"],
                "".join([x[-1] for x in res["meta_info"]["output_token_logprobs"]]),
            )

    def test_logprob_with_chunked_prefill(self):
        new_tokens = 4
        prompts = "I have a very good idea on this. " * 8000

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": new_tokens,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
                "top_logprobs_num": 5,
            },
        )
        response_json = response.json()

        res = response_json
        self.assertEqual(res["meta_info"]["completion_tokens"], new_tokens)

        self.assertEqual(len(res["meta_info"]["output_token_logprobs"]), new_tokens)
        self.assertEqual(len(res["meta_info"]["output_top_logprobs"]), new_tokens)

        for i in range(new_tokens):
            self.assertListEqual(
                res["meta_info"]["output_token_logprobs"][i],
                res["meta_info"]["output_top_logprobs"][i][0],
            )
            self.assertEqual(len(res["meta_info"]["output_top_logprobs"][i]), 5)

    def test_cache_tokens(self):
        for _ in range(2):
            time.sleep(1)
            response = requests.post(self.base_url + "/flush_cache")
            assert response.status_code == 200

        def send_and_check_cached_tokens(input_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": list(input_ids),
                    "sampling_params": {
                        "max_new_tokens": 1,
                    },
                },
            )
            response_json = response.json()
            return response_json["meta_info"]["cached_tokens"]

        self.assertEqual(send_and_check_cached_tokens(range(0, 100)), 0)
        self.assertEqual(send_and_check_cached_tokens(range(0, 10000)), 100)
        self.assertEqual(send_and_check_cached_tokens(range(0, 10000)), 9999)

    def test_get_server_info(self):
        response = requests.get(self.base_url + "/get_server_info")
        self.assertEqual(response.status_code, 200)

    def test_logit_bias(self):
        prompt = "The capital of France is"
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "logit_bias": {str(100): 100.0},
                },
            },
        )
        response_json = response.json()
        self.assertEqual(response.status_code, 200)

    def test_forbidden_token(self):
        prompt = "The capital of France is"
        forbidden_token_id = 100
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "logit_bias": {str(forbidden_token_id): -100.0},
                },
            },
        )
        response_json = response.json()
        self.assertEqual(response.status_code, 200)

    def run_custom_logit_processor(self, target_token_id: Optional[int] = None):
        custom_params = {"token_id": target_token_id}

        class DeterministicLogitProcessor(CustomLogitProcessor):
            def __call__(self, logits, custom_param_list):
                assert logits.shape[0] == len(custom_param_list)
                key = "token_id"

                for i, param_dict in enumerate(custom_param_list):
                    logits[i, :] = -float("inf")
                    logits[i, param_dict[key]] = 0.0
                return logits

        prompts = "Question: Is Paris the Capital of France? Answer:"

        base_json = {
            "text": prompts,
            "sampling_params": {"temperature": 0.0},
            "return_logprob": True,
        }

        custom_json = base_json.copy()
        if target_token_id is not None:
            custom_json["custom_logit_processor"] = DeterministicLogitProcessor.to_str()
            custom_json["sampling_params"]["custom_params"] = custom_params

        custom_response = requests.post(
            self.base_url + "/generate",
            json=custom_json,
        ).json()

        output_token_logprobs = custom_response["meta_info"]["output_token_logprobs"]
        sampled_tokens = [x[1] for x in output_token_logprobs]

        if target_token_id is not None:
            self.assertTrue(
                all(x == custom_params["token_id"] for x in sampled_tokens),
                f"{target_token_id=}\n{sampled_tokens=}\n{custom_response=}",
            )

    def test_custom_logit_processor(self):
        self.run_custom_logit_processor(target_token_id=5)

    def test_custom_logit_processor_batch_mixed(self):
        target_token_ids = list(range(32)) + [None] * 16
        random.shuffle(target_token_ids)
        with ThreadPoolExecutor(len(target_token_ids)) as executor:
            list(executor.map(self.run_custom_logit_processor, target_token_ids))


class TestTokenizeDetokenize(CustomTestCase):
    """Test tokenize and detokenize endpoints on NPU.

    [Test Category] Core
    [Test Target] tokenize API, detokenize API
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.3",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tokenize_various_inputs(self):
        inputs = ["Hello world", "This is a test", "Another input"]
        response = requests.post(
            self.base_url + "/tokenize",
            json={"text": inputs},
        )
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(len(response_json), len(inputs))

    def test_tokenize_invalid_type(self):
        response = requests.post(
            self.base_url + "/tokenize",
            json={"text": 123},
        )
        self.assertNotEqual(response.status_code, 200)

    def test_detokenize_roundtrip(self):
        tokenizer = get_tokenizer(self.model)
        text = "Hello world"
        token_ids = tokenizer.encode(text)

        response = requests.post(
            self.base_url + "/detokenize",
            json={"input_ids": token_ids},
        )
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["text"], text)

    def test_detokenize_invalid_tokens(self):
        response = requests.post(
            self.base_url + "/detokenize",
            json={"input_ids": [-1, -2, -3]},
        )
        self.assertNotEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
