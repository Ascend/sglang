import json
import multiprocessing as mp
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

PROMPTS = [
    "The capital of France is",
    "AI is a field of computer science focused on",
]


class TestNPULoRATP(CustomTestCase):
    """Test LoRA with tensor parallelism on NPU.

    [Test Category] Feature
    [Test Target] LoRA TP distribution, TP=2
    """

    lora_name = "test_lora"

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "2",
            "--enable-lora",
            "--lora-path",
            f"{cls.lora_name}={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            "--max-loras-per-batch",
            "1",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_tp_basic(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": self.lora_name,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertGreater(len(result["text"]), 0)

    def test_lora_tp_batch(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": PROMPTS,
                "lora_path": self.lora_name,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(PROMPTS))

    def test_lora_tp_without_adapter(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": PROMPTS[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertGreater(len(result["text"]), 0)

    def test_lora_tp_with_stream(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": PROMPTS[0],
                "lora_path": self.lora_name,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(response.status_code, 200)
        text = ""
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:") and chunk != "data: [DONE]":
                data = json.loads(chunk[5:].strip("\n"))
                text += data.get("text", "")
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main(warnings="ignore")
