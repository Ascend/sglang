import json
import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="debug-full-1-npu-a3", nightly=True)


class TestCompletionsPenalty(CustomTestCase):
    """Testcase: frequency_penalty / presence_penalty standalone effects on /v1/completions.

    The chat endpoint has functional coverage (test_npu_penalty.py) but the
    completions endpoint has zero penalty coverage. Each penalty is verified
    standalone with repetition_penalty pinned to 1.0 (disabled).

    [Test Category] Interface
    [Test Target] frequency_penalty / presence_penalty
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

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _complete(self, payload):
        return requests.post(
            f"{self.base_url}/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ).json()

    @staticmethod
    def _vocab_diversity(text):
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 1.0
        return len(set(words)) / len(words)

    def _diversity(self, prompt, penalty_key, penalty_value):
        """Average vocab diversity over 5 seeded runs."""
        diversities = []
        for i in range(10):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.8,
                "max_tokens": 150,
                "seed": 42 + i,
                "repetition_penalty": 1.0,
            }
            if penalty_value is not None:
                payload[penalty_key] = penalty_value
            text = self._complete(payload)["choices"][0]["text"]
            diversities.append(self._vocab_diversity(text))
        return sum(diversities) / len(diversities)

    def test_frequency_penalty_standalone(self):
        prompt = (
            "Write exactly 10 very small sentences, each containing the word "
            "'data'. Use the word 'data' as much as possible."
        )
        baseline = self._diversity(prompt, "frequency_penalty", None)
        penalized = self._diversity(prompt, "frequency_penalty", 1.99)
        self.assertGreater(
            penalized,
            baseline,
            f"frequency_penalty should increase diversity: {baseline:.3f} → {penalized:.3f}",
        )

    def test_presence_penalty_standalone(self):
        prompt = (
            "Write the word 'machine learning' exactly 20 times in a row, "
            "separated by spaces."
        )
        baseline = self._diversity(prompt, "presence_penalty", None)
        penalized = self._diversity(prompt, "presence_penalty", 1.99)
        self.assertGreater(
            penalized,
            baseline,
            f"presence_penalty should increase diversity: {baseline:.3f} → {penalized:.3f}",
        )

    def test_stream_penalty_consistent(self):
        """Stream path: penalty lives in the engine sampler; concat stream output
        should have comparable diversity to non-stream."""
        prompt = (
            "Write exactly 10 very small sentences, each containing the word "
            "'data'. Use the word 'data' as much as possible."
        )
        non_stream = self._diversity(prompt, "frequency_penalty", 1.99)

        stream_diversities = []
        for i in range(10):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.8,
                "max_tokens": 150,
                "seed": 42 + i,
                "repetition_penalty": 1.0,
                "frequency_penalty": 1.99,
                "stream": True,
            }
            response = requests.post(
                f"{self.base_url}/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                stream=True,
            )
            text = ""
            for line in response.iter_lines():
                if line and line.startswith(b"data: ") and line != b"data: [DONE]":
                    chunk = json.loads(line[6:])
                    if chunk["choices"] and chunk["choices"][0]["text"]:
                        text += chunk["choices"][0]["text"]
            stream_diversities.append(self._vocab_diversity(text))

        stream_avg = sum(stream_diversities) / len(stream_diversities)
        self.assertGreater(
            stream_avg,
            0,
            "stream penalty runs should produce measurable diversity",
        )
        # Stream and non-stream diversity should be in the same ballpark.
        self.assertGreater(
            stream_avg,
            non_stream * 0.5,
            f"stream diversity {stream_avg:.3f} unexpectedly low vs non-stream {non_stream:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
