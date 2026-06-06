import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=254, suite="nightly-1-npu-a3", nightly=True)


DEFAULT_NGRAM_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "16",
    "--mem-fraction-static",
    "0.7",
]


class TestNPU_NgramSpeculativeDecodingSam(CustomTestCase):
    """Test NGRAM speculative decoding with external SAM corpus on ascend backend.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-algorithm=NGRAM;
    --speculative-ngram-external-sam-budget; /add_external_corpus HTTP API;
    avg_spec_accept_length boost
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def get_server_args(cls):
        return DEFAULT_NGRAM_SERVER_ARGS + [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--speculative-ngram-external-sam-budget",
            "8",
        ]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_output_as_corpus_boosts_accept_length(self):
        prompts = [
            "The capital of France is",
            "In mathematics, the Pythagorean theorem states that",
            "The speed of light in a vacuum is approximately",
            "Water boils at a temperature of",
            "The largest planet in our solar system is",
        ]
        max_new_tokens = 128
        num_rounds = 3

        def generate_batch():
            outputs = []
            for prompt in prompts:
                resp = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": max_new_tokens,
                        },
                    },
                    timeout=120,
                )
                self.assertEqual(resp.status_code, 200, resp.text)
                outputs.append(resp.json()["text"])
            return outputs

        def get_accept_length():
            info = requests.get(self.base_url + "/server_info", timeout=10).json()
            return info["internal_states"][0]["avg_spec_accept_length"]

        generated_outputs = []
        for _ in range(num_rounds):
            generated_outputs = generate_batch()
        baseline_accept_len = get_accept_length()

        requests.post(self.base_url + "/flush_cache", timeout=30)

        resp = requests.post(
            self.base_url + "/add_external_corpus",
            json={"corpus_id": "bench", "documents": generated_outputs},
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["success"], resp.json().get("message"))

        for _ in range(num_rounds):
            generate_batch()
        sam_accept_len = get_accept_length()

        self.assertGreater(
            sam_accept_len,
            baseline_accept_len * 2.0,
            f"SAM accept length ({sam_accept_len:.2f}) should be at least 2x "
            f"baseline ({baseline_accept_len:.2f}) when corpus matches output",
        )


if __name__ == "__main__":
    unittest.main()
