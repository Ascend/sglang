"""Device regression: ordinary MLA NZ writes and cached-prefix inference."""

import unittest

import requests
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMLANZPrefix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)
            cls.process = None

    def _generate(self, text):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _flush(self):
        response = requests.post(self.base_url + "/flush_cache", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)

    def test_nd_nz_cold_warm_and_batched_prefix(self):
        prefix = "def add(a, b):\n    return a + b\n" * 80
        prompts = [
            prefix + "\n# Explain this function.",
            prefix + "\n# Write a test for add(3, 5).\n" * 9,
        ]
        results = []
        for is_nz in (False, True):
            with self.subTest(is_nz=is_nz):
                try:
                    with (
                        envs.SGLANG_USE_FIA_NZ.override(is_nz),
                        envs.SGLANG_NPU_USE_MLAPO.override(False),
                    ):
                        type(self).process = popen_launch_server(
                            DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
                            self.base_url,
                            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                            other_args=[
                                "--attention-backend",
                                "ascend",
                                "--tp-size",
                                "1",
                                "--page-size",
                                "128",
                                "--mem-fraction-static",
                                "0.8",
                                "--chunked-prefill-size",
                                "512",
                                "--cuda-graph-bs",
                                "1",
                                "2",
                            ],
                        )
                    cold = []
                    for prompt in prompts:
                        self._flush()
                        cold.append(self._generate(prompt))
                    self._flush()
                    self._generate(prefix)
                    warm = self._generate(prompts)
                    self.assertEqual(len(warm), 2)
                    for baseline, cached in zip(cold, warm):
                        self.assertGreater(cached["meta_info"]["cached_tokens"], 0)
                        self.assertEqual(cached["output_ids"], baseline["output_ids"])
                    results.append([item["output_ids"] for item in warm])
                finally:
                    self.tearDownClass()
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
