"""
Test HiCache with EAGLE speculative decoding variant on Ascend NPU.

Tests HiCache combined with EAGLE3 speculative decoding to verify
cache reuse and speculative accept length meet thresholds.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct

Note: Skip if EAGLE3 is not supported on current NPU aiter version.

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_eagle_variant.py -v
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


@unittest.skip("EAGLE3 speculative decoding not yet supported on Ascend NPU")
class TestNPUHiCacheEagle(CustomTestCase):
    """Test HiCache with EAGLE speculative decoding on NPU.

    [Test Category] HiCache
    [Test Target] HiCache + EAGLE3 speculative decoding integration
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.hicache_args = [
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "1.2",
            "--mem-fraction-static",
            "0.7",
            "--hicache-io-backend",
            "kernel_ascend",
            "--hicache-mem-layout",
            "page_first_direct",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            cls.model,
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.hicache_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_hicache_eagle_cache_reuse(self):
        """Test that HiCache works correctly with EAGLE speculative decoding."""
        import requests

        prompt = "Explain the concept of machine learning in detail. " * 20

        for i in range(2):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 50,
                        "ignore_eos": True,
                    },
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200)

            meta_info = response.json().get("meta_info", {})
            cached_tokens = int(meta_info.get("cached_tokens", 0))
            spec_accept_length = float(meta_info.get("spec_accept_length", 0))

            print(
                f"Request {i+1}: cached_tokens={cached_tokens}, spec_accept_length={spec_accept_length:.2f}"
            )

            if i == 0:
                self.assertEqual(cached_tokens, 0, "First request should have no cache")
                self.assertGreater(
                    spec_accept_length,
                    1.5,
                    "First request spec_accept_length should be reasonable",
                )
            else:
                self.assertGreater(
                    cached_tokens,
                    0,
                    "Second request should have cache hit",
                )
                self.assertGreater(
                    spec_accept_length,
                    1.5,
                    "Second request spec_accept_length should be maintained",
                )


if __name__ == "__main__":
    unittest.main()
