import json
import os
import random
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Dict
from urllib.parse import urlparse

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHiCacheStorageAccuracy(CustomTestCase):
    """Testcase: GSM8K accuracy test with HiCache storage persistence on NPU.

    Tests that GSM8K accuracy remains consistent after cache flush,
    demonstrating proper storage backend persistence.

    [Test Category] HiCache
    [Test Target] --hicache-storage-backend file + accuracy persistence
    """

    accuracy_threshold = 0.03
    initial_accuracy_threshold = 0.50

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)

        cls.tokenizer = get_tokenizer(cls.model)

        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
        }
        other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            "128",
            "--enable-cache-report",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-backend-extra-config",
            json.dumps(extra_config),
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        env_vars = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        }

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env_vars,
        )
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process,
        )

        print(f"Test server launched successfully at {cls.base_url}")
        print(f"Cache directory: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

        import shutil

        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
        """Send a generate request and return response."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def get_cached_tokens(self, response_json: Dict) -> int:
        """Extract cached tokens count from /generate response."""
        meta = response_json.get("meta_info", {})
        return int(meta.get("cached_tokens", 0))

    def flush_cache(self):
        """Flush device cache to force remote storage access."""
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

    def gen_prompt(self, token_num: int) -> str:
        """Generate a random prompt of specified token length."""
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def trigger_offloading_and_flush(self):
        """Helper method to trigger offloading and flush cache."""
        self.send_request(self.gen_prompt(1), max_tokens=150)
        self.flush_cache()

    def test_basic_backup_and_prefetch(self):
        """Test storage and retrieval of large context through remote cache."""
        print("\n=== Testing Large Context Cache Storage & Retrieval ===")

        base_prompt = self.gen_prompt(512)

        print("Step 1: Populating cache with large context...")
        response1 = self.send_request(base_prompt, max_tokens=150)
        self.assertIsNotNone(response1)

        self.trigger_offloading_and_flush()

        print("Step 2: Testing cache hit from remote storage...")
        start_time = time.time()
        response2 = self.send_request(base_prompt, max_tokens=150)
        retrieval_time = time.time() - start_time

        cached_tokens = self.get_cached_tokens(response2)
        print(
            f"Remote cache retrieval time: {retrieval_time:.3f}s, cached_tokens={cached_tokens}"
        )

        self.assertGreater(
            cached_tokens, 400, "Expected significant cached tokens for remote hit"
        )

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes."""
        print("\n=== Testing Eval Accuracy with Cache Persistence ===")

        print("Phase 1: Running initial GSM8K evaluation to populate cache...")
        args_initial = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.base_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=32,
        )
        metrics_initial = run_eval(args_initial)

        print("Phase 2: Flushing device cache...")
        self.flush_cache()

        print("Phase 3: Running second GSM8K evaluation using remote cache...")
        metrics_cached = run_eval(args_initial)

        accuracy_diff = abs(metrics_initial["score"] - metrics_cached["score"])
        print(f"Accuracy difference: {accuracy_diff:.4f}")
        print(f"Initial score: {metrics_initial['score']:.4f}")
        print(f"Cached score: {metrics_cached['score']:.4f}")

        self.assertGreater(
            metrics_initial["score"],
            self.initial_accuracy_threshold,
            "Initial accuracy should be reasonable",
        )
        self.assertGreater(
            metrics_cached["score"],
            self.initial_accuracy_threshold,
            "Cached accuracy should be reasonable",
        )
        self.assertLess(
            accuracy_diff,
            self.accuracy_threshold,
            "Accuracy should be consistent between cache states",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
