"""
E2E tests for HiCache Storage functionality on Ascend NPU.

Tests file backend storage with different memory layouts, IO backends,
and accuracy validation after cache flush.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct / page_first_kv_split

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_storage_file_backend.py -v
"""

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
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class NPUHiCacheStorageBaseMixin:
    """Base mixin class with common setup and utilities for NPU HiCache storage tests."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = cls._get_model_name()
        cls.base_url = DEFAULT_URL_FOR_TEST

        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)

        cls.tokenizer = get_tokenizer(cls.model)

        cls.process = cls._launch_server_with_hicache()
        cls._wait_for_server_ready(process=cls.process)

        print(f"Test server launched successfully at {cls.base_url}")
        print(f"Cache directory: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

        import shutil

        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _get_model_name(cls):
        return QWEN3_8B_WEIGHTS_PATH

    @classmethod
    def _get_base_server_args(cls):
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
        }
        return {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.2,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "file",
            "--hicache-storage-backend-extra-config": json.dumps(extra_config),
            "--hicache-io-backend": "kernel_ascend",
            "--attention-backend": "ascend",
            "--disable-cuda-graph": True,
        }

    @classmethod
    def _get_additional_server_args_and_env(cls):
        return {}, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}

    @classmethod
    def _launch_server_with_hicache(cls):
        additional_server_args, env_vars = cls._get_additional_server_args_and_env()
        env_vars["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
        server_args = cls._get_base_server_args()
        if additional_server_args:
            server_args.update(additional_server_args)

        final_server_args = []
        for k, v in server_args.items():
            if isinstance(v, bool):
                final_server_args.append(str(k))
            else:
                final_server_args.append(str(k))
                final_server_args.append(str(v))

        print(f"final_server_args: {final_server_args}")

        env_vars = {
            **os.environ,
            **env_vars,
        }

        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=final_server_args,
            env=env_vars,
        )

    @classmethod
    def _wait_for_server_ready(cls, timeout: int = 60, process=None) -> bool:
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=timeout,
            process=process,
        )
        return True

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ) -> Dict:
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
        meta = response_json.get("meta_info", {})
        return int(meta.get("cached_tokens", 0))

    def flush_cache(self):
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def trigger_offloading_and_flush(self):
        self.send_request(self.gen_prompt(1), max_tokens=150)
        self.flush_cache()

    def test_basic_backup_and_prefetch(self):
        """Test storage and retrieval of large context through remote cache."""
        print("\n=== Testing Large Context Cache Storage & Retrieval ===")

        base_prompt = self.gen_prompt(768)

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
            cached_tokens, 700, "Expected significant cached tokens for remote hit"
        )


class TestNPUHiCacheStoragePageFirstLayout(NPUHiCacheStorageBaseMixin, CustomTestCase):
    """Page first layout tests for HiCache Storage on NPU.

    [Test Category] HiCache
    [Test Target] --hicache-mem-layout page_first_direct
    """

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args = {"--hicache-mem-layout": "page_first_direct"}
        return server_args, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}


class TestNPUHiCacheStoragePageFirstKVSplit(NPUHiCacheStorageBaseMixin, CustomTestCase):
    """Page first KV split layout tests for HiCache Storage on NPU.

    [Test Category] HiCache
    [Test Target] --hicache-mem-layout page_first_kv_split
    """

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args = {"--hicache-mem-layout": "page_first_kv_split"}
        return server_args, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}


class TestNPUHiCacheStorageAccuracy(NPUHiCacheStorageBaseMixin, CustomTestCase):
    """Accuracy tests for HiCache Storage on NPU.

    [Test Category] HiCache
    [Test Target] GSM8K accuracy consistency after cache flush
    """

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args = {
            "--tp-size": 1,
            "--hicache-ratio": 1.5,
        }
        return server_args, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes."""
        run_eval_accuracy_test(self)


def run_eval_accuracy_test(test_instance, accuracy_threshold: float = 0.05):
    """Generic eval accuracy test with configurable accuracy threshold.

    Args:
        test_instance: The test class instance that provides base_host, base_port, flush_cache, and assert methods
        accuracy_threshold: Maximum allowed accuracy difference between runs
    """
    print("\n=== Testing Eval Accuracy with Cache Persistence ===")

    print("Phase 1: Running initial GSM8K evaluation to populate cache...")
    args_initial = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
        host="http://127.0.0.1",
        port=int(test_instance.base_port),
    )
    metrics_initial = run_eval_few_shot_gsm8k(args_initial)

    print("Phase 2: Flushing device cache...")
    test_instance.flush_cache()

    print("Phase 3: Running second GSM8K evaluation using remote cache...")
    metrics_cached = run_eval_few_shot_gsm8k(args_initial)

    accuracy_diff = abs(metrics_initial["accuracy"] - metrics_cached["accuracy"])
    print(f"Accuracy difference: {accuracy_diff:.4f}")

    test_instance.assertGreater(
        metrics_initial["accuracy"], 0.6, "Initial accuracy should be reasonable"
    )
    test_instance.assertGreater(
        metrics_cached["accuracy"], 0.6, "Cached accuracy should be reasonable"
    )
    test_instance.assertLess(
        accuracy_diff,
        accuracy_threshold,
        "Accuracy should be consistent between cache states",
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
