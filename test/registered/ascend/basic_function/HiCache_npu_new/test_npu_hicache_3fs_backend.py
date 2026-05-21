"""
Benchmark tests for HiCache Storage with 3FS backend on Ascend NPU.

Tests HF3FS distributed storage backend with different memory layouts
and accuracy validation after cache flush.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_3fs_backend.py -v
"""

import json
import os
import unittest

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


class NPUHiCacheStorage3FSBackendBaseMixin:
    """Base mixin class with common setup and utilities for NPU 3FS backend tests."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.temp_dir = tempfile.mkdtemp()
        cls.model = cls._get_model_name()
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = cls._launch_server_with_hicache()
        cls._wait_for_server_ready(process=cls.process)

        print(f"Test server launched successfully at {cls.base_url}")

    @classmethod
    def tearDownClass(cls):
        import shutil

        if hasattr(cls, "process") and cls.process:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(cls.process.pid)

        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _get_model_name(cls):
        return QWEN3_8B_WEIGHTS_PATH

    @classmethod
    def _launch_server_with_hicache(cls):
        hf3fs_config = {
            "file_path_prefix": os.path.join(cls.temp_dir, "hicache"),
            "file_size": 1024 * 1024 * 1024 * 2,
            "numjobs": 2,
            "entries": 8,
            "use_mock_hf3fs_client": True,
            "hicache_storage_pass_prefix_keys": True,
        }

        config_file = os.path.join(cls.temp_dir, "hf3fs_config.json")
        with open(config_file, "w") as f:
            json.dump(hf3fs_config, f, indent=2)

        server_args, env_vars = cls._get_additional_server_args_and_env()

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
            "SGLANG_HICACHE_HF3FS_CONFIG_PATH": config_file,
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
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

    @classmethod
    def _get_additional_server_args_and_env(cls):
        hf3fs_config = {
            "file_path_prefix": os.path.join(cls.temp_dir, "hicache"),
            "file_size": 1024 * 1024 * 1024 * 2,
            "numjobs": 2,
            "entries": 8,
            "use_mock_hf3fs_client": True,
            "hicache_storage_pass_prefix_keys": True,
        }

        server_args = {
            "--tp-size": 1,
            "--hicache-ratio": 1.2,
            "--hicache-storage-backend": "hf3fs",
            "--hicache-io-backend": "kernel_ascend",
            "--attention-backend": "ascend",
            "--disable-cuda-graph": True,
            "--hicache-storage-backend-extra-config": json.dumps(hf3fs_config),
        }

        env_vars = {}

        return server_args, env_vars


class TestNPUHf3fsBackendPageFirstLayout(
    NPUHiCacheStorage3FSBackendBaseMixin, CustomTestCase
):
    """Page first layout tests for HiCache-HF3FS backend on NPU.

    [Test Category] HiCache
    [Test Target] HF3FS distributed storage with page_first_direct layout
    """

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "page_first_direct"
        return server_args, env_vars

    def test_basic_storage_and_retrieval(self):
        """Test storage and retrieval through HF3FS backend."""
        import requests

        print("\n=== Testing HF3FS Cache Storage & Retrieval ===")

        prompt = "What is the capital of France? " * 50

        print("Step 1: Populating cache...")
        response1 = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 50,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        self.assertEqual(response1.status_code, 200)

        print("Step 2: Flushing cache...")
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

        import time

        time.sleep(2)

        print("Step 3: Testing cache hit from HF3FS storage...")
        response2 = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 50,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        self.assertEqual(response2.status_code, 200)

        cached_tokens = int(
            response2.json().get("meta_info", {}).get("cached_tokens", 0)
        )
        print(f"Cached tokens: {cached_tokens}")

        self.assertGreater(
            cached_tokens, 100, "Expected significant cached tokens for HF3FS hit"
        )


class TestNPUHf3fsBackendAccuracy(NPUHiCacheStorage3FSBackendBaseMixin, CustomTestCase):
    """Accuracy tests for HiCache-HF3FS backend on NPU.

    [Test Category] HiCache
    [Test Target] GSM8K accuracy consistency after cache flush with HF3FS backend
    """

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-ratio"] = 1.5
        server_args["--hicache-mem-layout"] = "page_first_direct"
        return server_args, env_vars

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes."""
        from types import SimpleNamespace

        import requests

        print("\n=== Testing Eval Accuracy with HF3FS Cache ===")

        print("Phase 1: Running initial GSM8K evaluation...")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics_initial = run_eval_few_shot_gsm8k(args)
        print(f"Initial accuracy: {metrics_initial['accuracy']}")

        print("Phase 2: Flushing device cache...")
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

        print("Phase 3: Running second GSM8K evaluation...")
        metrics_cached = run_eval_few_shot_gsm8k(args)
        print(f"Cached accuracy: {metrics_cached['accuracy']}")

        accuracy_diff = abs(metrics_initial["accuracy"] - metrics_cached["accuracy"])
        print(f"Accuracy difference: {accuracy_diff:.4f}")

        self.assertGreater(
            metrics_initial["accuracy"], 0.6, "Initial accuracy should be reasonable"
        )
        self.assertGreater(
            metrics_cached["accuracy"], 0.6, "Cached accuracy should be reasonable"
        )
        self.assertLess(
            accuracy_diff,
            0.05,
            "Accuracy should be consistent between cache states",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
