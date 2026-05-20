"""
Benchmark tests for HiCache Storage with Mooncake backend on Ascend NPU.

Tests Mooncake distributed storage backend with different memory layouts
and accuracy validation after cache flush.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct
- MOONCAKE_DEVICE may need NPU-specific device name

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_mooncake_backend.py -v
"""

import os
import subprocess
import time
import unittest

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
    find_available_port,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class NPUHiCacheStorageMooncakeBackendBaseMixin:
    """Base mixin class with common setup and utilities for NPU Mooncake backend tests."""

    mooncake_master_port_base = 50051
    mooncake_metadata_port_base = 8080

    @classmethod
    def setUpClass(cls):
        cls.mooncake_master_port = find_available_port(
            NPUHiCacheStorageMooncakeBackendBaseMixin.mooncake_master_port_base
        )
        cls.mooncake_metadata_port = find_available_port(
            NPUHiCacheStorageMooncakeBackendBaseMixin.mooncake_metadata_port_base
        )

        cls._start_mooncake_services()

        cls.temp_dir = None
        cls.model = cls._get_model_name()
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.tokenizer = get_tokenizer(cls.model)

        cls.process = cls._launch_server_with_hicache()
        cls._wait_for_server_ready(process=cls.process)

        print(f"Test server launched successfully at {cls.base_url}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

        cls._stop_mooncake_services()

    @classmethod
    def _get_model_name(cls):
        return QWEN3_8B_WEIGHTS_PATH

    @classmethod
    def _start_mooncake_services(cls):
        print("Starting Mooncake services...")
        print(
            f"Using master port: {cls.mooncake_master_port}, metadata port: {cls.mooncake_metadata_port}"
        )

        try:
            cls.metadata_service_process = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "mooncake.http_metadata_server",
                    "--port",
                    str(cls.mooncake_metadata_port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            print(
                f"Mooncake metadata service started on port {cls.mooncake_metadata_port}"
            )
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Warning: Could not start Mooncake metadata service: {e}")
            cls.metadata_service_process = None

        try:
            cls.master_service_process = subprocess.Popen(
                ["mooncake_master", "--port", str(cls.mooncake_master_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            print(f"Mooncake master service started on port {cls.mooncake_master_port}")
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Warning: Could not start Mooncake master service: {e}")
            cls.master_service_process = None

        cls._wait_for_mooncake_services_ready()

    @classmethod
    def _wait_for_mooncake_services_ready(cls, timeout: int = 30) -> bool:
        print("Waiting for Mooncake services to be ready...")

        start_time = time.time()
        services_ready = False

        while time.time() - start_time < timeout:
            try:
                metadata_ready = False
                if (
                    cls.metadata_service_process
                    and cls.metadata_service_process.poll() is None
                ):
                    try:
                        metadata_url = (
                            f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata"
                        )
                        response = requests.get(metadata_url, timeout=2)
                        if response.status_code == 200:
                            metadata_ready = True
                            print("Mooncake metadata service is ready")
                    except (requests.RequestException, ConnectionError):
                        pass

                master_ready = False
                if (
                    cls.master_service_process
                    and cls.master_service_process.poll() is None
                ):
                    if time.time() - start_time > 5:
                        master_ready = True
                        print("Mooncake master service is ready")

                if metadata_ready and master_ready:
                    services_ready = True
                    print("All Mooncake services are ready")
                    break

            except Exception as e:
                print(f"Error checking service readiness: {e}")

            time.sleep(2)

        if not services_ready:
            print(
                "Warning: Mooncake services may not be fully ready, continuing anyway..."
            )

        return services_ready

    @classmethod
    def _stop_mooncake_services(cls):
        print("Stopping Mooncake services...")

        if hasattr(cls, "metadata_service_process") and cls.metadata_service_process:
            try:
                os.killpg(os.getpgid(cls.metadata_service_process.pid), 9)
                cls.metadata_service_process.wait(timeout=5)
                print("Mooncake metadata service stopped")
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
                print(f"Warning: Could not stop Mooncake metadata service: {e}")

        if hasattr(cls, "master_service_process") and cls.master_service_process:
            try:
                os.killpg(os.getpgid(cls.master_service_process.pid), 9)
                cls.master_service_process.wait(timeout=5)
                print("Mooncake master service stopped")
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as e:
                print(f"Warning: Could not stop Mooncake master service: {e}")

    @classmethod
    def _launch_server_with_hicache(cls):
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
        server_args = {
            "--tp-size": 2,
            "--hicache-ratio": 2,
            "--hicache-storage-backend": "mooncake",
            "--hicache-io-backend": "kernel_ascend",
            "--attention-backend": "ascend",
            "--disable-cuda-graph": True,
        }

        env_vars = {
            "MOONCAKE_MASTER": f"127.0.0.1:{cls.mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",
        }

        return server_args, env_vars

    def send_request(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.0
    ):
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

    def flush_cache(self):
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()


class TestNPUMooncakeBackendPageFirstLayout(
    NPUHiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """Page first layout tests for HiCache-Mooncake backend on NPU.

    [Test Category] HiCache
    [Test Target] Mooncake distributed storage with page_first_direct layout
    """

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "page_first_direct"
        return server_args, env_vars

    def test_basic_backup_and_prefetch(self):
        """Test storage and retrieval through Mooncake backend."""
        print("\n=== Testing Mooncake Cache Storage & Retrieval ===")

        prompt = "What is the capital of France? " * 50

        print("Step 1: Populating cache...")
        response1 = self.send_request(prompt, max_tokens=50)
        self.assertIsNotNone(response1)

        print("Step 2: Flushing cache...")
        self.flush_cache()
        time.sleep(2)

        print("Step 3: Testing cache hit from Mooncake storage...")
        response2 = self.send_request(prompt, max_tokens=50)
        cached_tokens = int(response2.get("meta_info", {}).get("cached_tokens", 0))
        print(f"Cached tokens: {cached_tokens}")

        self.assertGreater(
            cached_tokens, 100, "Expected significant cached tokens for Mooncake hit"
        )


class TestNPUMooncakeBackendAccuracy(
    NPUHiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """Accuracy tests for HiCache-Mooncake backend on NPU.

    [Test Category] HiCache
    [Test Target] GSM8K accuracy consistency after cache flush with Mooncake backend
    """

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-ratio"] = 1.5
        server_args["--hicache-mem-layout"] = "page_first_direct"
        return server_args, env_vars

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes."""
        print("\n=== Testing Eval Accuracy with Mooncake Cache ===")

        from types import SimpleNamespace

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
        self.flush_cache()

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
