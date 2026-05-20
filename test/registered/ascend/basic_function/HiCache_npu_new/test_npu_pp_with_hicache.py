"""
Test Pipeline Parallelism with HiCache on Ascend NPU.

Tests HiCache with pipeline parallel (PP) configuration to verify
cache functionality works correctly across pipeline stages.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_pp_with_hicache.py -v
"""

import unittest
from types import SimpleNamespace

import requests

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

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNPUPPWithHiCache(CustomTestCase):
    """Test Pipeline Parallelism with HiCache on NPU.

    [Test Category] HiCache
    [Test Target] PP + HiCache integration
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                "0.6",
                "--hicache-ratio",
                "1.2",
                "--page-size",
                "64",
                "--enable-cache-report",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-storage-backend",
                "file",
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-io-backend",
                "kernel_ascend",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def flush_cache(self):
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes with PP."""
        print("\n=== Testing PP + HiCache Accuracy ===")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        print("Phase 1: Running initial GSM8K evaluation...")
        metrics_initial = run_eval_few_shot_gsm8k(args)
        print(f"Initial accuracy: {metrics_initial['accuracy']}")

        self.assertGreater(
            metrics_initial["accuracy"], 0.6, "Initial accuracy should be reasonable"
        )

        print("Phase 2: Flushing device cache...")
        self.flush_cache()

        print("Phase 3: Running second GSM8K evaluation...")
        metrics_cached = run_eval_few_shot_gsm8k(args)
        print(f"Cached accuracy: {metrics_cached['accuracy']}")

        accuracy_diff = abs(metrics_initial["accuracy"] - metrics_cached["accuracy"])
        print(f"Accuracy difference: {accuracy_diff:.4f}")

        self.assertGreater(
            metrics_cached["accuracy"], 0.6, "Cached accuracy should be reasonable"
        )
        self.assertLess(
            accuracy_diff,
            0.05,
            "Accuracy should be consistent between cache states",
        )


if __name__ == "__main__":
    unittest.main()
