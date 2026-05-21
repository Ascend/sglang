"""
Test Large Model with HiCache on 4-NPU Ascend cluster.

Tests HiCache with large model (Qwen3-32B) on 4 NPUs to verify
cache functionality and accuracy consistency at scale.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_large_model.py -v
"""

import shutil
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

ACC_THRESHOLDS = {"gsm8k": 0.8}


class TestNPUHiCacheLargeModel(CustomTestCase):
    """Test Large Model with HiCache on 4-NPU Ascend cluster.

    [Test Category] HiCache
    [Test Target] Large model + HiCache accuracy at scale
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="npu-hicache-large-")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--max-total-tokens",
                "120000",
                "--chunked-prefill-size",
                "2048",
                "--max-running-requests",
                "128",
                "--hicache-mem-layout",
                "page_first_direct",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-size",
                "0",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-io-backend",
                "kernel_ascend",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
            env={
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        return run_eval_few_shot_gsm8k(args)

    def test_gsm8k(self):
        """Test GSM8K accuracy consistency with HiCache on large model."""
        print("\n=== Testing Large Model HiCache Accuracy ===")

        first_metrics = self._run_gsm8k()
        print(f"first_metrics={first_metrics}")
        self.assertGreaterEqual(
            first_metrics["accuracy"],
            ACC_THRESHOLDS["gsm8k"],
            "Initial accuracy should meet threshold",
        )

        print("Flushing cache...")
        res = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        res.raise_for_status()

        second_metrics = self._run_gsm8k()
        print(f"second_metrics={second_metrics}")
        self.assertGreaterEqual(
            second_metrics["accuracy"],
            ACC_THRESHOLDS["gsm8k"],
            "Cached accuracy should meet threshold",
        )
        self.assertLessEqual(
            abs(second_metrics["accuracy"] - first_metrics["accuracy"]),
            0.05,
            f"HiCache prefetch accuracy drift too large: "
            f"first={first_metrics['accuracy']}, second={second_metrics['accuracy']}",
        )


if __name__ == "__main__":
    unittest.main()
