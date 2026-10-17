import os
import unittest

import psutil

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


class TestEngineChildPids(CustomTestCase):
    """
    Verifies that launching an Engine exposes the PIDs of all child processes
    (schedulers, detokenizer) and that those PIDs correspond to live processes.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=QWEN3_0_6B_WEIGHTS_PATH,
            random_seed=42,
            attention_backend="ascend",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_get_all_child_pids_returns_live_pids(self):
        pids = self.engine.get_all_child_pids()
        self.assertIsInstance(pids, list)
        self.assertGreater(len(pids), 0, "Expected at least one child PID")
        for pid in pids:
            self.assertIsInstance(pid, int)
            self.assertTrue(
                psutil.pid_exists(pid),
                f"PID {pid} does not correspond to a running process",
            )
        current_proc = psutil.Process(os.getpid())
        child_pids = {c.pid for c in current_proc.children(recursive=True)}
        for pid in pids:
            self.assertIn(
                pid,
                child_pids,
                f"PID {pid} is not a child of the current process",
            )

    def test_child_pids_include_scheduler_and_detokenizer(self):
        pids = self.engine.get_all_child_pids()
        # dp_size=1 gives one scheduler + one detokenizer = at least 2 PIDs
        self.assertGreaterEqual(
            len(pids),
            2,
            "Expected at least 2 child PIDs (scheduler + detokenizer)",
        )

    def test_child_pids_no_duplicates(self):
        pids = self.engine.get_all_child_pids()
        self.assertEqual(
            len(pids),
            len(set(pids)),
            f"Duplicate PIDs found: {pids}",
        )


if __name__ == "__main__":
    unittest.main()
