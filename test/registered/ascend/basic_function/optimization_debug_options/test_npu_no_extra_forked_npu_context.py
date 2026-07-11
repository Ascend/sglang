import subprocess
import time
import unittest

import psutil

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="debug-full-2-npu-a3", nightly=True)


class TestTPServerNPUProcesses(CustomTestCase):
    """Testcase: Verify TP server does not create extra NPU processes beyond TP workers.

    [Test Category] Parameter
    [Test Target] --cuda-graph-backend-decode; --cuda-graph-backend-prefill
    """

    tp_size = 2

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                str(cls.tp_size),
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.70",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_tp_server_has_only_worker_npu_processes(self):
        rows = self._wait_for_server_npu_processes()
        # On NPU the launcher parent may acquire a lightweight context
        # (e.g. HCCL init) — filter it out, only count TP worker processes.
        worker_pids = {
            row["pid"] for row in rows if row["pid"] != self.process.pid
        }

        self.assertEqual(
            len(worker_pids),
            self.tp_size,
            f"TP={self.tp_size} server should have exactly {self.tp_size} "
            f"NPU worker processes, got {len(worker_pids)}: {self._format_rows(rows)}",
        )

    def _wait_for_server_npu_processes(self):
        deadline = time.monotonic() + 60
        stable_since = None
        last_rows = []

        while time.monotonic() < deadline:
            tree_pids = self._server_process_tree_pids()
            rows = [
                row
                for row in self._query_npu_processes()
                if row["pid"] in tree_pids
            ]
            last_rows = rows

            if len({row["pid"] for row in rows}) >= self.tp_size:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 3:
                    return rows
            else:
                stable_since = None

            time.sleep(0.5)

        self.fail(
            f"Timed out waiting for TP={self.tp_size} NPU worker processes. "
            f"Last observed rows: {self._format_rows(last_rows)}"
        )

    def _server_process_tree_pids(self):
        pids = {self.process.pid}
        try:
            parent = psutil.Process(self.process.pid)
            pids.update(child.pid for child in parent.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
        return pids

    def _query_npu_processes(self):
        result = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            capture_output=True,
            text=True,
        )

        rows = []
        for line in result.stdout.splitlines():
            # Only parse data rows: pipe-delimited lines that contain the
            # memory usage pattern "used / total" (digits / digits).
            if "/" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            # Process ID column: look for entries that are purely numeric
            # and could be a PID (length 3–7 digits).
            for part in parts:
                if part.isdigit() and 3 <= len(part) <= 7:
                    rows.append({"pid": int(part)})
                    break
        return rows

    def _format_rows(self, rows):
        if not rows:
            return "[]"
        return "[" + ", ".join(str(row) for row in rows) + "]"


if __name__ == "__main__":
    unittest.main()
