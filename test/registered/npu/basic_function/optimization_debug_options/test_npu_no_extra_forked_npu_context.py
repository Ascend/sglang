import subprocess
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-2-npu-a3", nightly=True)


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

        # Record existing NPU processes before launching this test server.
        cls.baseline_npu_pids = {
            row["pid"] for row in cls._query_npu_processes_static()
        }

        print(
            "baseline npu pids:",
            cls.baseline_npu_pids,
            flush=True,
        )

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
        new_npu_pids = {row["pid"] for row in rows}

        # Normally TP=2 should create two NPU worker processes.
        #
        # On some NPU environments the launcher may also acquire a lightweight
        # NPU context, so allow at most one additional NPU process.
        self.assertIn(
            len(new_npu_pids),
            (self.tp_size, self.tp_size + 1),
            f"TP={self.tp_size} server should create {self.tp_size} worker "
            f"NPU processes and at most one launcher NPU context, "
            f"got {len(new_npu_pids)}: {self._format_rows(rows)}",
        )

    def _wait_for_server_npu_processes(self):
        deadline = time.monotonic() + 60
        stable_since = None
        last_rows = []

        while time.monotonic() < deadline:
            raw_rows = self._query_npu_processes_static()

            # npu-smi returns host-side PIDs. Instead of comparing them with
            # container PIDs, only keep NPU processes that appeared after this
            # test server was launched.
            rows = [
                row
                for row in raw_rows
                if row["pid"] not in self.baseline_npu_pids
            ]

            last_rows = rows

            print(
                "raw npu-smi rows:",
                raw_rows,
                flush=True,
            )
            print(
                "baseline npu pids:",
                self.baseline_npu_pids,
                flush=True,
            )
            print(
                "new npu rows:",
                rows,
                flush=True,
            )

            num_new_pids = len({row["pid"] for row in rows})

            if num_new_pids >= self.tp_size:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 3:
                    return rows
            else:
                stable_since = None

            time.sleep(0.5)

        self.fail(
            f"Timed out waiting for TP={self.tp_size} NPU processes. "
            f"Last observed rows: {self._format_rows(last_rows)}"
        )

    @staticmethod
    def _query_npu_processes_static():
        result = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            capture_output=True,
            text=True,
        )

        rows = []
        pid_col = None  # column index of "Process id" header

        for line in result.stdout.splitlines():
            if "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]

            # Locate the process table by its "Process id" column header.
            if pid_col is None:
                for i, part in enumerate(parts):
                    if part == "Process id":
                        pid_col = i
                        break
                continue

            # Reached the next section header ("NPU" row) -> stop.
            if len(parts) > 1 and parts[1].startswith("NPU"):
                break

            # Separator line or empty data row -> skip.
            if "+" in line or not any(parts):
                continue

            # Malformed row -> skip.
            if pid_col >= len(parts):
                continue

            # No data in PID column -> skip.
            if not parts[pid_col].isdigit():
                continue

            pid = int(parts[pid_col])
            rows.append({"pid": pid})

        return rows

    def _format_rows(self, rows):
        if not rows:
            return "[]"

        return "[" + ", ".join(str(row) for row in rows) + "]"


if __name__ == "__main__":
    unittest.main()
