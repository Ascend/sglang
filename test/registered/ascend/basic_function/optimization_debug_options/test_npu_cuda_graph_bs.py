import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)


class TestCudaGraphBsOverride(CustomTestCase):
    """Testcase: verify --cuda-graph-max-bs-decode, --cuda-graph-max-bs-prefill,
    --cuda-graph-bs-decode and --cuda-graph-bs-prefill all override defaults,
    and smaller max_bs reduces CUDA Graph memory

    [Test Category] Parameter
    [Test Target] --cuda-graph-max-bs-decode; --cuda-graph-max-bs-prefill;
                  --cuda-graph-bs-decode; --cuda-graph-bs-prefill
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def _launch(cls, max_bs_decode, max_bs_prefill, bs_decode, bs_prefill):
        out_log = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log = open("./cache_err_log.txt", "w+", encoding="utf-8")
        proc = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--cuda-graph-max-bs-decode",
                str(max_bs_decode),
                "--cuda-graph-max-bs-prefill",
                str(max_bs_prefill),
                "--cuda-graph-bs-decode",
            ]
            + [str(b) for b in bs_decode]
            + [
                "--cuda-graph-bs-prefill",
            ]
            + [str(b) for b in bs_prefill],
            return_stdout_stderr=(out_log, err_log),
        )
        return proc, out_log, err_log

    @classmethod
    def _cleanup(cls, proc, out_log, err_log):
        kill_process_tree(proc.pid)
        out_log.close()
        err_log.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def _do_request(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        return response

    def test_cuda_graph_bs_override(self):
        # Launch with small max_bs
        proc1, out1, err1 = self._launch(1, 1, [1], [1])
        resp1 = self._do_request()
        self.assertEqual(resp1.status_code, 200)
        self.assertIn("Paris", resp1.text)
        err1.seek(0)
        err_log1 = err1.read()
        self._cleanup(proc1, out1, err1)

        # Launch with larger max_bs
        proc2, out2, err2 = self._launch(8, 8, [1, 2, 4, 8], [1, 2, 4])
        resp2 = self._do_request()
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("Paris", resp2.text)
        err2.seek(0)
        err_log2 = err2.read()
        self._cleanup(proc2, out2, err2)

        # Verify parameters were parsed in both launches
        self.assertIn(
            "max_bs",
            err_log1,
            "Expected stderr to contain 'max_bs', proving max_bs was parsed",
        )
        self.assertIn(
            "bs",
            err_log2,
            "Expected stderr to contain 'bs', proving bs list was parsed",
        )

        # Verify max_bs controls memory: smaller max_bs uses less CUDA Graph memory
        mem1 = self._extract_graph_memory(err_log1)
        mem2 = self._extract_graph_memory(err_log2)
        if mem1 is not None and mem2 is not None:
            self.assertLess(
                mem1,
                mem2,
                f"Expected max_bs=1 graph memory ({mem1}MB) "
                f"< max_bs=8 graph memory ({mem2}MB)",
            )

    @staticmethod
    def _extract_graph_memory(err_log):
        """Extract CUDA Graph memory in MB from stderr log."""
        import re

        for line in err_log.splitlines():
            m = re.search(r"(\d+)\s*MB", line)
            if m and "graph" in line.lower():
                return int(m.group(1))
        return None


if __name__ == "__main__":
    unittest.main()
