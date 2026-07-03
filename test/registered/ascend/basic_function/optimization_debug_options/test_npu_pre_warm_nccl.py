import os
import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="debug-full-2-npu-a3", nightly=True)

# Expected log line from model_runner.py:1147-1150 —
# "NCCL/RCCL warmup completed in {X.XXX}s (tp_size=2, pp_size=1, ep_size=1)"
_WARMUP_LOG_RE = re.compile(
    r"NCCL/RCCL warmup completed in ([\d.]+)s "
    r"\(tp_size=\d+, pp_size=\d+, ep_size=\d+\)"
)


class TestPreWarmNccl(CustomTestCase):
    """Testcase: verify --pre-warm-nccl executes all-reduce warmup during
    server startup and measures non-zero time.

    The warmup runs dist.all_reduce + synchronize (model_runner.py:1135-1150).
    Parsing warmup_elapsed from the log proves the all-reduce actually executed.

    NOTE: server_args.py:3065 currently sets pre_warm_nccl=False on non-CUDA/HIP
    hardware (including NPU).  The HCCL backend path needs to be added there
    before this test can pass on NPU.

    [Test Category] Parameter
    [Test Target] --pre-warm-nccl
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "2",
    ]

    def _launch(self, with_warmup):
        out_log = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log = open("./cache_err_log.txt", "w+", encoding="utf-8")
        args = list(self.base_args)
        if with_warmup:
            args.append("--pre-warm-nccl")
        proc = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            return_stdout_stderr=(out_log, err_log),
        )
        return proc, out_log, err_log

    def _cleanup(self, proc, out_log, err_log):
        kill_process_tree(proc.pid)
        out_log.close()
        err_log.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def _parse_warmup_elapsed(self, err_log):
        """Return warmup elapsed seconds (float), or None if not found."""
        err_log.seek(0)
        for line in err_log:
            m = _WARMUP_LOG_RE.search(line)
            if m:
                return float(m.group(1))
        return None

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

    def test_pre_warm_nccl(self):
        # ---- With --pre-warm-nccl ----
        proc1, out1, err1 = self._launch(with_warmup=True)
        resp1 = self._do_request()
        self.assertEqual(resp1.status_code, 200)
        self.assertIn("Paris", resp1.text)
        warmup_elapsed = self._parse_warmup_elapsed(err1)
        self._cleanup(proc1, out1, err1)

        self.assertIsNotNone(
            warmup_elapsed,
            "Expected stderr to contain 'NCCL/RCCL warmup completed in {X}s', "
            "proving --pre-warm-nccl triggered the warmup code path "
            "(model_runner.py:1135-1150).",
        )
        self.assertGreater(
            warmup_elapsed,
            0,
            f"Expected warmup elapsed time > 0, got {warmup_elapsed:.6f}s. "
            "A zero value would mean all-reduce was skipped.",
        )

        # ---- Without --pre-warm-nccl ----
        proc2, out2, err2 = self._launch(with_warmup=False)
        resp2 = self._do_request()
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("Paris", resp2.text)
        no_warmup = self._parse_warmup_elapsed(err2)
        self._cleanup(proc2, out2, err2)

        self.assertIsNone(
            no_warmup,
            "Expected stderr NOT to contain 'NCCL/RCCL warmup completed', "
            "proving the warmup is only triggered when the flag is set.",
        )


if __name__ == "__main__":
    unittest.main()
