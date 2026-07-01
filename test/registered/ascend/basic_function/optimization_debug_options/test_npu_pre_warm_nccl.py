import os
import time
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

register_npu_ci(est_time=400, suite="debug-full-2-npu-a3", nightly=True)


class TestPreWarmNccl(CustomTestCase):
    """Testcase: verify --pre-warm-nccl reduces first-request latency under TP=2

    [Test Category] Parameter
    [Test Target] --pre-warm-nccl
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
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

    @classmethod
    def _launch(cls, with_warmup):
        out_log = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log = open("./cache_err_log.txt", "w+", encoding="utf-8")
        args = list(cls.base_args)
        if with_warmup:
            args.append("--pre-warm-nccl")
        proc = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
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
        start = time.time()
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
        elapsed = time.time() - start
        return response, elapsed

    def test_pre_warm_nccl(self):
        # Launch without warmup
        proc1, out1, err1 = self._launch(with_warmup=False)
        resp1, lat1 = self._do_request()
        self.assertEqual(resp1.status_code, 200)
        self.assertIn("Paris", resp1.text)
        err1.seek(0)
        err_log1 = err1.read()
        self._cleanup(proc1, out1, err1)

        # Launch with warmup
        proc2, out2, err2 = self._launch(with_warmup=True)
        resp2, lat2 = self._do_request()
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("Paris", resp2.text)
        err2.seek(0)
        err_log2 = err2.read()
        self._cleanup(proc2, out2, err2)

        # Verify warmup executed
        self.assertIn(
            "nccl",
            err_log2.lower(),
            "Expected stderr to contain NCCL warmup log, "
            "proving --pre-warm-nccl was executed",
        )

        # Verify warmup reduces first-request latency
        self.assertLess(
            lat2,
            lat1,
            f"Expected warmup latency ({lat2:.2f}s) "
            f"< no-warmup latency ({lat1:.2f}s)",
        )


if __name__ == "__main__":
    unittest.main()
