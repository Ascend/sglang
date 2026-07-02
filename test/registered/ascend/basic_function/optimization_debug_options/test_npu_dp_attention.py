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

register_npu_ci(est_time=400, suite="debug-full-2-npu-a3", nightly=True)


class TestDpAttentionLocalControlBroadcast(CustomTestCase):
    """Testcase: verify --enable-dp-attention-local-control-broadcast
    enables local broadcast under DP=2

    [Test Category] Parameter
    [Test Target] --enable-dp-attention-local-control-broadcast
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
    def _launch(cls, enable_broadcast):
        out_log = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log = open("./cache_err_log.txt", "w+", encoding="utf-8")
        args = list(cls.base_args)
        if enable_broadcast:
            args.append("--enable-dp-attention-local-control-broadcast")
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

    def test_dp_attention_local_control_broadcast(self):
        # Launch without the flag (default False)
        proc1, out1, err1 = self._launch(enable_broadcast=False)
        resp1 = self._do_request()
        self.assertEqual(resp1.status_code, 200)
        self.assertIn("Paris", resp1.text)
        err1.seek(0)
        err_log1 = err1.read()
        self._cleanup(proc1, out1, err1)

        # Launch with the flag (True)
        proc2, out2, err2 = self._launch(enable_broadcast=True)
        resp2 = self._do_request()
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("Paris", resp2.text)
        err2.seek(0)
        err_log2 = err2.read()
        self._cleanup(proc2, out2, err2)

        # Verify local broadcast was enabled
        self.assertIn(
            "broadcast",
            err_log2.lower(),
            "Expected stderr to contain 'broadcast', proving "
            "--enable-dp-attention-local-control-broadcast took effect",
        )


if __name__ == "__main__":
    unittest.main()
