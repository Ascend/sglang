import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestTokenizerBatchEncode(CustomTestCase):
    """Testcase: Verify throughput improvement when enabling --enable-tokenizer-batch-encode.

    [Test Category] Parameter
    [Test Target] --enable-tokenizer-batch-encode
    """

    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def _start_server(self, enable_tokenizer_batch_encode: bool):
        other_args = [
            "--attention-backend",
            "ascend",
        ]
        if enable_tokenizer_batch_encode:
            other_args.append("--enable-tokenizer-batch-encode")

        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def _get_throughput(self) -> float:
        self.run_decode(16)
        max_tokens = 256
        tic = time.perf_counter()
        self.run_decode(max_tokens)
        tok = time.perf_counter()
        return max_tokens / (tok - tic)

    def test_tokenizer_batch_encode_throughput_improvement(self):
        # Without tokenizer batch encode
        proc_off = self._start_server(enable_tokenizer_batch_encode=False)
        tp_off = self._get_throughput()
        kill_process_tree(proc_off.pid)

        # With tokenizer batch encode
        proc_on = self._start_server(enable_tokenizer_batch_encode=True)
        tp_on = self._get_throughput()
        kill_process_tree(proc_on.pid)

        self.assertGreater(tp_on, tp_off)

    def test_mmlu(self):
        process = self._start_server(enable_tokenizer_batch_encode=True)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.50)
        kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
