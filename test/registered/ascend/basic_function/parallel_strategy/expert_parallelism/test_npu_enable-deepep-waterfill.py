import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestDeepSeekV32(CustomTestCase):
    """Testcase: Verify that the inference accuracy of the vllm-ascend/DeepSeek-V3.2-W8A8 model on the GSM8K dataset is no less than 0.95.

    [Test Category] Model
    [Test Target] vllm-ascend/DeepSeek-V3.2-W8A8
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.out_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        )
        cls.err_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="err.log"
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=6000,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--enable-deepep-waterfill",
                "--enforce-shared-experts-fusion",
                "--mem-fraction-static",
                0.82,
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--context-length",
                40960,
                "--max-prefill-tokens",
                40960,
                "--max-total-tokens",
                40960,
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
            env={
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
                "HCCL_BUFFSIZE": "1600",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "SGLANG_NPU_USE_MLAPO": "0",
                "SGLANG_NPU_USE_MULTI_STREAM": "1",
                "TASK_QUEUE_ENABLE": "0",
                "TRANSFORMERS_VERBOSITY": "error",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=128,
            num_threads=200,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.95)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn("DeepEP Waterfill is enabled", content)


if __name__ == "__main__":
    unittest.main()
