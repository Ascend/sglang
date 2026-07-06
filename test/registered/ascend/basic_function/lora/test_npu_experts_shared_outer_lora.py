import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

LORA_HF_REPO = "/root/.cache/huggingface/hub/lora-diff-Qwen3-30B-A3B-Instruct-2507"
register_npu_ci(est_time=200, suite="full-2-npu-a3", nightly=True)


class TestDeepSeekV32(CustomTestCase):
    """Testcase: Verify that the inference accuracy of the vllm-ascend/DeepSeek-V3.2-W8A8 model on the GSM8K dataset is no less than 0.95.

    [Test Category] Model
    [Test Target] vllm-ascend/DeepSeek-V3.2-W8A8
    """

    lora_a = LORA_HF_REPO

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
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
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                0.7,
                "--expert-distribution-recorder-mode",
                "stat",
                "--max-running-requests",
                32,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--cuda-graph-max-bs",
                32,
                "--tp-size",
                2,
                "--lora-path",
                f"lora_a={cls.lora_a}",
                "--experts-shared-outer-loras",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
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

        self.assertGreater(metrics["score"], 0.90)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        error_message = "Shared outer LoRA mode enabled"
        self.assertIn(error_message, content)


if __name__ == "__main__":
    unittest.main()
