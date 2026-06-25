import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="full-1-npu-a3", nightly=True)


class TestMatchedStop(CustomTestCase, MatchedStopMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=["--max-running-requests", "10"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_finish_stop_eos(self):
        # Qwen3 EOS token ID is 151645
        eos_token_ids = [151645]
        # Use Qwen3 chat format for completions endpoint
        qwen3_prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
        self._run_completions_generation(
            prompt=qwen3_prompt,
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_ids,
        )
        self._run_chat_completions_generation(
            prompt="What is 2 + 2?",
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_ids,
        )


if __name__ == "__main__":
    unittest.main()
