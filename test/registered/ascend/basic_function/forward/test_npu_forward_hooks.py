import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-4-npu-a3",
    nightly=True,
)


class TestSetForwardHooks(CustomTestCase):
    """Testcase: Verify set --forward-hooks parameter, can identify the set hook function
    and the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --forward-hooks
    """

    hooks_spec = [
        {
            "name": "qwen_first_layer_attn_monitor",
            "target_modules": ["model.layers.0.self_attn"],
            "hook_factory": "sglang.test.ascend.test_ascend_utils:create_attention_monitor_hook_factory",
            "config": {"layer_index": 0},
        }
    ]
    forward_hooks = json.dumps(hooks_spec)

    @classmethod
    def setUpClass(cls):
        cls.out_log_file_name = "./tmp_out_log.txt"
        cls.hook_log_file_name = "./tmp_hook_log.txt"
        cls.out_log_file = open(cls.out_log_file_name, "w+", encoding="utf-8")
        cls.hook_log_file = open(cls.hook_log_file_name, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            QWEN3_32B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tp-size",
                "4",
                "--forward-hooks",
                cls.forward_hooks,
            ],
            return_stdout_stderr=(cls.out_log_file, cls.hook_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.hook_log_file.close()
        os.remove(cls.out_log_file_name)
        os.remove(cls.hook_log_file_name)

    def test_forward_hooks(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        self.hook_log_file.seek(0)
        hook_content = self.hook_log_file.read()
        self.assertIn("hook effect", hook_content)


if __name__ == "__main__":
    unittest.main()
