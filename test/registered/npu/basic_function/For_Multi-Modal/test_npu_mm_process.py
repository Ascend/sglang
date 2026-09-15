import json
import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH,
    VIDEO_JOBS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=900, suite="full-4-npu-a3", nightly=True)

# Video processing config matching the reference command
_MM_PROCESS_CONFIG = json.dumps(
    {
        "video": {
            "min_pixels": 76800,
            "max_pixels": 921600,
            "resized_height": 448,
            "resized_width": 448,
            "fps": 2,
            "min_frames": 4,
            "max_frames": 64,
        }
    }
)

_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
    "--tp-size",
    "4",
    "--disable-cuda-graph",
]


class TestMmProcessConfigDpEncoder(CustomTestCase):
    """Verify video chat works with --mm-process-config (custom video
    preprocessing) and --mm-enable-dp-encoder enabled together.

    The server is launched twice with the same request:
    1. With both parameters enabled, verify video chat completion works
       (deterministic output) and record usage.prompt_tokens.
    2. Without the two parameters, record usage.prompt_tokens again.
    The video preprocessing config (448x448 resize, fps=2) compresses visual
    tokens, so prompt_tokens with the config must be smaller than without it.

    [Test Category] Parameter
    [Test Target] --mm-process-config, --mm-enable-dp-encoder
    """

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # prompt_tokens recorded in test_01 (with config), read in test_02
    prompt_tokens_with_config = None

    def _launch_server(self, other_args):
        """Launch server and store process/log files on the class."""
        self.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        self.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(self.out_file, self.err_file),
        )

    def _terminate_server(self):
        """Kill server process and clean up temp log files."""
        kill_process_tree(self.process.pid)
        self.out_file.close()
        self.err_file.close()
        os.unlink(self.out_file.name)
        os.unlink(self.err_file.name)

    def _send_video_request(self):
        """Send the same video chat request, return (prompt_tokens, content)."""
        data = {
            "model": "Qwen3-VL-30B-A3B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": VIDEO_JOBS_PATH,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What's happening in this video?",
                        },
                    ],
                }
            ],
            "stream": False,
            "temperature": 0.0,
            "max_new_tokens": 200,
        }
        resp = requests.post(
            self.base_url + "/v1/chat/completions",
            json=data,
            timeout=200,
        )
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text[:500]}")
        result = resp.json()
        self.assertIn("choices", result)
        self.assertGreater(len(result["choices"]), 0)
        content = result["choices"][0]["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)
        return result["usage"]["prompt_tokens"], content

    def test_01_video_chat_with_config(self):
        """Launch server with both params, verify response and record tokens."""
        try:
            self._launch_server(
                [
                    *_COMMON_ARGS,
                    "--mm-enable-dp-encoder",
                    "--mm-process-config",
                    _MM_PROCESS_CONFIG,
                ]
            )

            # Server must be healthy
            resp = requests.get(self.base_url + "/health", timeout=30)
            self.assertEqual(resp.status_code, 200)

            # Verify --mm-enable-dp-encoder took effect across TP ranks (server log)
            with open(self.err_file.name) as f:
                log_content = f.read()
            self.assertIn(
                "--mm-enable-dp-encoder is enabled across TP=4",
                log_content,
                "Expected '--mm-enable-dp-encoder is enabled across TP=4' not found in server log",
            )

            prompt_tokens, content = self._send_video_request()
            print(f"\n[with config] prompt_tokens: {prompt_tokens}")
            print(f"Video chat response: {content[:200]}...")

            # Deterministic output at temperature=0 with the config enabled
            self.assertIn(
                "In this video, a man is standing on a stage in front of a large screen",
                content,
            )
            type(self).prompt_tokens_with_config = prompt_tokens
        finally:
            self._terminate_server()

    def test_02_prompt_tokens_reduced_without_config_comparison(self):
        """Launch server without the two params, record tokens and compare.

        The 448x448 resize + fps=2 sampling from --mm-process-config compresses
        visual tokens, so prompt_tokens with the config must be smaller.
        """
        self.assertIsNotNone(
            self.prompt_tokens_with_config,
            "test_01 must run first to record prompt_tokens with config",
        )
        try:
            self._launch_server([*_COMMON_ARGS])

            # Server must be healthy
            resp = requests.get(self.base_url + "/health", timeout=30)
            self.assertEqual(resp.status_code, 200)

            prompt_tokens, content = self._send_video_request()
            print(f"\n[without config] prompt_tokens: {prompt_tokens}")
            print(f"Video chat response: {content[:200]}...")

            self.assertLess(
                self.prompt_tokens_with_config,
                prompt_tokens,
                f"prompt_tokens with config ({self.prompt_tokens_with_config}) "
                f"should be smaller than without config ({prompt_tokens})",
            )
        finally:
            self._terminate_server()


if __name__ == "__main__":
    unittest.main()
