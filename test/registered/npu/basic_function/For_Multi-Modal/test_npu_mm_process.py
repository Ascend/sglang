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

    @classmethod
    def setUpClass(cls):
        # Launch the first server WITH both params
        cls._launch_server(
            [
                *_COMMON_ARGS,
                "--mm-enable-dp-encoder",
                "--mm-process-config",
                _MM_PROCESS_CONFIG,
            ]
        )

    @classmethod
    def tearDownClass(cls):
        # Kill whichever server is currently running
        cls._terminate_server()

    @classmethod
    def _launch_server(cls, other_args):
        """Launch server and store process/log files on the class."""
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def _terminate_server(cls):
        """Kill server process and clean up temp log files."""
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)
            cls.process = None
        for f in ("out_file", "err_file"):
            handle = getattr(cls, f, None)
            if handle is not None:
                handle.close()
                os.unlink(handle.name)
                setattr(cls, f, None)

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

    def test_video_chat_and_token_comparison(self):
        """Phase 1: with config, verify response and record tokens.
        Phase 2: without config, record tokens and compare."""

        # ---- Phase 1: with both params (server from setUpClass) ----
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

        prompt_tokens_with_config, content = self._send_video_request()
        print(f"\n[with config] prompt_tokens: {prompt_tokens_with_config}")
        print(f"Video chat response: {content[:200]}...")

        # Deterministic output at temperature=0 with the config enabled
        self.assertIn(
            "In this video, a man is standing on a stage in front of a large screen",
            content,
        )

        # ---- Switch to the server WITHOUT the two params ----
        type(self)._terminate_server()
        type(self)._launch_server([*_COMMON_ARGS])

        # ---- Phase 2: without config ----
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        prompt_tokens_without_config, content = self._send_video_request()
        print(f"\n[without config] prompt_tokens: {prompt_tokens_without_config}")
        print(f"Video chat response: {content[:200]}...")

        # 448x448 resize + fps=2 sampling compresses visual tokens,
        # so prompt_tokens with the config must be smaller
        self.assertLess(
            prompt_tokens_with_config,
            prompt_tokens_without_config,
            f"prompt_tokens with config ({prompt_tokens_with_config}) "
            f"should be smaller than without config ({prompt_tokens_without_config})",
        )


if __name__ == "__main__":
    unittest.main()
