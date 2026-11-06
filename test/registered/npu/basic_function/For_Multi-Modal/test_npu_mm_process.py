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

# Phase-1 result shared across the two test classes below (they run in one
# pytest process, in file definition order).
_prompt_tokens_with_config = None


class _TestMmProcessBase(CustomTestCase):
    """Shared server lifecycle for the mm-process-config comparison tests.

    Each concrete class launches its own server in setUpClass and kills it in
    tearDownClass. Servers must NOT be killed/restarted mid-test: the CI
    harness's fail-fast watcher aborts the whole pytest process when a watched
    server exits outside of class teardown.
    """

    __test__ = False
    mm_process_config = None
    """JSON string passed via --mm-process-config; None = launch without it."""

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        extra_args = []
        if cls.mm_process_config is not None:
            extra_args = [
                "--mm-enable-dp-encoder",
                "--mm-process-config",
                cls.mm_process_config,
            ]
        cls._launch_server(extra_args)

    @classmethod
    def tearDownClass(cls):
        cls._terminate_server()

    @classmethod
    def _launch_server(cls, extra_args):
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
            other_args=[*_COMMON_ARGS, *extra_args],
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
                        {"type": "text", "text": "What's happening in this video?"},
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


class TestMmProcessWithConfig(_TestMmProcessBase):
    """Phase 1: video chat with --mm-process-config + --mm-enable-dp-encoder.

    [Test Category] Parameter
    [Test Target] --mm-process-config; --mm-enable-dp-encoder
    """

    # opt-in: the base sets __test__ = False and pytest inherits it
    __test__ = True

    mm_process_config = _MM_PROCESS_CONFIG

    def test_video_chat_with_config(self):
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

        global _prompt_tokens_with_config
        _prompt_tokens_with_config = prompt_tokens


class TestMmProcessWithoutConfig(_TestMmProcessBase):
    """Phase 2: the 448x448 resize + fps=2 sampling config must compress
    visual tokens, so prompt_tokens with the config is smaller than without."""

    # opt-in: the base sets __test__ = False and pytest inherits it
    __test__ = True

    mm_process_config = None

    def test_video_chat_without_config(self):
        self.assertIsNotNone(
            _prompt_tokens_with_config,
            "TestMmProcessWithConfig must run first and record prompt_tokens",
        )

        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        prompt_tokens_without_config, content = self._send_video_request()
        print(f"\n[without config] prompt_tokens: {prompt_tokens_without_config}")
        print(f"Video chat response: {content[:200]}...")

        self.assertLess(
            _prompt_tokens_with_config,
            prompt_tokens_without_config,
            f"prompt_tokens with config ({_prompt_tokens_with_config}) "
            f"should be smaller than without config ({prompt_tokens_without_config})",
        )


if __name__ == "__main__":
    unittest.main()
