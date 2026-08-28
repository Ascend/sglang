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

register_npu_ci(est_time=600, suite="nightly-4-npu-a3", nightly=True)

# Video processing config matching the reference command
_MM_PROCESS_CONFIG = json.dumps({
    "video": {
        "min_pixels": 76800,
        "max_pixels": 921600,
        "resized_height": 448,
        "resized_width": 448,
        "fps": 2,
        "min_frames": 4,
        "max_frames": 64,
    }
})

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
    """--mm-process-config + --mm-enable-dp-encoder -- verify video chat
    completion works with custom video processing and DP encoder.

    [Test Category] Parameter
    [Test Target] --mm-process-config, --mm-enable-dp-encoder
    """

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
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
            other_args=[
                *_COMMON_ARGS,
                "--mm-enable-dp-encoder",
                "--mm-process-config",
                _MM_PROCESS_CONFIG,
            ],
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_file.close()
        cls.err_file.close()
        os.unlink(cls.out_file.name)
        os.unlink(cls.err_file.name)

    def test_video_chat_completion(self):
        """Send a video chat request and verify a valid response."""

        # Server must be healthy
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        # Video chat request matching the reference curl command
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

        print(f"\nVideo chat response: {content[:200]}...")


if __name__ == "__main__":
    unittest.main()
