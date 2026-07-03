"""
End-to-end test for --enable-prefix-mm-cache parameter.
Tests the prefix multimodal cache functionality in encoder-only mode.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


class TestPrefixMMCacheE2E(CustomTestCase):
    """End-to-end test for --enable-prefix-mm-cache with multimodal encoding."""

    model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    base_host = "127.0.0.1"
    encode_port = "30000"
    encode_url = f"http://{base_host}:{encode_port}"
    image_url = (
        "https://raw.githubusercontent.com/sgl-project/sglang/main/test/images/cat.jpg"
    )
    video_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/test/videos/sample_video.mp4"
    audio_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/test/audio/sample_audio.mp3"

    @classmethod
    def setUpClass(cls):
        """Start encoder server with --enable-prefix-mm-cache."""
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--port",
            cls.encode_port,
            "--enable-prefix-mm-cache",
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up processes."""
        if cls.process_encode:
            try:
                kill_process_tree(cls.process_encode.pid)
            except Exception as e:
                print(f"Error killing process: {e}")

    def _parse_cache_log(self):
        """Parse encode server logs and return cache hit/miss information."""
        # This is a placeholder - actual implementation would parse server logs
        return []

    def test_health_check(self):
        """Test that encoder server is healthy."""
        response = requests.get(f"{self.encode_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)

    def test_image_encoding_with_cache(self):
        """Test that image encoding works with prefix mm cache enabled."""

        # First request (cache miss)
        response1 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "image",
                "data": [self.image_url],
            },
            timeout=60,
        )
        self.assertEqual(response1.status_code, 200)

        # Second request (should use cache)
        response2 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "image",
                "data": [self.image_url],
            },
            timeout=60,
        )
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response1.json()["meta_info"]["cached_tokens"], 0)
        self.assertGreater(response2.json()["meta_info"]["cached_tokens"], 0)

    def test_video_encoding_with_cache(self):
        """Test that video encoding works with prefix mm cache enabled."""

        # First request (cache miss)
        response1 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "video",
                "data": [self.video_url],
            },
            timeout=120,
        )
        self.assertEqual(response1.status_code, 200)

        # Second request (should use cache)
        response2 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "video",
                "data": [self.video_url],
            },
            timeout=120,
        )
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response1.json()["meta_info"]["cached_tokens"], 0)
        self.assertGreater(response2.json()["meta_info"]["cached_tokens"], 0)

    def test_audio_encoding_with_cache(self):
        """Test that audio encoding works with prefix mm cache enabled."""

        # First request (cache miss)
        response1 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "audio",
                "data": [self.audio_url],
            },
            timeout=60,
        )
        self.assertEqual(response1.status_code, 200)

        # Second request (should use cache)
        response2 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "audio",
                "data": [self.audio_url],
            },
            timeout=60,
        )
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response1.json()["meta_info"]["cached_tokens"], 0)
        self.assertGreater(response2.json()["meta_info"]["cached_tokens"], 0)

    def test_mixed_modality_encoding(self):
        """Test that mixed modality encoding works with prefix mm cache enabled."""

        # Request with image, video, and audio
        response = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "mixed",
                "data": [self.image_url, self.video_url, self.audio_url],
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
