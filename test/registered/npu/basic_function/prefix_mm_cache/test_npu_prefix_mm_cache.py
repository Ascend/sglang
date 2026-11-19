import base64
import io
import unittest

import requests
from PIL import Image

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)


def _make_image_data_url(size=512):
    """Build the input image in memory.

    The kernel-test image does not ship the shared images dataset, so the
    previous file-path asset made the encoder reject every request before the
    prefix cache could be exercised. A deterministic in-memory JPEG keeps the
    case self-contained while still sending the same bytes on both requests.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), "white").save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


image_url = _make_image_data_url()


class TestPrefixMmCache(TestDisaggregationBase):
    """Testcase: Verify set --enable-prefix-mm-cache Send multimodal requests, and the cache will be reused，

    [Test Category] Parameters
    [Test Target] --enable-prefix-mm-cache
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
        cls.start_encoder()
        cls.start_language()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

    @classmethod
    def start_encoder(cls):
        encoder_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.8",
            "--enable-prefix-mm-cache",
        ]

        cls.process_prefill = popen_launch_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
        )

    @classmethod
    def start_language(cls):
        language_args = [
            "--language-only",
            "--encoder-urls",
            cls.prefill_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--base-gpu-id",
            "1",
            "--attention-backend",
            "ascend",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--enable-cache-report",
        ]
        cls.process_decode = popen_launch_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
        )

    def test_image_encoding_with_cache(self):
        """Test that image encoding works with prefix mm cache enabled."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {"type": "text", "text": "Describe the image briefly."},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
        }

        # First request (cache miss)
        response1 = requests.post(
            f"{self.decode_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(response1.status_code, 200)

        # Second request (should use cache)
        response2 = requests.post(
            f"{self.decode_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        self.assertEqual(response2.status_code, 200)

        self.assertEqual(
            response1.json()["usage"]["prompt_tokens_details"]["cached_tokens"], 0
        )
        self.assertGreater(
            response2.json()["usage"]["prompt_tokens_details"]["cached_tokens"], 0
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
