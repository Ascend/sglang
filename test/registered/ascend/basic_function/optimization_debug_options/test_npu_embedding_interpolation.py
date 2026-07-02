import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.vlm_utils import IMAGE_MAN_IRONING_URL

register_npu_ci(est_time=600, suite="debug-full-1-npu-a3", nightly=True)


class TestPreciseEmbeddingInterpolation(CustomTestCase):
    """Testcase: verify --enable-precise-embedding-interpolation changes ViT
    position-embedding interpolation on Qwen3-VL, producing different outputs
    for the same image at temperature=0

    [Test Category] Parameter
    [Test Target] --enable-precise-embedding-interpolation
    """

    model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    server_args = [
        "--trust-remote-code",
        "--enable-multimodal",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
    ]

    def _launch_server(self, extra_args=None):
        env = os.environ.copy()
        # The parameter is only read inside the ViT cuda-graph path
        # (_prepare_graph_inputs → fast_pos_embed_interpolate).
        # Default eager path hardcodes torch.linspace and ignores the flag.
        env["SGLANG_VIT_ENABLE_CUDA_GRAPH"] = "true"

        args = list(self.server_args)
        if extra_args:
            args.extend(extra_args)

        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    def _image_request(self):
        return requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": IMAGE_MAN_IRONING_URL},
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in a sentence.",
                            },
                        ],
                    },
                ],
                "temperature": 0,
            },
        )

    def test_precise_embedding_interpolation_contrastive(self):
        # Run WITH --enable-precise-embedding-interpolation
        self._launch_server(
            extra_args=["--enable-precise-embedding-interpolation"]
        )
        resp_enabled = self._image_request()
        self.assertEqual(resp_enabled.status_code, 200)
        text_enabled = resp_enabled.json()["choices"][0]["message"]["content"]
        kill_process_tree(self.process.pid)

        # Run WITHOUT the flag (default: False)
        self._launch_server()
        resp_default = self._image_request()
        self.assertEqual(resp_default.status_code, 200)
        text_default = resp_default.json()["choices"][0]["message"]["content"]
        kill_process_tree(self.process.pid)

        # Both outputs should describe the same image
        for text in (text_enabled, text_default):
            text_lower = text.lower()
            self.assertTrue(
                any(w in text_lower for w in ("man", "person", "driver", "holding")),
                f"Expected person-related word in: {text}",
            )
            self.assertTrue(
                any(w in text_lower for w in ("car", "vehicle", "suv", "cab", "taxi")),
                f"Expected vehicle-related word in: {text}",
            )

        # Core assertion: outputs differ, proving the flag changes interpolation
        self.assertNotEqual(
            text_enabled,
            text_default,
            "Outputs should differ because --enable-precise-embedding-interpolation "
            "changes _get_interpolation_indices (align_corners=True vs False)",
        )


if __name__ == "__main__":
    unittest.main()
