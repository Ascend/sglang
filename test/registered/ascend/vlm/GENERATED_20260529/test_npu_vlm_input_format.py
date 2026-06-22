import json
import unittest
from io import BytesIO

import requests

from sglang import Engine
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.test.ascend.test_ascend_utils import (
    IMAGES_LOGO_PATH,
    IMAGES_MAN_PATH,
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

IMAGE_MAN_IRONING_URL = IMAGES_MAN_PATH
IMAGE_SGL_LOGO_URL = IMAGES_LOGO_PATH


class TestNPUVLMInputFormat(CustomTestCase):
    """Test VLM input format support on NPU.

    [Test Category] VLM Feature
    [Test Target] Image input format, processor output, multimodal inputs
    """

    model_path = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    chat_template = "qwen2-vl"

    @classmethod
    def setUpClass(cls):
        cls.image_urls = [IMAGE_MAN_IRONING_URL, IMAGE_SGL_LOGO_URL]
        cls.main_image = []
        for image_url in cls.image_urls:
            if image_url.startswith("http"):
                response = requests.get(image_url)
                from PIL import Image

                cls.main_image.append(Image.open(BytesIO(response.content)))
            else:
                from PIL import Image

                cls.main_image.append(Image.open(image_url))

    def setUp(self):
        self.engine = Engine(
            model_path=self.model_path,
            chat_template=self.chat_template,
            device="npu",
            mem_fraction_static=0.35,
            enable_multimodal=True,
            disable_cuda_graph=True,
            trust_remote_code=True,
            attention_backend="ascend",
        )

    def tearDown(self):
        self.engine.shutdown()

    def verify_response(self, output):
        out_text = output["text"].lower()
        assert any(
            w in out_text for w in ("taxi", "cab", "car", "man", "ironing")
        ), out_text
        has_logo = any(
            kw in out_text
            for kw in (
                "logo",
                "sgl",
                "software",
                "company",
                "text",
            )
        )
        assert has_logo or "image" in out_text, out_text

    def get_completion_request(self) -> ChatCompletionRequest:
        json_structure = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_urls[0]}},
                        {"type": "image_url", "image_url": {"url": self.image_urls[1]}},
                        {
                            "type": "text",
                            "text": "Describe both images in detail.",
                        },
                    ],
                }
            ],
        }
        json_str = json.dumps(json_structure)
        return ChatCompletionRequest.model_validate_json(json_str)

    async def test_accepts_image(self):
        req = self.get_completion_request()
        conv = generate_chat_conv(req, template_name=self.chat_template)
        text = conv.get_prompt()
        output = await self.engine.async_generate(
            prompt=text,
            image_data=self.main_image,
            sampling_params=dict(temperature=0.0, max_new_tokens=512),
        )
        self.verify_response(output)


class TestNPUVLMInputFormatAsync(
    TestNPUVLMInputFormat, unittest.IsolatedAsyncioTestCase
):
    """Test VLM input format with async support on NPU.

    [Test Category] VLM Feature
    [Test Target] Async image processing, multimodal engine
    """

    pass


if __name__ == "__main__":
    unittest.main()
