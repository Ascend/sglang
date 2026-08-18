"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import unittest

import openai

from sglang.test.ascend.test_ascend_utils import (
    GEMMA_4_31B_WEIGHTS_PATH,
    KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
    LLAVA_ONEVISION_QWEN2_7B_OV_WEIGHTS_PATH,
    MINICPM_O_2_6_WEIGHTS_PATH,
    MINICPM_V_2_6_WEIGHTS_PATH,
    QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.vlm_utils import *
from sglang.test.ascend.vlm_utils import (
    AudioOpenAITestMixin,
    ImageOpenAITestMixin,
    OmniOpenAITestMixin,
    TestOpenAIMLLMServerBase,
    VideoOpenAITestMixin,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3200, suite="full-2-npu-a3", nightly=True)


class TestGemma4itServer(ImageOpenAITestMixin):
    model = GEMMA_4_31B_WEIGHTS_PATH
    extra_args = [
        "--disable-cuda-graph",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "4",
    ]


# Delete the mixin classes so that they are not collected by pytest
del (
    TestOpenAIMLLMServerBase,
    ImageOpenAITestMixin,
    VideoOpenAITestMixin,
    AudioOpenAITestMixin,
    OmniOpenAITestMixin,
)


if __name__ == "__main__":
    unittest.main()
