import unittest

from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_npu_ci(est_time=300, suite="nightly-2-npu-a3", nightly=True)


class TestNPULlama32OneB(GSM8KMixin, DefaultServerBase):
    """Test Llama-3.2-1B-Instruct generation model on NPU.

    [Test Category] Generation Model
    [Test Target] Llama-3.2-1B-Instruct, GSM8K accuracy
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    gsm8k_accuracy_thres = 0.35
    gsm8k_num_questions = 100
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
    ]


class TestNPUQwen3EightB(GSM8KMixin, DefaultServerBase):
    """Test Qwen3-8B generation model on NPU.

    [Test Category] Generation Model
    [Test Target] Qwen3-8B, GSM8K accuracy
    """

    model = QWEN3_8B_WEIGHTS_PATH
    gsm8k_accuracy_thres = 0.75
    gsm8k_num_questions = 200
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
        "--trust-remote-code",
    ]


class TestNPUQwen25SevenB(GSM8KMixin, DefaultServerBase):
    """Test Qwen2.5-7B-Instruct generation model on NPU.

    [Test Category] Generation Model
    [Test Target] Qwen2.5-7B-Instruct, GSM8K accuracy
    """

    model = QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
    gsm8k_accuracy_thres = 0.70
    gsm8k_num_questions = 200
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.7",
    ]


if __name__ == "__main__":
    unittest.main()
