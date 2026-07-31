import unittest

import torch

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import (
    QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=29, suite="full-1-npu-a3", nightly=True)


class TestExternalModelsNPU(CustomTestCase):
    def test_external_model(self):
        if not torch.npu.is_available():
            self.skipTest("NPU device not available, skipping NPU external model test")

        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
        prompt = "Today is a sunny day and I like"
        model_path = QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH

        engine = sgl.Engine(
            model_path=model_path,
            enable_multimodal=True,
            attention_backend="torch_native",
        )
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
