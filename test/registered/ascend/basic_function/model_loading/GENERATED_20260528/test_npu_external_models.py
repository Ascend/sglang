import unittest

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import QWEN3_4B_GGUF_Q4_K_M_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=45, suite="nightly-2-npu-a3", nightly=True)


class TestNPUExternalModels(CustomTestCase):
    """Test external model loading functionality on NPU.

    [Test Category] Model Loading
    [Test Target] External model package; Multimodal model loading
    """

    def test_external_model(self):
        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
        prompt = "Today is a sunny day and I like"
        model_path = QWEN3_4B_GGUF_Q4_K_M_WEIGHTS_PATH

        engine = sgl.Engine(
            model_path=model_path,
            attention_backend="ascend",
            disable_cuda_graph=True,
            mem_fraction_static=0.3,
            max_total_tokens=512,
            trust_remote_code=True,
        )
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
