import unittest

import torch

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=29, suite="full-1-npu-a3", nightly=True)


class TestExternalModelsNPU(CustomTestCase):
    def test_external_model(self):
        # 检查 NPU 环境是否可用
        if not torch.npu.is_available():
            self.skipTest("NPU device not available, skipping NPU external model test")

        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
        prompt = "Today is a sunny day and I like"
        model_path = "Qwen/Qwen2-VL-2B-Instruct"

        engine = sgl.Engine(
            model_path=model_path,
            cuda_graph_max_bs=1,
            max_total_tokens=64,
            enable_multimodal=True,
            # NPU 专用配置
            attention_backend="torch_native",  # NPU 不支持 flashinfer/triton，使用 torch_native
            disable_cuda_graph=True,            # NPU 不需要 CUDA Graph
            disable_radix_cache=True,           # 禁用 radix cache 避免 NPU 兼容性问题
        )
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
