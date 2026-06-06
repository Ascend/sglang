import unittest

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_4B_GGUF_Q4_K_M_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=150, suite="nightly-2-npu-a3", nightly=True)

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestNPUPrefetchCheckpoints(CustomTestCase):
    """Test checkpoint prefetch functionality on NPU.

    [Test Category] Model Loading
    [Test Target] --weight-loader-prefetch-checkpoints
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=QWEN3_4B_GGUF_Q4_K_M_WEIGHTS_PATH,
            tp_size=1,
            attention_backend="ascend",
            disable_cuda_graph=True,
            mem_fraction_static=0.3,
            weight_loader_prefetch_checkpoints=True,
            cuda_graph_max_bs=1,
            max_total_tokens=256,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()

    def test_generate_with_prefetch(self):
        outputs = self.engine.generate(PROMPTS)
        self.assertEqual(len(outputs), len(PROMPTS))
        for i, output in enumerate(outputs):
            text = output["text"]
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0, f"Prompt {i} produced empty output")


if __name__ == "__main__":
    unittest.main()
