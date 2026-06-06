import json
import os
import unittest

from safetensors.torch import load_file

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=150, suite="full-2-npu-a3", nightly=True)

MODEL_PATH = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
LORA_PATH = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
TEST_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 16


class TestNPULoRALoadFromTensor(CustomTestCase):
    """Test LoRA load from tensor on NPU.

    [Test Category] RL LoRA
    [Test Target] Engine.load_lora_adapter_from_tensors
    """

    def setUp(self):
        """Set up test instance with Engine."""
        self.engine = sgl.Engine(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            mem_fraction_static=0.6,
            log_level="error",
        )

        # Load LoRA from local path
        self.lora_tensors = load_file(
            os.path.join(LORA_PATH, "adapter_model.safetensors")
        )
        with open(os.path.join(LORA_PATH, "adapter_config.json"), "r") as f:
            self.lora_config_dict = json.load(f)

    def tearDown(self):
        """Clean up test instance."""
        if hasattr(self, "engine") and self.engine:
            try:
                self.engine.shutdown()
            except Exception:
                pass  # Ignore shutdown errors

    def test_lora_e2e_load_from_tensor_params(self):
        """Test basic LoRA loading from tensor and inference."""
        result = self.engine.load_lora_adapter_from_tensors(
            lora_name="tool_calling_lora",
            tensors=self.lora_tensors,
            config_dict=self.lora_config_dict,
        )
        self.assertTrue(
            result.success,
            f"Failed to load LoRA from tensors: {result.error_message}",
        )

        output_without_lora = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
        )

        output_lora = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
            lora_path=["tool_calling_lora"],
        )

        # Verify LoRA produces different output than base model
        self.assertNotEqual(
            output_without_lora[0]["text"],
            output_lora[0]["text"],
            "LoRA should produce different output than base model",
        )


if __name__ == "__main__":
    unittest.main()
