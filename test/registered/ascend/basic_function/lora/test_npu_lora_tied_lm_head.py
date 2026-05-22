import os
import tempfile
import unittest

import requests
import torch

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    import subprocess

    subprocess.check_call(
        [
            "pip",
            "install",
            "peft",
            "accelerate",
            "torchao>=0.16.0",
            "-i",
            "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        ]
    )
    from peft import LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-2-npu-a3", nightly=True)

MAX_NEW_TOKENS = 16


def create_lora_adapter_with_lm_head(base_model_path: str, output_dir: str):
    """
    Create a LoRA adapter that targets lm_head,
    using a model with tie_word_embeddings=True.
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    assert (
        model.config.tie_word_embeddings
    ), f"{base_model_path} does not have tie_word_embeddings=True"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["lm_head"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    peft_model.save_pretrained(output_dir)

    from safetensors import safe_open

    safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
    f = safe_open(safetensors_path, framework="pt")
    lm_head_keys = [k for k in f.keys() if "lm_head" in k]

    assert os.path.isdir(
        output_dir
    ), f"LoRA adapter directory not created: {output_dir}"
    assert os.path.isfile(
        safetensors_path
    ), f"adapter_model.safetensors not found in {output_dir}"
    assert len(lm_head_keys) > 0, f"No lm_head keys found in adapter at {output_dir}"

    del peft_model, model
    torch.npu.empty_cache()


class TestNPULoRATiedLMHead(CustomTestCase):
    """Testcase: Test LoRA on models with tied lm_head (tie_word_embeddings=True) on NPU.
    When tie_word_embeddings=True, lm_head shares the same weight tensor as
    embed_tokens. This test validates that SGLang correctly handles this case
    by untying lm_head before LoRA wrapping on NPU backend.

    [Test Category] Feature
    [Test Target] tie_word_embeddings, lm_head LoRA
    """

    _adapter_dir = None
    base_model = QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._adapter_dir = tempfile.mkdtemp(prefix="sglang_npu_test_lora_tied_lm_head_")
        create_lora_adapter_with_lm_head(cls.base_model, cls._adapter_dir)

        other_args = [
            "--enable-lora",
            "--lora-path",
            f"tied_adapter={cls._adapter_dir}",
            "--max-loras-per-batch",
            "1",
            "--lora-backend",
            "triton",
            "--lora-target-modules",
            "lm_head",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
        ]

        cls.process = popen_launch_server(
            cls.base_model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if cls._adapter_dir and os.path.exists(cls._adapter_dir):
            import shutil

            shutil.rmtree(cls._adapter_dir)
        super().tearDownClass()

    def test_tied_lm_head_lora_inference(self):
        """Test inference with tied lm_head LoRA adapter."""
        prompts = [
            "The capital of France is",
            "AI is a field of computer science focused on",
        ]

        base_response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompts[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
            },
        )
        self.assertEqual(base_response.status_code, 200)
        base_output = base_response.json()["text"]

        lora_response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompts[0],
                "lora_path": "tied_adapter",
                "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
            },
        )
        self.assertEqual(lora_response.status_code, 200)
        lora_output = lora_response.json()["text"]

        self.assertNotEqual(base_output, lora_output, "LoRA should modify output")

    def test_tied_lm_head_lora_batch_inference(self):
        """Test batch inference with tied lm_head LoRA."""
        prompts = [
            "What is machine learning",
            "Explain neural networks",
        ]

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompts,
                "lora_path": "tied_adapter",
                "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), len(prompts))
        for result in results:
            self.assertGreater(len(result["text"]), 0)


if __name__ == "__main__":
    unittest.main()
