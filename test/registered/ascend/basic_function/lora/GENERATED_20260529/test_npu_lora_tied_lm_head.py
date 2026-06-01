import multiprocessing as mp
import os
import shutil
import tempfile
import unittest

import torch

# Check if required dependencies are available
try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

TEST_PROMPTS = [
    "AI is a field of computer science focused on",
    "The capital of France is",
]
MAX_NEW_TOKENS = 16


def create_lora_adapter_with_lm_head(base_model_path: str, output_dir: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

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
    assert len(lm_head_keys) > 0, f"Expected lm_head LoRA weights in adapter"

    print(f"Created LoRA adapter at {output_dir}")
    print(f"  lm_head keys: {lm_head_keys}")

    del peft_model, model


@unittest.skipUnless(PEFT_AVAILABLE, "peft and accelerate not available")
class TestNPULoRATiedLMHead(CustomTestCase):
    """Test LoRA on models with tied lm_head on NPU.

    [Test Category] Feature
    [Test Target] tied lm_head LoRA, lm_head module handling
    """

    _adapter_dir = None

    @classmethod
    def setUpClass(cls):
        if not PEFT_AVAILABLE:
            raise unittest.SkipTest("peft and accelerate not available")
        super().setUpClass()
        cls._adapter_dir = tempfile.mkdtemp(prefix="sglang_test_npu_lora_tied_lm_head_")
        create_lora_adapter_with_lm_head(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH, cls._adapter_dir
        )

    @classmethod
    def tearDownClass(cls):
        if cls._adapter_dir and os.path.exists(cls._adapter_dir):
            shutil.rmtree(cls._adapter_dir)
        super().tearDownClass()

    def test_tied_lm_head_lora_basic(self):
        with SRTRunner(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=[self._adapter_dir],
            max_loras_per_batch=1,
            lora_backend="triton",
            lora_target_modules=["lm_head"],
            disable_cuda_graph=True,
            disable_radix_cache=True,
            mem_fraction_static=0.30,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                TEST_PROMPTS[:2],
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[self._adapter_dir] * len(TEST_PROMPTS[:2]),
            )

        self.assertEqual(len(srt_outputs.output_strs), len(TEST_PROMPTS[:2]))
        for output in srt_outputs.output_strs:
            self.assertGreater(len(output), 0)

    def test_tied_lm_head_lora_adapter_loading(self):
        import requests

        other_args = [
            "--enable-lora",
            "--lora-path",
            f"test_adapter={self._adapter_dir}",
            "--lora-target-modules",
            "lm_head",
            "--max-loras-per-batch",
            "1",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.30",
            "--lora-backend",
            "triton",
        ]
        process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": TEST_PROMPTS[0],
                    "lora_path": "test_adapter",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertGreater(len(result["text"]), 0)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main(warnings="ignore")
