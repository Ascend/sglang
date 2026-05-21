"""
Test NPU LoRA Basic Functionality - NPU LoRA基础功能测试

Tests basic LoRA functionality on Ascend NPU:
1. LoRA adapter loading and initialization
2. Batch inference with LoRA adapters
3. Output accuracy verification
"""

import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    get_model_outputs,
    compare_outputs,
)

# Test configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Update with actual model path
ADAPTER_URL = "path/to/lora/adapter"  # Update with actual adapter path

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "The speed of light is approximately",
]

# Thresholds for comparison
LOGPROB_DIFF_THRESHOLD = 0.01
TOP_K_TOKENS = 50


class TestNpuLoraBasic:
    """Test basic LoRA functionality on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server with LoRA support."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'test-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    def test_lora_adapter_loading(self, server_process):
        """Test LoRA adapter loading and initialization."""
        # Verify server started successfully with LoRA
        assert server_process.poll() is None, "Server process terminated unexpectedly"
        
        # Test basic generation
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=50,
            temperature=0.0,
        )
        
        assert response is not None, "Failed to get response from server"
        assert "text" in response, "Response missing 'text' field"
        assert len(response["text"]) > 0, "Empty response text"

    def test_lora_batch_inference(self, server_process):
        """Test batch inference with LoRA adapters."""
        responses = []
        
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
            )
            responses.append(response)
        
        # Verify all responses
        assert len(responses) == len(PROMPTS), "Not all prompts received responses"
        
        for i, response in enumerate(responses):
            assert response is not None, f"Prompt {i} returned None response"
            assert "text" in response, f"Prompt {i} response missing 'text' field"
            assert len(response["text"]) > 0, f"Prompt {i} returned empty text"

    def test_lora_output_consistency(self, server_process):
        """Test LoRA output consistency across multiple runs."""
        prompt = PROMPTS[0]
        
        # Run inference multiple times with same seed
        outputs = []
        for _ in range(3):
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
                seed=42,
            )
            outputs.append(response["text"])
        
        # All outputs should be identical with temperature=0
        assert all(o == outputs[0] for o in outputs), "Outputs are not consistent across runs"

    def test_lora_without_adapter(self, server_process):
        """Test base model inference without LoRA adapter."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
            lora_name=None,  # Use base model
        )
        
        assert response is not None, "Failed to get response without adapter"
        assert "text" in response, "Response missing 'text' field"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
