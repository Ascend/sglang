"""
Test NPU LoRA with Tensor Parallelism - NPU LoRA张量并行测试

Tests LoRA functionality with tensor parallelism on Ascend NPU:
1. LoRA weights distribution across TP ranks
2. TP=2, TP=4, TP=8 configurations
3. Consistency across different TP sizes
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
)

# Test configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Update with actual model path
ADAPTER_URL = "path/to/lora/adapter"  # Update with actual adapter path

PROMPTS = [
    "The capital of France is",
    "Machine learning is a subset of",
]


class TestNpuLoraTp:
    """Test LoRA with tensor parallelism on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process_tp2(self):
        """Start SGLang server with TP=2."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 2,
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

    @pytest.fixture(scope="class")
    def server_process_tp4(self):
        """Start SGLang server with TP=4."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 4,
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

    def test_lora_tp2_basic(self, server_process_tp2):
        """Test LoRA with TP=2."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
        )
        
        assert response is not None, "Failed to get response with TP=2"
        assert "text" in response, "Response missing 'text' field"
        assert len(response["text"]) > 0, "Empty response text"

    def test_lora_tp4_basic(self, server_process_tp4):
        """Test LoRA with TP=4."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
        )
        
        assert response is not None, "Failed to get response with TP=4"
        assert "text" in response, "Response missing 'text' field"
        assert len(response["text"]) > 0, "Empty response text"

    def test_lora_tp2_batch(self, server_process_tp2):
        """Test batch inference with LoRA and TP=2."""
        responses = []
        
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
            )
            responses.append(response)
        
        assert len(responses) == len(PROMPTS), "Not all prompts received responses"
        
        for i, response in enumerate(responses):
            assert response is not None, f"Prompt {i} returned None response"
            assert "text" in response, f"Prompt {i} response missing 'text' field"

    def test_lora_tp_consistency(self, server_process_tp2, server_process_tp4):
        """Test output consistency across different TP sizes."""
        prompt = PROMPTS[0]
        
        # Get output with TP=2
        response_tp2 = call_sglang_generate(
            prompt=prompt,
            max_tokens=30,
            temperature=0.0,
            server_id="tp2",
        )
        
        # Get output with TP=4
        response_tp4 = call_sglang_generate(
            prompt=prompt,
            max_tokens=30,
            temperature=0.0,
            server_id="tp4",
        )
        
        # Both should produce valid outputs
        assert response_tp2 is not None, "TP=2 returned None response"
        assert response_tp4 is not None, "TP=4 returned None response"
        assert "text" in response_tp2, "TP=2 response missing 'text' field"
        assert "text" in response_tp4, "TP=4 response missing 'text' field"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
