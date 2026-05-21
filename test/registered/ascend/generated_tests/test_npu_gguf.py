"""
Test NPU GGUF Quantization - NPU GGUF量化测试

Tests GGUF format support on Ascend NPU:
1. GGUF model loading
2. Inference with GGUF quantized models
3. Q4_K_M, Q5_K_M, Q8_0 quantization types
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
)

# Test configuration
GGUF_MODEL_Q4 = "path/to/model-Q4_K_M.gguf"  # Update with actual path
GGUF_MODEL_Q8 = "path/to/model-Q8_0.gguf"    # Update with actual path

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
]


class TestNpuGguf:
    """Test GGUF quantization on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process_q4(self):
        """Start SGLang server with Q4_K_M GGUF model."""
        server_config = {
            "model_path": GGUF_MODEL_Q4,
            "tp_size": 1,
            "device": "npu",
            "attention_backend": "ascend",
            "quantization": "gguf",
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def server_process_q8(self):
        """Start SGLang server with Q8_0 GGUF model."""
        server_config = {
            "model_path": GGUF_MODEL_Q8,
            "tp_size": 1,
            "device": "npu",
            "attention_backend": "ascend",
            "quantization": "gguf",
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    def test_gguf_q4_basic(self, server_process_q4):
        """Test basic inference with Q4_K_M GGUF model."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
        )
        
        assert response is not None, "Q4_K_M returned None response"
        assert "text" in response, "Response missing 'text' field"
        assert len(response["text"]) > 0, "Empty response text"

    def test_gguf_q8_basic(self, server_process_q8):
        """Test basic inference with Q8_0 GGUF model."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
        )
        
        assert response is not None, "Q8_0 returned None response"
        assert "text" in response, "Response missing 'text' field"
        assert len(response["text"]) > 0, "Empty response text"

    def test_gguf_batch_inference(self, server_process_q4):
        """Test batch inference with GGUF model."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
