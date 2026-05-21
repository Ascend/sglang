"""
Test NPU Pipeline Parallelism - NPU流水线并行测试

Tests pipeline parallelism on Ascend NPU:
1. Pipeline stage execution
2. Micro-batch processing
3. PP=2, PP=4 configurations
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
MODEL_NAME = "meta-llama/Llama-2-70b-hf"  # Update with actual model path

PROMPTS = [
    "The capital of France is",
    "The theory of relativity states that",
    "In the field of machine learning,",
]


class TestNpuPipelineParallel:
    """Test pipeline parallelism on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process_pp2(self):
        """Start SGLang server with PP=2."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "pp_size": 2,
            "device": "npu",
            "attention_backend": "ascend",
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def server_process_pp4(self):
        """Start SGLang server with PP=4."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "pp_size": 4,
            "device": "npu",
            "attention_backend": "ascend",
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    def test_pp2_basic(self, server_process_pp2):
        """Test basic inference with PP=2."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
        )
        
        assert response is not None, "PP=2 returned None response"
        assert "text" in response, "Response missing 'text' field"
        assert len(response["text"]) > 0, "Empty response text"

    def test_pp4_basic(self, server_process_pp4):
        """Test basic inference with PP=4."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
        )
        
        assert response is not None, "PP=4 returned None response"
        assert "text" in response, "Response missing 'text' field"
        assert len(response["text"]) > 0, "Empty response text"

    def test_pp2_batch(self, server_process_pp2):
        """Test batch inference with PP=2."""
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

    def test_pp_micro_batch(self, server_process_pp2):
        """Test micro-batch processing in pipeline."""
        # Send multiple requests to trigger micro-batch processing
        responses = []
        for i in range(5):
            response = call_sglang_generate(
                prompt=PROMPTS[i % len(PROMPTS)],
                max_tokens=20,
                temperature=0.0,
            )
            responses.append(response)
        
        assert all(r is not None for r in responses), "Some requests returned None"
        assert all("text" in r for r in responses), "Some responses missing 'text' field"

    def test_pp_with_tp(self, server_process_pp2):
        """Test pipeline parallelism combined with tensor parallelism."""
        # This test assumes a server with both PP and TP
        # Adjust tp_size based on available hardware
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 2,
            "pp_size": 2,
            "device": "npu",
            "attention_backend": "ascend",
        }
        
        server_process = run_sglang_server(**server_config)
        try:
            wait_for_server_ready(timeout=300)
            
            response = call_sglang_generate(
                prompt=PROMPTS[0],
                max_tokens=30,
                temperature=0.0,
            )
            
            assert response is not None, "PP+TP returned None response"
            assert "text" in response, "Response missing 'text' field"
        finally:
            kill_sglang_server(server_process)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
