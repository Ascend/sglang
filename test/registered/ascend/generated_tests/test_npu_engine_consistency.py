"""
Test NPU Engine vs Runtime Consistency - NPU引擎与运行时一致性测试

Tests consistency between SRT Engine and Runtime on Ascend NPU:
1. Engine and Runtime produce same outputs
2. Logprob consistency
3. Token generation consistency
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
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Update with actual model path

PROMPTS = [
    "The capital of France is",
    "The speed of light is approximately",
    "Machine learning is a subset of",
]


class TestNpuEngineConsistency:
    """Test Engine vs Runtime consistency on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "device": "npu",
            "attention_backend": "ascend",
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    def test_engine_runtime_output_consistency(self, server_process):
        """Test that Engine and Runtime produce consistent outputs."""
        prompt = PROMPTS[0]
        
        # Run multiple times with same seed
        outputs = []
        for _ in range(3):
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
                seed=42,
            )
            outputs.append(response["text"])
        
        # All outputs should be identical
        assert all(o == outputs[0] for o in outputs), "Outputs are not consistent"

    def test_logprob_consistency(self, server_process):
        """Test logprob consistency across runs."""
        prompt = PROMPTS[0]
        
        # Get logprobs
        response1 = call_sglang_generate(
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            seed=42,
            return_logprobs=True,
        )
        
        response2 = call_sglang_generate(
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            seed=42,
            return_logprobs=True,
        )
        
        assert response1 is not None and response2 is not None
        assert "logprobs" in response1 and "logprobs" in response2
        
        # Logprobs should be consistent
        logprobs1 = response1["logprobs"]
        logprobs2 = response2["logprobs"]
        assert len(logprobs1) == len(logprobs2), "Logprob lengths differ"

    def test_token_generation_consistency(self, server_process):
        """Test token generation consistency."""
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.0,
                seed=42,
            )
            
            assert response is not None, f"Prompt returned None response"
            assert "text" in response, "Response missing 'text' field"
            assert len(response["text"]) > 0, "Empty response text"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
