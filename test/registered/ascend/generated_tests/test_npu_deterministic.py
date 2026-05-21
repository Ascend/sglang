"""
Test NPU Deterministic Output - NPU确定性输出测试

Tests deterministic output with seed on Ascend NPU:
1. Same seed produces same output
2. Different seeds produce different outputs
3. Consistency across multiple runs
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
    "The theory of relativity states that",
    "In computer science,",
]


class TestNpuDeterministic:
    """Test deterministic output on Ascend NPU."""

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

    def test_same_seed_same_output(self, server_process):
        """Test that same seed produces same output."""
        prompt = PROMPTS[0]
        seed = 42
        
        outputs = []
        for _ in range(5):
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=1.0,  # Use temperature > 0 to test randomness control
                seed=seed,
            )
            outputs.append(response["text"])
        
        # All outputs should be identical with same seed
        assert all(o == outputs[0] for o in outputs), "Same seed produced different outputs"

    def test_different_seed_different_output(self, server_process):
        """Test that different seeds can produce different outputs."""
        prompt = PROMPTS[0]
        
        outputs = []
        for seed in [1, 2, 3]:
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=1.0,
                seed=seed,
            )
            outputs.append(response["text"])
        
        # At least some outputs should be different
        # Note: This is probabilistic, so we check that not all are identical
        unique_outputs = set(outputs)
        assert len(unique_outputs) > 1, "Different seeds produced identical outputs"

    def test_zero_temperature_deterministic(self, server_process):
        """Test that temperature=0 produces deterministic output."""
        prompt = PROMPTS[0]
        
        outputs = []
        for _ in range(5):
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
            )
            outputs.append(response["text"])
        
        # All outputs should be identical with temperature=0
        assert all(o == outputs[0] for o in outputs), "Temperature=0 produced different outputs"

    def test_deterministic_multiple_prompts(self, server_process):
        """Test determinism across multiple prompts."""
        for prompt in PROMPTS:
            outputs = []
            for _ in range(3):
                response = call_sglang_generate(
                    prompt=prompt,
                    max_tokens=20,
                    temperature=0.0,
                    seed=42,
                )
                outputs.append(response["text"])
            
            assert all(o == outputs[0] for o in outputs), f"Prompt '{prompt[:20]}...' not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
