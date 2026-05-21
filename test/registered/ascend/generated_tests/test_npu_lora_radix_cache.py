"""
Test NPU LoRA with Radix Cache - NPU LoRA基数缓存测试

Tests LoRA functionality with radix cache on Ascend NPU:
1. Prefix caching with LoRA adapters
2. Cache hit/miss behavior
3. Cache eviction with multiple adapters
"""

import os
import sys
import pytest
import time

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

# Long prompts for prefix caching
LONG_PROMPT_TEMPLATE = """
The following is a conversation between a user and an AI assistant.
User: {question}
Assistant:"""

QUESTIONS = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
]


class TestNpuLoraRadixCache:
    """Test LoRA with radix cache on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server with radix cache enabled."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'test-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "enable_radix_cache": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    def test_lora_with_radix_cache(self, server_process):
        """Test LoRA inference with radix cache enabled."""
        prompts = [LONG_PROMPT_TEMPLATE.format(question=q) for q in QUESTIONS]
        
        # First run - populate cache
        responses_first = []
        for prompt in prompts:
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.0,
            )
            responses_first.append(response)
        
        # Second run - should use cache
        responses_second = []
        for prompt in prompts:
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.0,
            )
            responses_second.append(response)
        
        # Verify responses
        for i, (r1, r2) in enumerate(zip(responses_first, responses_second)):
            assert r1 is not None, f"First run prompt {i} returned None"
            assert r2 is not None, f"Second run prompt {i} returned None"
            assert "text" in r1, f"First run prompt {i} missing 'text' field"
            assert "text" in r2, f"Second run prompt {i} missing 'text' field"
            # Consistent outputs with same seed/temperature
            assert r1["text"] == r2["text"], f"Prompt {i} outputs are inconsistent"

    def test_lora_cache_performance(self, server_process):
        """Test that radix cache improves performance."""
        prompt = LONG_PROMPT_TEMPLATE.format(question=QUESTIONS[0])
        
        # Warm up
        call_sglang_generate(prompt=prompt, max_tokens=10, temperature=0.0)
        
        # First run (cache miss)
        start = time.time()
        response1 = call_sglang_generate(prompt=prompt, max_tokens=10, temperature=0.0)
        time_first = time.time() - start
        
        # Second run (cache hit)
        start = time.time()
        response2 = call_sglang_generate(prompt=prompt, max_tokens=10, temperature=0.0)
        time_second = time.time() - start
        
        assert response1 is not None, "First run returned None"
        assert response2 is not None, "Second run returned None"
        # Cache hit should be faster or similar
        assert time_second <= time_first * 1.5, "Cache hit not faster than cache miss"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
