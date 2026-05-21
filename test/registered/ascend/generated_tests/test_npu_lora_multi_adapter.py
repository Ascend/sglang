"""
Test NPU LoRA with Multiple Adapters - NPU多适配器LoRA测试

Tests multiple LoRA adapters functionality on Ascend NPU:
1. Dynamic adapter switching
2. Multiple adapters loaded simultaneously
3. Adapter-specific outputs
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
ADAPTER_URL_1 = "path/to/lora/adapter1"  # Update with actual adapter path
ADAPTER_URL_2 = "path/to/lora/adapter2"  # Update with actual adapter path

PROMPTS = [
    "The capital of France is",
    "Write a poem about spring:",
]


class TestNpuLoraMultiAdapter:
    """Test multiple LoRA adapters on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server with multiple LoRA adapters."""
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'adapter1':'{ADAPTER_URL_1}', 'adapter2':'{ADAPTER_URL_2}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        kill_sglang_server(server_process)

    def test_multi_adapter_loading(self, server_process):
        """Test loading multiple adapters simultaneously."""
        # Verify server started with multiple adapters
        assert server_process.poll() is None, "Server process terminated unexpectedly"
        
        # Test with first adapter
        response1 = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
            lora_name="adapter1",
        )
        
        # Test with second adapter
        response2 = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
            lora_name="adapter2",
        )
        
        assert response1 is not None, "Adapter1 returned None response"
        assert response2 is not None, "Adapter2 returned None response"
        assert "text" in response1, "Adapter1 response missing 'text' field"
        assert "text" in response2, "Adapter2 response missing 'text' field"

    def test_adapter_switching(self, server_process):
        """Test dynamic adapter switching."""
        prompt = PROMPTS[0]
        
        # Use adapter1
        response1 = call_sglang_generate(
            prompt=prompt,
            max_tokens=20,
            temperature=0.0,
            lora_name="adapter1",
        )
        
        # Switch to adapter2
        response2 = call_sglang_generate(
            prompt=prompt,
            max_tokens=20,
            temperature=0.0,
            lora_name="adapter2",
        )
        
        # Switch back to adapter1
        response3 = call_sglang_generate(
            prompt=prompt,
            max_tokens=20,
            temperature=0.0,
            lora_name="adapter1",
        )
        
        assert response1 is not None and response2 is not None and response3 is not None
        assert "text" in response1 and "text" in response2 and "text" in response3
        
        # Same adapter should produce consistent results
        assert response1["text"] == response3["text"], "Same adapter produced different outputs"

    def test_base_model_with_multi_adapter(self, server_process):
        """Test base model inference with multiple adapters loaded."""
        response = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=30,
            temperature=0.0,
            lora_name=None,  # Use base model
        )
        
        assert response is not None, "Base model returned None response"
        assert "text" in response, "Base model response missing 'text' field"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
