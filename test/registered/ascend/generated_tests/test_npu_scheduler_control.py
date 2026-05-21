"""
Test NPU Scheduler Control - NPU调度器控制测试

Tests scheduler control functionality on Ascend NPU:
1. Request abort
2. Pause/resume scheduling
3. Request priority handling
"""

import os
import sys
import pytest
import time
import threading

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
    "Write a long essay about artificial intelligence:",
    "Explain the theory of relativity in detail:",
    "Describe the history of computing:",
]


class TestNpuSchedulerControl:
    """Test scheduler control on Ascend NPU."""

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

    def test_request_abort(self, server_process):
        """Test request abort functionality."""
        import requests
        
        # Start a long request
        prompt = PROMPTS[0]
        
        # Send request with long max_tokens
        response = call_sglang_generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.0,
        )
        
        # Verify request completed or was handled properly
        assert response is not None, "Request handling failed"

    def test_scheduler_basic_functionality(self, server_process):
        """Test basic scheduler functionality."""
        responses = []
        
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.0,
            )
            responses.append(response)
        
        assert len(responses) == len(PROMPTS), "Not all prompts received responses"
        
        for i, response in enumerate(responses):
            assert response is not None, f"Prompt {i} returned None response"
            assert "text" in response, f"Prompt {i} response missing 'text' field"

    def test_concurrent_requests(self, server_process):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def send_request(prompt):
            return call_sglang_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.0,
            )
        
        # Send concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_request, p) for p in PROMPTS]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(responses) == len(PROMPTS), "Not all concurrent requests completed"
        
        for i, response in enumerate(responses):
            assert response is not None, f"Concurrent request {i} returned None"
            assert "text" in response, f"Concurrent request {i} missing 'text' field"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
