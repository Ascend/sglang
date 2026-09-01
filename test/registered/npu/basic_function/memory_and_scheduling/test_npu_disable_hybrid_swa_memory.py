"""Tests for --disable-hybrid-swa-memory parameter.

On Hybrid SWA models (DeepSeek V4, MiMo V2, Inkling, etc.), this flag
disables the separate SWA memory pool and falls back to a unified pool.

Two test strategies:
- Unit test: mock ModelConfig to verify is_hybrid_swa flag behavior
- Server test: launch with a real model to verify parameter acceptance
"""

import unittest
from unittest.mock import patch

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


# ---------------------------------------------------------------------------
# Unit tests: verify is_hybrid_swa flag controlled by the parameter
# ---------------------------------------------------------------------------

class TestDisableHybridSwaMemoryUnit(CustomTestCase):
    """Testcase: Verify --disable-hybrid-swa-memory controls is_hybrid_swa
    in ModelConfig._derive_hybrid_model().

    On a Hybrid SWA model:
    - disable_hybrid_swa_memory=False → is_hybrid_swa=True
    - disable_hybrid_swa_memory=True  → is_hybrid_swa=False

    [Test Category] Parameter
    [Test Target] --disable-hybrid-swa-memory
    [Scenario] D1: Hybrid SWA model default behavior
    [Scenario] D2: Hybrid SWA model with disable flag
    """

    def test_hybrid_swa_default_is_hybrid_swa_true(self):
        """D1: On a Hybrid SWA model, is_hybrid_swa=True by default."""
        from sglang.srt.configs.model_config import ModelConfig

        with patch.object(ModelConfig, "_maybe_pull_model_for_runai"):
            with patch.object(ModelConfig, "_maybe_pull_model_tokenizer_from_remote"):
                with patch.object(
                    ModelConfig, "get_hf_config"
                ) as mock_get_hf_config:
                    mock_get_hf_config.return_value = type(
                        "MockHFConfig",
                        (),
                        {"architectures": ["DeepseekV4ForCausalLM"]},
                    )()
                    with patch(
                        "sglang.srt.configs.model_config.is_hybrid_swa_model",
                        return_value=True,
                    ):
                        mc = ModelConfig(
                            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
                            trust_remote_code=True,
                            disable_hybrid_swa_memory=False,
                        )
                        mc._derive_hybrid_model()
                        self.assertTrue(
                            mc.is_hybrid_swa,
                            "is_hybrid_swa should be True when disable_hybrid_swa_memory=False",
                        )

    def test_hybrid_swa_disable_is_hybrid_swa_false(self):
        """D2: disable_hybrid_swa_memory=True → is_hybrid_swa=False."""
        from sglang.srt.configs.model_config import ModelConfig

        with patch.object(ModelConfig, "_maybe_pull_model_for_runai"):
            with patch.object(ModelConfig, "_maybe_pull_model_tokenizer_from_remote"):
                with patch.object(
                    ModelConfig, "get_hf_config"
                ) as mock_get_hf_config:
                    mock_get_hf_config.return_value = type(
                        "MockHFConfig",
                        (),
                        {"architectures": ["DeepseekV4ForCausalLM"]},
                    )()
                    with patch(
                        "sglang.srt.configs.model_config.is_hybrid_swa_model",
                        return_value=True,
                    ):
                        mc = ModelConfig(
                            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
                            trust_remote_code=True,
                            disable_hybrid_swa_memory=True,
                        )
                        mc._derive_hybrid_model()
                        self.assertFalse(
                            mc.is_hybrid_swa,
                            "is_hybrid_swa should be False when disable_hybrid_swa_memory=True",
                        )

    def test_non_hybrid_swa_is_always_false(self):
        """On a non-Hybrid-SWA model, is_hybrid_swa=False regardless of flag."""
        from sglang.srt.configs.model_config import ModelConfig

        with patch.object(ModelConfig, "_maybe_pull_model_for_runai"):
            with patch.object(ModelConfig, "_maybe_pull_model_tokenizer_from_remote"):
                with patch.object(
                    ModelConfig, "get_hf_config"
                ) as mock_get_hf_config:
                    mock_get_hf_config.return_value = type(
                        "MockHFConfig",
                        (),
                        {"architectures": ["LlamaForCausalLM"]},
                    )()
                    with patch(
                        "sglang.srt.configs.model_config.is_hybrid_swa_model",
                        return_value=False,
                    ):
                        # Test both flag values — is_hybrid_swa should always be False
                        for flag in [False, True]:
                            mc = ModelConfig(
                                model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
                                trust_remote_code=True,
                                disable_hybrid_swa_memory=flag,
                            )
                            mc._derive_hybrid_model()
                            self.assertFalse(
                                mc.is_hybrid_swa,
                                f"is_hybrid_swa should be False on non-SWA model "
                                f"(disable_hybrid_swa_memory={flag})",
                            )


# ---------------------------------------------------------------------------
# Server tests: verify parameter is accepted by the server
# ---------------------------------------------------------------------------

class TestDisableHybridSwaMemoryServer(CustomTestCase):
    """Testcase: Verify --disable-hybrid-swa-memory is accepted by the server
    and does not break inference.

    On non-Hybrid-SWA models (like Llama-3.2-1B), the parameter is a no-op
    but should be accepted without errors. This test verifies the parameter
    is properly parsed and does not prevent server startup or inference.

    [Test Category] Parameter
    [Test Target] --disable-hybrid-swa-memory
    [Scenario] D1: default (no flag)
    [Scenario] D2: explicit --disable-hybrid-swa-memory
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]

    def _send_request(self):
        """Send a simple generation request and verify the response."""
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Paris", resp.text)

    def test_default_without_disable(self):
        """D1: Server starts normally without --disable-hybrid-swa-memory."""
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._BASE_ARGS,
        )
        try:
            self._send_request()
        finally:
            kill_process_tree(process.pid)

    def test_explicit_disable(self):
        """D2: Server starts and inference succeeds with --disable-hybrid-swa-memory."""
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._BASE_ARGS + ["--disable-hybrid-swa-memory"],
        )
        try:
            self._send_request()
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()