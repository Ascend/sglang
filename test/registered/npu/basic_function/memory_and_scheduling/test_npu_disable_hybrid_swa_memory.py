"""Tests for --disable-hybrid-swa-memory parameter.

On Hybrid SWA models (DeepSeek V4, MiMo V2, Inkling, etc.), this flag
disables the separate SWA memory pool and falls back to a unified pool.

Two test strategies:
- Unit test: mock ModelConfig to verify is_hybrid_swa flag behavior
- Server test: launch a real Hybrid SWA model to verify parameter acceptance
"""

import unittest
from unittest.mock import patch

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MIMO_V2_FLASH_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3-test-debug", nightly=True)


_MIMO_BASE_ARGS = [
    "--tp-size",
    "16",
    "--trust-remote-code",
    "--device", "npu",
    "--mem-fraction-static",
    "0.85",
    "--swa-full-tokens-ratio",
    "0.95",
    "--reasoning-parser",
    "mimo",
    "--attention-backend",
    "ascend",
    "--disable-piecewise-cuda-graph",
    "--base-gpu-id",
    "0",
    "--cuda-graph-bs",
    "1",
    "2",
    "4",
    "8",
    "16",
    "--dp-size",
    "4",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--quantization",
    "modelslim",
    "--skip-server-warmup",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-multi-layer-eagle",
    "--speculative-draft-model-quantization",
    "unquant",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]

_MIMO_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "ASCEND_USE_FIA": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "800",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_BF16_DISPATCH": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
}


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
                with patch.object(ModelConfig, "get_hf_config") as mock_get_hf_config:
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
                            model_path=MIMO_V2_FLASH_MODEL_PATH,
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
                with patch.object(ModelConfig, "get_hf_config") as mock_get_hf_config:
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
                            model_path=MIMO_V2_FLASH_MODEL_PATH,
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
                with patch.object(ModelConfig, "get_hf_config") as mock_get_hf_config:
                    mock_get_hf_config.return_value = type(
                        "MockHFConfig",
                        (),
                        {"architectures": ["LlamaForCausalLM"]},
                    )()
                    with patch(
                        "sglang.srt.configs.model_config.is_hybrid_swa_model",
                        return_value=False,
                    ):
                        for flag in [False, True]:
                            mc = ModelConfig(
                                model_path=MIMO_V2_FLASH_MODEL_PATH,
                                trust_remote_code=True,
                                disable_hybrid_swa_memory=flag,
                            )
                            mc._derive_hybrid_model()
                            self.assertFalse(
                                mc.is_hybrid_swa,
                                f"is_hybrid_swa should be False on non-SWA model "
                                f"(disable_hybrid_swa_memory={flag})",
                            )


class TestDisableHybridSwaMemoryServer(CustomTestCase):
    """Testcase: Verify --disable-hybrid-swa-memory is accepted by the server
    on a real Hybrid SWA model (MiMo V2 Flash).

    On Hybrid SWA models, this flag disables the separate SWA memory pool.
    This test verifies the server starts and inference works with the flag.

    [Test Category] Parameter
    [Test Target] --disable-hybrid-swa-memory
    [Scenario] D1: default (no flag)
    [Scenario] D2: explicit --disable-hybrid-swa-memory
    """

    model = MIMO_V2_FLASH_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT

    def _send_request(self):
        """Send a simple generation request and verify the response."""
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Paris", resp.text)

    def test_default_without_disable(self):
        """D1: Server starts normally without --disable-hybrid-swa-memory."""
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_MIMO_BASE_ARGS,
            env=_MIMO_ENVS,
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
            other_args=_MIMO_BASE_ARGS + ["--disable-hybrid-swa-memory"],
            env=_MIMO_ENVS,
        )
        try:
            self._send_request()
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
