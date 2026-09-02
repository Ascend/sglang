"""Tests for --swa-full-tokens-ratio parameter.

The parameter controls: SWA pool tokens = Full pool tokens * ratio.
Only effective on Hybrid SWA models (DeepSeek V4, MiMo, Inkling, etc.).

Two test strategies:
- Unit test: mock Hybrid SWA model, verify pool size calculation (CPU only)
- Server test: launch a real Hybrid SWA model, verify parameter is accepted
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from sglang.srt.runtime_context import get_parallel
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

register_npu_ci(est_time=400, suite="nightly-16-npu-a3-test-debug", nightly=True)


@contextlib.contextmanager
def _mock_cpu_env(kv_size=2, tp_size=1):
    """Mock GPU-dependent functions for CPU-only testing."""
    with (
        patch("torch._utils._element_size", return_value=kv_size),
        get_parallel().override(attn_tp_size=tp_size),
    ):
        yield


def _make_hybrid_swa_model_runner(
    *,
    full_layers=16,
    swa_layers=16,
    ratio=0.5,
    page_size=1,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
):
    """Create a mock ModelRunner for a Hybrid SWA model."""
    total_layers = full_layers + swa_layers
    mr = MagicMock()

    mr.use_mla_backend = False
    mr.is_draft_worker = False
    mr.num_effective_layers = total_layers
    mr.start_layer = 0
    mr.end_layer = total_layers
    mr.dp_size = 1
    mr.page_size = page_size
    mr.mambaish_config = None
    mr.is_hybrid_swa = True
    mr.sliding_window_size = None

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.kv_lora_rank = 512
    mc.qk_rope_head_dim = 64
    mc.is_hybrid_swa = True
    mc.full_attention_layer_ids = list(range(full_layers))
    mc.swa_attention_layer_ids = list(range(full_layers, total_layers))
    mc.swa_head_dim = head_dim
    mc.swa_v_head_dim = v_head_dim
    mc.get_num_kv_heads = lambda tp_size, dcp_size=1: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size: num_kv_heads
    mc.hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    mc.hf_config.get_text_config = lambda: mc.hf_config
    mc.linear_attn_registry_result = None
    mc.context_len = 8192
    mr.model_config = mc
    mr.kv_cache_dtype = "fake_bf16"

    sa = SimpleNamespace()
    sa.max_total_tokens = None
    sa.swa_full_tokens_ratio = ratio
    sa.page_size = page_size
    sa.disable_radix_cache = False
    sa.chunked_prefill_size = None
    sa.disable_overlap_schedule = False
    sa.speculative_num_draft_tokens = None
    sa.max_speculative_num_draft_tokens = None
    sa.speculative_algorithm = None
    sa.speculative_num_steps = None
    sa.speculative_eagle_topk = None
    sa.disaggregation_mode = "null"
    sa.max_running_requests = None
    sa.disaggregation_decode_extra_slots = 0
    sa.enable_hisparse = False
    sa.enable_dsa_cache_layer_split = False
    sa.kv_cache_dtype = "auto"
    mr.server_args = sa

    spec = MagicMock()
    spec.score_token_id = None
    mr.speculative_controller = spec

    return mr


def _run_pool_config(mr, available_bytes=10_000_000):
    """Run the pool configurator and return the config."""
    with _mock_cpu_env():
        from sglang.srt.model_executor.pool_configurator import (
            create_memory_pool_configurator,
        )

        cfg = create_memory_pool_configurator(mr)
        return cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)


_MIMO_BASE_ARGS = [
    "--tp-size",
    "16",
    "--trust-remote-code",
    "--device",
    "npu",
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


class TestSwaFullTokensRatioPool(CustomTestCase):
    """Testcase: Verify --swa-full-tokens-ratio controls SWA pool size
    via pool_configurator (CPU-only, no GPU needed).

    On a Hybrid SWA model: SWA pool tokens = Full pool tokens * ratio.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    [Scenario] S1: ratio controls SWA pool proportionally
    """

    def test_ratio_respected(self):
        """S1: swa_tokens = full_tokens * ratio for various ratios."""
        for ratio in [0.25, 0.5, 1.0]:
            mr = _make_hybrid_swa_model_runner(ratio=ratio, page_size=1)
            config = _run_pool_config(mr)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(
                swa,
                int(full * ratio),
                f"ratio={ratio}: swa={swa} != full({full}) * {ratio}",
            )

    def test_ratio_with_page_alignment(self):
        """S1: With page_size=128, swa_tokens = align(full * ratio)."""
        mr = _make_hybrid_swa_model_runner(ratio=0.5, page_size=128)
        config = _run_pool_config(mr)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_default_ratio_0_8(self):
        """S1: Default ratio=0.8 produces SWA pool = Full pool * 0.8."""
        mr = _make_hybrid_swa_model_runner(ratio=0.8, page_size=1)
        config = _run_pool_config(mr)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.8),
        )

    def test_max_total_equals_full(self):
        """S1: For Hybrid SWA, max_total_num_tokens = full_max_total_num_tokens."""
        mr = _make_hybrid_swa_model_runner(ratio=0.5)
        config = _run_pool_config(mr)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_ratio_1_0_equal_pools(self):
        """S1: ratio=1.0 → SWA pool = Full pool."""
        mr = _make_hybrid_swa_model_runner(ratio=1.0, page_size=1)
        config = _run_pool_config(mr)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            config.full_max_total_num_tokens,
        )


class TestSwaFullTokensRatioServer(CustomTestCase):
    """Testcase: Verify --swa-full-tokens-ratio is accepted by the server
    on a real Hybrid SWA model (MiMo V2 Flash).

    On Hybrid SWA models, the parameter controls SWA pool allocation.
    This test verifies the server starts and inference works with the
    parameter configured.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    [Scenario] S2: parameter accepted on Hybrid SWA model
    """

    model = MIMO_V2_FLASH_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT

    def _launch_and_verify(self, ratio):
        """Launch server with given ratio and verify inference works."""
        args = _MIMO_BASE_ARGS + ["--swa-full-tokens-ratio", ratio]
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=_MIMO_ENVS,
        )
        try:
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
        finally:
            kill_process_tree(process.pid)

    def test_ratio_default_accepted(self):
        """S2: Default ratio (0.8) is accepted, server starts and infers."""
        self._launch_and_verify("0.8")

    def test_ratio_custom_accepted(self):
        """S2: --swa-full-tokens-ratio 0.5 is accepted."""
        self._launch_and_verify("0.5")

    def test_ratio_max_accepted(self):
        """S2: --swa-full-tokens-ratio 1.0 is accepted."""
        self._launch_and_verify("1.0")


if __name__ == "__main__":
    unittest.main()
