"""Tests for --swa-full-tokens-ratio parameter.

The parameter controls: SWA pool tokens = Full pool tokens * ratio.
Only effective on Hybrid SWA models (DeepSeek V4, MiMo, Inkling, etc.).

Two test strategies:
- Unit test: mock Hybrid SWA model, verify pool size calculation (CPU only)
- Server test: launch a real Hybrid SWA model, verify inference and print pool sizes
"""

import contextlib
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
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

register_npu_ci(est_time=400, suite="full-16-npu-a3-test-debug", nightly=True)


KV_SIZE = 2  # bf16


@contextlib.contextmanager
def _mock_cpu_env(kv_size=2, tp_size=1, swa_eviction_interval=4):
    """Mock GPU-dependent functions for CPU-only testing."""
    from sglang.srt.environ import envs

    with (
        patch("torch._utils._element_size", return_value=kv_size),
        get_parallel().override(attn_tp_size=tp_size),
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(swa_eviction_interval),
    ):
        yield


def _make_model_runner(
    *,
    num_kv_heads=4,
    head_dim=64,
    v_head_dim=64,
    num_layers=32,
    use_mla_backend=False,
    is_hybrid_swa=False,
    full_attention_layer_ids=None,
    swa_attention_layer_ids=None,
    swa_num_kv_heads=None,
    swa_head_dim=None,
    swa_v_head_dim=None,
    swa_full_tokens_ratio=0.5,
    page_size=1,
    mambaish_config=None,
    disable_radix_cache=False,
    chunked_prefill_size=None,
    disable_overlap_schedule=False,
    sliding_window_size=None,
    speculative_num_draft_tokens=None,
    max_speculative_num_draft_tokens=None,
    speculative_algorithm=None,
    speculative_num_steps=None,
    speculative_eagle_topk=None,
    disaggregation_mode="null",
    max_running_requests=None,
    disaggregation_decode_extra_slots=0,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
):
    """Create a mock ModelRunner with the fields configurators need."""
    mr = MagicMock()

    mr.use_mla_backend = use_mla_backend
    mr.is_draft_worker = False
    mr.num_effective_layers = num_layers
    mr.start_layer = 0
    mr.end_layer = num_layers
    mr.dp_size = 1
    mr.page_size = page_size
    mr.mambaish_config = mambaish_config
    mr.is_hybrid_swa = is_hybrid_swa
    mr.sliding_window_size = sliding_window_size

    mc = SimpleNamespace()
    mc.head_dim = head_dim
    mc.v_head_dim = v_head_dim
    mc.kv_lora_rank = kv_lora_rank
    mc.qk_rope_head_dim = qk_rope_head_dim
    mc.is_hybrid_swa = is_hybrid_swa
    mc.full_attention_layer_ids = (
        full_attention_layer_ids
        if full_attention_layer_ids is not None
        else list(range(num_layers))
    )
    mc.swa_attention_layer_ids = (
        swa_attention_layer_ids if swa_attention_layer_ids is not None else []
    )
    mc.swa_head_dim = swa_head_dim or head_dim
    mc.swa_v_head_dim = swa_v_head_dim or v_head_dim
    mc.get_num_kv_heads = lambda tp_size, dcp_size=1: num_kv_heads
    mc.get_swa_num_kv_heads = lambda tp_size: swa_num_kv_heads or num_kv_heads
    mc.hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    mc.hf_config.get_text_config = lambda: mc.hf_config
    mc.linear_attn_registry_result = None
    mc.context_len = 8192
    mr.model_config = mc
    mr.kv_cache_dtype = "fake_bf16"

    sa = SimpleNamespace()
    sa.max_total_tokens = None
    sa.swa_full_tokens_ratio = swa_full_tokens_ratio
    sa.page_size = page_size
    sa.disable_radix_cache = disable_radix_cache
    sa.chunked_prefill_size = chunked_prefill_size
    sa.disable_overlap_schedule = disable_overlap_schedule
    sa.speculative_num_draft_tokens = speculative_num_draft_tokens
    sa.max_speculative_num_draft_tokens = (
        max_speculative_num_draft_tokens or speculative_num_draft_tokens
    )
    sa.speculative_algorithm = speculative_algorithm
    sa.speculative_num_steps = speculative_num_steps
    sa.speculative_eagle_topk = speculative_eagle_topk
    sa.disaggregation_mode = disaggregation_mode
    sa.max_running_requests = max_running_requests
    sa.disaggregation_decode_extra_slots = disaggregation_decode_extra_slots
    sa.enable_hisparse = False
    sa.enable_dsa_cache_layer_split = False
    sa.kv_cache_dtype = "auto"
    mr.server_args = sa

    spec = MagicMock()
    spec.is_eagle.return_value = False
    spec.is_standalone.return_value = False
    spec.is_dflash.return_value = False
    spec.is_dflash_family.return_value = False
    spec.is_none.return_value = True
    mr.spec_algorithm = spec

    mr.layer_info = SimpleNamespace(
        start_layer=0, end_layer=num_layers, num_effective_layers=num_layers
    )
    mr.ps = ParallelState.trivial()
    mr.pp_group = SimpleNamespace(rank_in_group=0)
    mr.spec_aux_config = SimpleNamespace(
        eagle_draft_num_layers=None, dflash_draft_num_layers=None
    )

    return mr


def _full_per_token(mr):
    mc = mr.model_config
    return mc.get_num_kv_heads(1) * (mc.head_dim + mc.v_head_dim) * KV_SIZE


def _swa_per_token(mr):
    mc = mr.model_config
    return mc.get_swa_num_kv_heads(1) * (mc.swa_head_dim + mc.swa_v_head_dim) * KV_SIZE


def _actual_memory_used(mr, config):
    """Compute actual memory consumed by the pool sizes in config."""
    mc = mr.model_config
    full_pt = _full_per_token(mr)
    swa_pt = _swa_per_token(mr)
    nf = len(mc.full_attention_layer_ids)
    ns = len(mc.swa_attention_layer_ids)

    if mr.is_hybrid_swa:
        full = config.full_max_total_num_tokens or 0
        swa = config.swa_max_total_num_tokens or 0
        return full * full_pt * nf + swa * swa_pt * ns
    else:
        return config.max_total_num_tokens * full_pt * (nf + ns)


class TestSwaFullTokensRatio(CustomTestCase):
    """Unit tests for --swa-full-tokens-ratio via pool_configurator.

    Fully references test_pool_configurator.py's TestHybridSWAConfigurator
    pattern: mock Hybrid SWA model -> run pool configurator -> assert
    pool sizes and memory invariants.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    [Scenario] S1: ratio controls SWA pool proportionally
    """

    def _make_swa_runner(self, full_layers=16, swa_layers=16, ratio=0.5, page_size=1):
        return _make_model_runner(
            is_hybrid_swa=True,
            full_attention_layer_ids=list(range(full_layers)),
            swa_attention_layer_ids=list(range(full_layers, full_layers + swa_layers)),
            swa_num_kv_heads=4,
            page_size=page_size,
            swa_full_tokens_ratio=ratio,
        )

    def _run(self, available_bytes, **kwargs):
        mr = self._make_swa_runner(**kwargs)
        with _mock_cpu_env():
            from sglang.srt.model_executor.pool_configurator import (
                create_memory_pool_configurator,
            )

            cfg = create_memory_pool_configurator(mr)
            config = cfg.calculate_pool_sizes(available_bytes, mr.server_args.page_size)
        return mr, cfg, config

    def test_memory_utilization(self):
        """Memory used <= available and within 1% of available."""
        available = 10_000_000
        mr, _, config = self._run(available)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertGreater(used, available * 0.99)

    def test_ratio_respected(self):
        """swa_tokens = full_tokens * ratio for various ratios."""
        available = 10_000_000
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            _, _, config = self._run(available, ratio=ratio, page_size=1)
            full = config.full_max_total_num_tokens
            swa = config.swa_max_total_num_tokens
            self.assertEqual(
                swa,
                int(full * ratio),
                f"ratio={ratio}: swa={swa} != full({full}) * {ratio}",
            )

    def test_ratio_with_page_alignment(self):
        """With page_size=128, swa_tokens = floor(full_tokens * ratio / 128) * 128."""
        available = 10_000_000
        _, _, config = self._run(available, ratio=0.5, page_size=128)
        full = config.full_max_total_num_tokens
        swa = config.swa_max_total_num_tokens
        self.assertEqual(full % 128, 0)
        self.assertEqual(swa % 128, 0)
        self.assertEqual(swa, (int(full * 0.5) // 128) * 128)

    def test_max_total_equals_full(self):
        """For Hybrid SWA, max_total_num_tokens = full_max_total_num_tokens."""
        _, _, config = self._run(10_000_000)
        self.assertEqual(config.max_total_num_tokens, config.full_max_total_num_tokens)

    def test_constraint_respected(self):
        """full_tokens = constrained value after re-run."""
        mr, cfg, _ = self._run(10_000_000, page_size=1)
        with _mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(200, page_size=1)
        self.assertEqual(config.full_max_total_num_tokens, 200)
        self.assertEqual(config.swa_max_total_num_tokens, 100)

    def test_constraint_memory_within_budget(self):
        """After constraint, memory <= original budget."""
        available = 10_000_000
        mr, cfg, original = self._run(available, page_size=1)
        user_limit = original.full_max_total_num_tokens // 2
        with _mock_cpu_env():
            config = cfg.calculate_pool_sizes_from_max_tokens(
                user_limit, mr.server_args.page_size
            )
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        original_used = _actual_memory_used(mr, original)
        self.assertAlmostEqual(used / original_used, 0.5, delta=0.01)

    def test_different_layer_counts(self):
        """Asymmetric full/swa layer counts."""
        available = 10_000_000
        mr, _, config = self._run(available, full_layers=24, swa_layers=8, ratio=0.5)
        used = _actual_memory_used(mr, config)
        self.assertLessEqual(used, available)
        self.assertEqual(
            config.swa_max_total_num_tokens,
            int(config.full_max_total_num_tokens * 0.5),
        )


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

_POOL_LOG_PATTERN = re.compile(
    r"full_max_total_num_tokens=(\d+).*swa_max_total_num_tokens=(\d+)"
)


class TestSwaFullTokensRatioServer(CustomTestCase):
    """Verify --swa-full-tokens-ratio on a real Hybrid SWA model (MiMo V2 Flash).

    Launches the server, sends an inference request, and prints the
    Full and SWA pool sizes from server logs.

    [Test Category] Parameter
    [Test Target] --swa-full-tokens-ratio
    [Scenario] S2: parameter accepted on Hybrid SWA model, pool sizes printed
    """

    model = MIMO_V2_FLASH_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT

    def _capture_pool_sizes(self, stdout):
        """Extract full/swa pool sizes from server stdout."""
        for line in stdout.splitlines():
            m = _POOL_LOG_PATTERN.search(line)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None, None

    def test_launch_and_print_pool_sizes(self):
        """S2: Launch MiMo V2 Flash, infer, and print Full/SWA pool sizes."""
        out_log_fd, out_log_path = tempfile.mkstemp(suffix=".log")
        err_log_fd, err_log_path = tempfile.mkstemp(suffix=".log")
        out_log_file = os.fdopen(out_log_fd, "w+", encoding="utf-8")
        err_log_file = os.fdopen(err_log_fd, "w+", encoding="utf-8")

        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_MIMO_BASE_ARGS,
            env=_MIMO_ENVS,
            return_stdout_stderr=(out_log_file, err_log_file),
        )
        try:
            # 1. Verify inference works
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

            # 2. Extract and print Full/SWA pool sizes from server logs
            out_log_file.seek(0)
            stdout = out_log_file.read()
            full, swa = self._capture_pool_sizes(stdout)

            if full is not None and swa is not None:
                ratio = swa / full
                print(
                    f"\n  [SWA Pool Info] full={full}, swa={swa}, "
                    f"ratio={ratio:.4f} (config=0.95)"
                )
                # self.assertAlmostEqual(
                #     ratio,
                #     0.95,
                #     delta=0.01,
                #     msg=f"SWA/Full ratio {ratio:.4f} deviates from config 0.95",
                # )
            else:
                print(
                    "\n  [SWA Pool Info] Pool size log not found in server stdout. "
                    "Look for '[unified-memory-pool]' or similar log lines."
                )
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.unlink(out_log_path)
            os.unlink(err_log_path)


if __name__ == "__main__":
    unittest.main()
