import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import (
    DefaultServerBase,
    openai_api_env,
)
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1200, suite="full-4-npu-a3", nightly=True)

QWEN3_NEXT_MODEL = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH

# NPU runtime environment required for HCCL/memory pools on Ascend.
_NPU_ENV = {
    **os.environ,
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "HCCL_BUFFSIZE": "200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
    "USE_VLLM_CUSTOM_ALLREDUCE": "1",
    "HCCL_EXEC_TIMEOUT": "200",
    "STREAMS_PER_DEVICE": "32",
    "AUTO_USE_UC_MEMORY": "0",
    "P2P_HCCL_BUFFSIZE": "20",
}


class _NpuDefaultServerBase(DefaultServerBase):
    """DefaultServerBase variant that injects the NPU runtime environment.

    The GPU kits (GSM8KMixin / KLDivergenceMixin / PrefixCacheBranchingMixin) are
    hardware-agnostic and only depend on ``base_url`` / ``model``, so they are
    fully reusable on NPU once the server is launched with the Ascend backend
    and the HCCL/MF_STORE environment variables.
    """

    api_key = "sk-123456"
    timeout = 3000

    @classmethod
    def setUpClass(cls):
        assert cls.model is not None, "Please set cls.model in subclass"
        with openai_api_env(cls.api_key):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=cls.other_args,
                env=_NPU_ENV,
            )


class TestQwen3Next80B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-Next-80B-A3B-Instruct model on the GSM8K dataset is no less than 0.92.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-Next-80B-A3B-Instruct
    """

    model = QWEN3_NEXT_MODEL
    accuracy = 0.92
    other_args = [
        "--tp-size",
        "4",
        "--disable-cuda-graph",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        0.8,
        "--disable-radix-cache",
    ]


# ---------------------------------------------------------------------------
# Ported from sgl-project/sglang/test/registered/models_e2e/test_qwen3_next_models.py
# ---------------------------------------------------------------------------
# The GPU source defines 4 classes for the hybrid-Mamba Qwen3-Next-80B-A3B model:
#   - TestQwen3NextLazyExtraBuffer            (page_size=1, track_interval=2)
#   - TestQwen3NextLazyExtraBufferLargePage   (page_size=2, track_interval=2)
#   - TestQwen3NextLazyExtraBufferAllocFail   (manual-only, skipped -> NOT ported)
#   - TestQwen3NextLazyExtraBufferLargePageAllocFail (manual-only -> NOT ported)
# Porting changes:
#   - model: switched to local modelscope path QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
#   - --attention-backend: triton -> ascend (NPU does not support triton)
#   - --disable-cuda-graph: added (NPU convention)
#   - --mamba-scheduler-strategy extra_buffer_lazy: kept (NPU supports extra_buffer,
#     see sglang/srt/server_args._validate_mamba_extra_buffer which checks is_npu())
#   - --mamba-track-interval / --page-size: kept identical to GPU so the same
#     scheduling code paths are exercised. If page_size=1/2 is rejected by the
#     Ascend backend at runtime, fall back to the NPU default (128) and adjust
#     track_interval accordingly (see TestQwen3NextLazyExtraBufferDefaultPage below).
#   - gsm8k_accuracy_thres: 0.93 -> 0.92 (NPU baseline per QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST)
#   - DefaultServerBase -> _NpuDefaultServerBase (inject NPU env)
# Two non-skipped classes are ported as-is to mirror the GPU page-size coverage.

_COMMON_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--mamba-scheduler-strategy",
    "extra_buffer_lazy",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
]


def _make_args(*, page_size=1, track_interval=2):
    return [
        *_COMMON_ARGS,
        "--mamba-track-interval",
        str(track_interval),
        "--page-size",
        str(page_size),
    ]


class TestQwen3NextLazyExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, _NpuDefaultServerBase
):
    """Port of GPU TestQwen3NextLazyExtraBuffer (page_size=1).

    Observes: GSM8K accuracy, KL divergence on prefill/decode cache hit, prefix
    cache branching at cache_chunk_size=64.
    """

    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.92
    kl_div_thres = 0.002
    other_args = _make_args(page_size=1, track_interval=2)


class TestQwen3NextLazyExtraBufferLargePage(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, _NpuDefaultServerBase
):
    """Port of GPU TestQwen3NextLazyExtraBufferLargePage (page_size=2)."""

    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.92
    kl_div_thres = 0.002
    other_args = _make_args(page_size=2, track_interval=2)


class TestQwen3NextLazyExtraBufferDefaultPage(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, _NpuDefaultServerBase
):
    """NPU-fallback variant: uses the Ascend default page_size (128).

    Kept disabled by default; enable only if the page_size=1/2 classes above
    are rejected by the Ascend backend. track_interval must satisfy
    ``track_interval % page_size == 0`` (see server_args._validate_mamba_extra_buffer),
    so it is set to 128 here.
    """

    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.92
    kl_div_thres = 0.002
    other_args = [
        *_COMMON_ARGS,
        "--mamba-track-interval",
        "128",
    ]


# ---------------------------------------------------------------------------
# Ported from sgl-project/sglang/test/registered/models_e2e/test_qwen3_next_models_mtp.py
# ---------------------------------------------------------------------------
# Porting changes:
#   - model/attention-backend/disable-cuda-graph: same as above.
#   - --speculative-algorithm NEXTN: kept (NPU already has NEXTN precedent,
#     see test/manual/ascend/test_ascend_deepseek_mtp.py).
#   - --mamba-scheduler-strategy extra_buffer (non-lazy): kept. Per
#     server_args._validate_mamba_extra_buffer, lazy is unsupported with spec,
#     so the GPU original uses the non-lazy extra_buffer strategy; NPU supports it.
#   - --mamba-track-interval 128: kept (satisfies 128 % page_size == 0 for the
#     NPU default page_size=128).
#   - Topk class retains PrefixCacheBranchingMixin (per agreement; observe and
#     fall back by removing it if the combination fails on NPU).
#   - gsm8k_accuracy_thres: 0.93 -> 0.92 (NPU baseline).


class TestQwen3NextMTPTopk(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, _NpuDefaultServerBase
):
    """Port of GPU TestQwen3NextMTPTopk: tree (topk>1) NEXTN MTP.

    Observes: GSM8K accuracy, KL divergence (looser threshold 0.008 because tree
    candidates introduce more numerical perturbation), and prefix cache branching.
    """

    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.92
    kl_div_thres = 0.008
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, _NpuDefaultServerBase):
    """Port of GPU TestQwen3NextMTPV2: linear (topk=1) NEXTN MTP.

    Observes: GSM8K accuracy and KL divergence (tighter threshold 0.0035 because
    linear candidates produce less perturbation). PrefixCacheBranchingMixin is
    not used, matching the GPU source.
    """

    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.92
    kl_div_thres = 0.0035
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
