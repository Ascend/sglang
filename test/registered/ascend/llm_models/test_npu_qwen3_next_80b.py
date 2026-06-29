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

# NPU runtime env: HCCL/memory pools required by Ascend backend.
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
    """DefaultServerBase with NPU runtime env (HCCL/MF_STORE)."""

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


# Ported from sgl-project/sglang/test/registered/models_e2e/test_qwen3_next_models.py.
# NPU uses default page_size=128 (Ascend requires it; 1/2 produce garbage).
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


class TestQwen3NextLazyExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, _NpuDefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.92
    # NPU Mamba state-update nondeterminism: decode 0.002->0.003, prefill 0.02.
    kl_div_thres = 0.003
    kl_div_thres_prefill = 0.02
    other_args = [
        *_COMMON_ARGS,
        "--mamba-track-interval",
        "128",
    ]


# Ported from sgl-project/sglang/test/registered/models_e2e/test_qwen3_next_models_mtp.py.
# Both classes skipped: NPU mamba kernel BiShengIR UB overflow (static tiling).
@unittest.skip(
    "NPU mamba kernel fails to compile mamba_state_update for NEXTN"
)
class TestQwen3NextMTPTopk(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, _NpuDefaultServerBase
):
    # topk > 1 (tree) MTP on a hybrid-GDN model, on spec v2: the tree-aware mamba
    # state update lives in the spec v2 verify path, so mamba + topk > 1 no longer
    # falls back to spec v1.
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


@unittest.skip(
    "NPU mamba kernel BiShengIR UB overflow at compile time (static tiling)"
)
class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, _NpuDefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.92
    kl_div_thres = 0.0035
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--mem-fraction-static",
        "0.75",
        "--tp-size",
        "8",
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
