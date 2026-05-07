import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_5_397B_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-pd-sep-3-node",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)


PREFILL_ENVS = {
    "ASCEND_USE_FIA": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "3000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_SOCKET_IFNAME": "enp196s0f0",
    "GLOO_SOCKET_IFNAME": "enp196s0f0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
}

DECODE_ENVS = {
    "ASCEND_USE_FIA": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_BUFFSIZE": "2400",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_SOCKET_IFNAME": "enp196s0f0",
    "GLOO_SOCKET_IFNAME": "enp196s0f0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
}


PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    1,
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.75,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    40,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    4000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--disable-cuda-graph",
    "--disable-overlap-schedule",
    "--tokenizer-worker-num",
    4,
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--mm-enable-dp-encoder",
    "--max-mamba-cache-size",
    160,
]

DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    2,
    "--tp-size",
    16,
    "--ep-size",
    32,
    "--mem-fraction-static",
    0.75,
    "--max-running-requests",
    192,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--enable-dp-attention",
    "--deepep-mode",
    "auto",
    "--enable-dp-lm-head",
    "--dp-size",
    4,
    "--cuda-graph-bs",
    8,
    16,
    24,
    32,
    40,
    48,
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--tokenizer-worker-num",
    4,
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
    "--disable-radix-cache",
]

MODEL_CONFIG = {
    "model_path": QWEN3_5_397B_W8A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": [],
    "router_envs": {},
}


class TestNPUQwen3_5_397B_W8A8_1P2D_24P_In3k5_Out1k5_50ms(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for Qwen3.5-397B-w8a8 1p2d PD separation ..."""

    model_config = MODEL_CONFIG

    max_concurrency = 440
    num_prompts = 1760
    input_len = 3500
    output_len = 1500
    tpot = 50

    def test_npu_qwen3_5_397b_1p2d_24p_3k5_1k_50ms(self):
        """Run NPU performance test for Qwen3.5-397B-w8a8 1p2d"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
