"""TEMP (validation only): pure-prefill benchmark for the breakable prefill graph.

This file is intentionally non-merge. It mirrors the developer self-test
(serve_prefill_graph.sh): prefill-only workload (input 16384 / output 1),
chunked-prefill-size 8192, ascend_ops prefill=allgather, breakable prefill
graph. The 8P class uses TP8/DP8 (1024 token/rank); the 16P class uses
TP16/DP16 (512 token/rank, the range with the reported +2.73% gain).
"""

import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="sr1-perf-8-npu-a3", nightly=True)

COMMON_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_INTRA_DMA_ENABLE": "1",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "HCCL_BUFFSIZE": "1000",
    "DEEPEP_HCCL_BUFFSIZE": "1600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "160",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "compact",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
}


def _args(tp_size: int, dp_size: int):
    return [
        "--page-size",
        128,
        "--tp-size",
        tp_size,
        "--trust-remote-code",
        "--device",
        "npu",
        "--attention-backend",
        "dsv4",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.62,
        "--prefill-max-requests",
        32,
        "--max-prefill-tokens",
        80000,
        "--chunked-prefill-size",
        8192,
        "--max-running-requests",
        64,
        "--dp-size",
        dp_size,
        "--enable-dp-attention",
        "--moe-a2a-backend",
        "ascend_ops",
        "--moe-ascend-ops-config",
        '{"prefill":"allgather","decode":"low_latency"}',
        "--quantization",
        "modelslim",
        "--enable-dp-lm-head",
        "--kv-cache-dtype",
        "auto",
        "--disable-radix-cache",
        "--load-balance-method",
        "round_robin",
        "--cuda-graph-bs-decode",
        1,
        2,
        4,
        8,
        10,
        "--cuda-graph-backend-prefill",
        "breakable",
        "--cuda-graph-bs-prefill",
        128,
        512,
        1024,
        2048,
    ]


class _PurePrefillBase(TestNpuPerformanceTestCaseBase):
    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    envs = COMMON_ENVS
    dataset_name = "random"
    input_len = 16384
    output_len = 1
    num_prompts = 32
    max_concurrency = 1
    random_range_ratio = 1
    warmup_requests = 4
    request_rate = float("inf")
    timeout = 3600
    max_attempts = 2
    # Pure-prefill: decode-side thresholds do not apply.
    tpot = None
    output_token_throughput = None
    accept_rate = None

    def test_prefill_graph_purep(self):
        self.run_throughput()


class TestNPUDeepSeekV4FlashPrefillGraphPureP8P(_PurePrefillBase):
    other_args = _args(tp_size=8, dp_size=8)


if __name__ == "__main__":
    unittest.main()
