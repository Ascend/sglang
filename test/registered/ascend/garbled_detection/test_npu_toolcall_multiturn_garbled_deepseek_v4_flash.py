"""
Test multi-turn tool call with --tool-call-parser qwen3_coder and --reasoning-parser qwen3 on NPU.

Sends the pre-cleaned request body from tool_calls_cleaned.json to validate
the parser handles multi-turn conversations with pre-existing tool-call round-trips.
"""

import os
import unittest

from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH,
)
from sglang.test.ascend.test_garbled_detection_utils import GarbledDetectionBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="full-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed manually.",
)

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REQUEST_DATA_BODY_FILE = os.path.join(_TEST_DIR, "tool_calls_cleaned.json")

MAX_TEST_ROUNDS_NUM = 50

# Environment variables matching the kimi_k2_6 performance test configuration
SERVER_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # deepep
    "DEEPEP_HCCL_BUFFSIZE": "1000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    # skip gpu branch
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
    # MTP (EAGLE) related envs
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

# Server arguments matching the kimi_k2_6 performance test configuration
SERVER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.7,
    "--prefill-max-requests",
    2,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    160,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    10,
    # MTP (EAGLE) configuration.
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
]


class TestNpuGdnMtpToolCallMultiTurn(GarbledDetectionBase):
    """Multi-turn tool call scenario using pre-cleaned data from tool_calls_cleaned.json with GDN and MTP.

    [Test Category] Functional
    [Test Target] Multi-turn tool call scenario with GDN and MTP.
    """

    # --- Model-specific configuration ---
    model_path = DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH
    server_args = SERVER_ARGS
    server_envs = SERVER_ENVS
    max_rounds = MAX_TEST_ROUNDS_NUM

    def test_tool_call_scenario(self):
        with open(REQUEST_DATA_BODY_FILE, "r", encoding="utf-8") as f:
            self.request_body = f.read()
        self._run_non_streaming_tool_call_scenario()


if __name__ == "__main__":
    unittest.main()
