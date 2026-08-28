"""
Test multi-turn tool call with --tool-call-parser qwen3_coder and --reasoning-parser qwen3 on NPU.

Sends the pre-cleaned request body from tool_calls_cleaned.json to validate
the parser handles multi-turn conversations with pre-existing tool-call round-trips.
"""

import os
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.test_ascend_utils import (
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
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
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "DEEPEP_HCCL_BUFFSIZE": "1200",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

# Server arguments matching the kimi_k2_6 performance test configuration
SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    "16",
    "--mem-fraction-static",
    "0.94",
    "--max-running-requests",
    "32",
    "--chunked-prefill-size",
    "32768",
    "--context-length",
    "262144",
    "--max-prefill-tokens",
    "16384",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    "16",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-cuda-graph",
    # "--cuda-graph-bs-decode",
    # "1",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "5",
    "--speculative-draft-model-quantization",
    "unquant",
    "--prefill-delayer-max-delay-passes",
    "200",
    "--enable-prefill-delayer",
]


class TestNpuGdnMtpToolCallMultiTurn(GarbledDetectionBase):
    """Multi-turn tool call scenario using pre-cleaned data from tool_calls_cleaned.json with GDN and MTP.

    [Test Category] Functional
    [Test Target] Multi-turn tool call scenario with GDN and MTP.
    """

    # --- Model-specific configuration ---
    model_path = KIMI_K2_6_W4A8_MODEL_PATH
    server_args = SERVER_ARGS
    server_envs = SERVER_ENVS
    max_rounds = MAX_TEST_ROUNDS_NUM

    def test_tool_call_scenario(self):
        with open(REQUEST_DATA_BODY_FILE, "r", encoding="utf-8") as f:
            self.request_body = f.read()
        self._run_non_streaming_tool_call_scenario()


if __name__ == "__main__":
    unittest.main()
