"""
Test multi-turn tool call with --tool-call-parser qwen3_coder and --reasoning-parser qwen3 on NPU.

Sends the pre-cleaned request body from tool_calls_cleaned.json to validate
the parser handles multi-turn conversations with pre-existing tool-call round-trips.
"""

import os
import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_6_35B_A3B_WEIGHTS_PATH
from sglang.test.ascend.test_garbled_detection_utils import GarbledDetectionBase

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REQUEST_DATA_BODY_FILE = os.path.join(_TEST_DIR, "tool_calls_cleaned.json")

# Server environment variables
SERVER_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_VIT_ENABLE_CUDA_GRAPH": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
}

# Server args
SERVER_ARGS = [
    # "--base-gpu-id", "6",
    "--stream-response-default-include-usage",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    "2",
    "--chunked-prefill-size",
    "20480",
    "--max-prefill-tokens",
    "20480",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.75",
    "--enable-prefill-delayer",
    "--prefill-delayer-max-delay-passes",
    "30",
    "--cuda-graph-bs",
    "2",
    "4",
    "8",
    "10",
    "12",
    "16",
    "20",
    "24",
    "--max-running-requests",
    "24",
    "--context-length",
    "262144",
    "--mm-attention-backend",
    "ascend_attn",
    "--max-mamba-cache-size",
    "80",
    "--enable-metrics",
    "--enable-multimodal",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "qwen3",
]

MAX_TEST_ROUNDS_NUM = 100


class TestNpuGdnMtpToolCallMultiTurn(GarbledDetectionBase):
    """Multi-turn tool call scenario using pre-cleaned data from tool_calls_cleaned.json with GDN and MTP.

    [Test Category] Functional
    [Test Target] Multi-turn tool call scenario with GDN and MTP.
    """

    # --- Model-specific configuration ---
    model_path = QWEN3_6_35B_A3B_WEIGHTS_PATH
    server_args = SERVER_ARGS
    server_envs = SERVER_ENVS
    max_rounds = MAX_TEST_ROUNDS_NUM

    def test_tool_call_scenario(self):
        with open(REQUEST_DATA_BODY_FILE, "r", encoding="utf-8") as f:
            self.request_body = f.read()
        self._run_non_streaming_tool_call_scenario()


if __name__ == "__main__":
    unittest.main()
