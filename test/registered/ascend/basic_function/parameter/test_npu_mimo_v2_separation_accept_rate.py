import re
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME, check_role
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    MIMO_V2_FLASH_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="npu-performance",
    nightly=True,
)

accept_rate = 0.25
_temp_dir_obj = tempfile.TemporaryDirectory()
temp_dir = _temp_dir_obj.name

PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "800",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_BF16_DISPATCH": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
}

DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "800",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_NPU_PROFILING": "0",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_BF16_DISPATCH": "0",
}

PREFILL_ARGS = [
    "--attention-backend",
    "ascend",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    16384,
    "--trust-remote-code",
    "--max-running-requests",
    128,
    "--mem-fraction-static",
    0.75,
    "--swa-full-tokens-ratio",
    0.3,
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--disable-radix-cache",
]

DECODE_ARGS = [
    "--attention-backend",
    "ascend",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--trust-remote-code",
    "--max-running-requests",
    128,
    "--mem-fraction-static",
    0.83,
    "--swa-full-tokens-ratio",
    0.3,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--enable-multi-layer-eagle",
    "--disable-radix-cache",
    "--log-requests",
    "--log-requests-level",
    3,
    "--log-requests-target",
    temp_dir,
]

ROUTER_ARGS = [
    "--health-check-interval-secs",
    "3600",
    "--mini-lb",
]

ROUTER_ENVS = {}

MODEL_CONFIG = {
    "model_path": MIMO_V2_FLASH_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ROUTER_ARGS,
    "router_envs": ROUTER_ENVS,
}


class TestMimoV2acceptrate(TestAscendPerfMultiNodePdSepTestCaseBase):
    """Start the service and send a request to test the accept rate"""

    model_config = MODEL_CONFIG

    @check_role(allowed_roles=["router"])
    def test_request(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                },
            },
        )
        print("==="*50)
        print(f"*********temp_dir = {temp_dir}")
        self.assertEqual(response.status_code, 200)
        log_files = list(Path(temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)
        file_content = log_files[0].read_text()
        self.assertIn("accept rate", file_content)
        matches = re.findall(r"accept rate:\s*([\d.]+)", file_content)
        self.assertTrue(len(matches) > 0)
        current_accept_rate = float(matches[-1])
        self.assertGreater(current_accept_rate, accept_rate)

    def test_request_clear(self):
        _temp_dir_obj.cleanup()


if __name__ == "__main__":
    unittest.main()
