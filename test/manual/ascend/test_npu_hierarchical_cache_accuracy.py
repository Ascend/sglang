import unittest
from types import SimpleNamespace

import numpy as np

from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    TestAscendMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import retry
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
)
from sglang.test.run_eval import run_eval

# ====================== Base Configuration ======================
BASE_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "ENABLE_MOE_NZ": "1",
    "HCCL_BUFFSIZE": "1536",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

BASE_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_USE_FIA_NZ": "1",
    "ENABLE_MOE_NZ": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_BUFFSIZE": "720",
    "SGLANG_DP_ROUND_ROBIN": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

BASE_PREFILL_ARGS = [
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    "16",
    "--mem-fraction-static",
    "0.8",
    "--quantization",
    "modelslim",
    "--context-length",
    "8192",
    "--chunked-prefill-size",
    "-1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]

BASE_DECODE_ARGS = [
    "--nnodes",
    "1",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    "16",
    "--mem-fraction-static",
    "0.8",
    "--quantization",
    "modelslim",
    "--context-length",
    "8192",
    "--chunked-prefill-size",
    "-1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--cuda-graph-bs",
    "256",
    "128",
    "64",
    "--watchdog-timeout",
    "9000",
    "--dtype",
    "bfloat16",
]
MODEL_CONFIG_CACHE_ENABLED = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "prefill_args": BASE_PREFILL_ARGS + ["--enable-hierarchical-cache"],
    "decode_args": BASE_DECODE_ARGS,
    "router_args": [],
}


# ====================== Test Case ======================
class TestDeepSeekV32CacheAccuracy(TestAscendMultiNodePdSepTestCaseBase):
    max_attempts = 3

    @classmethod
    def setUpClass(cls):
        cls.accuracy = 0.95
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        super().setUpClass()

    @retry()
    def test_accuracy(self):
        self.__class__.model_config = MODEL_CONFIG_CACHE_ENABLED
        self.start_pd_server()
        self.start_router_server()
        
        self.run_gsm8k_test(self.accuracy, num_shots=5)


if __name__ == "__main__":
    unittest.main()
