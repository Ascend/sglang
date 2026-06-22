import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    TestAscendMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import retry
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="multi nodes testcase",
)

MODEL_CONFIG_CACHE_ENABLED = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "prefill_args": [
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
        "--enable-hierarchical-cache",
    ],
    "decode_args": [
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
    ],
    "router_args": [],
}


# ====================== Test Case ======================
class TestDeepSeekV32CacheAccuracy(TestAscendMultiNodePdSepTestCaseBase):

    @classmethod
    def setUpClass(cls):
        cls.accuracy = 0.95
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        super().setUpClass()

        cls.model_config = MODEL_CONFIG_CACHE_ENABLED
        cls.start_pd_server()
        cls.start_router_server()

    @retry()
    def test_accuracy(self):
        self.run_gsm8k_test(self.accuracy, num_shots=5)


if __name__ == "__main__":
    unittest.main()
