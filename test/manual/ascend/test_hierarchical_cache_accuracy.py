import unittest
import numpy as np

from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    TestAscendMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
)

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

# ====================== Cache Configurations ======================
MODEL_CONFIG_CACHE_DISABLED = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "prefill_args": BASE_PREFILL_ARGS + ["--disable-radix-cache"],
    "decode_args": BASE_DECODE_ARGS,
    "router_args": [],
}

MODEL_CONFIG_CACHE_ENABLED = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "prefill_args": [arg for arg in BASE_PREFILL_ARGS if arg != "--disable-radix-cache"] + ["--enable-hierarchical-cache"],
    "decode_args": BASE_DECODE_ARGS,
    "router_args": [],
}

# ====================== Test Case ======================
class TestDeepSeekV32CacheAccuracy(TestAscendMultiNodePdSepTestCaseBase):
    @classmethod
    def setUpClass(cls):
        cls.degradation_tolerance = 0
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def run_gsm8k_with_config(self, config, repeat_times=5):
        self.__class__.model_config = config
        acc_list = []

        try:
            self.start_pd_server()
            self.start_router_server()

            for i in range(repeat_times):
                acc = self.run_gsm8k_test()
                acc_list.append(acc)

            avg_acc = np.mean(acc_list) if acc_list else 0.0

        finally:
            self.stop_sglang_thread()

        return avg_acc

    def test_accuracy(self):
        acc_off = self.run_gsm8k_with_config(MODEL_CONFIG_CACHE_DISABLED, repeat_times=5)
        acc_on = self.run_gsm8k_with_config(MODEL_CONFIG_CACHE_ENABLED, repeat_times=5)

        self.assertGreaterEqual(acc_on, acc_off - self.degradation_tolerance,
                                msg="Accuracy degraded after enabling cache!")


if __name__ == "__main__":
    unittest.main()