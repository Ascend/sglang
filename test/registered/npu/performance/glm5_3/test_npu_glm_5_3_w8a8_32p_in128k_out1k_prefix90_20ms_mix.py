import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_GSM8K,
    GLM_5_3_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=5400,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_3_W8A8_32P_TWO_NODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEPEP_HCCL_BUFFSIZE": "900",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
}

GLM_5_3_W8A8_32P_TWO_NODE_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--attn-dp-size",
    16,
    "--chunked-prefill-size",
    131072,
    "--enable-mixed-chunk",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.84,
    "--context-length",
    200000,
    "--served-model-name",
    "glm-5.3-w8a8",
    "--cuda-graph-max-bs-decode",
    2,
    "--max-running-requests",
    32,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--load-balance-method",
    "round_robin",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
]

GLM_5_3_W8A8_32P_TWO_NODE_MODEL_CONFIG = {
    "model_path": GLM_5_3_W8A8_MODEL_PATH,
    "other_args": GLM_5_3_W8A8_32P_TWO_NODE_OTHER_ARGS,
    "node_envs": GLM_5_3_W8A8_32P_TWO_NODE_ENVS,
}


class TestNPUGLM5_3_W8A8_32P_In128k_Out1k_Prefix90_20ms(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """Test NPU performance for GLM-5.3-w8a8 two-node PD mix in128k out1k prefix90"""

    model_config = GLM_5_3_W8A8_32P_TWO_NODE_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    dataset_path = None
    max_concurrency = 64
    num_prompts = 256
    input_len = 131072
    output_len = 1024
    repeat_rate = 0.9
    dp = 16
    # tpot = 20
    output_token_throughput = 800

    def test_npu_glm_5_3_w8a8_32p_in128k_out1k_prefix90_20ms(self):
        """Run NPU performance test for GLM-5.3-w8a8 two-node PD mix"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
