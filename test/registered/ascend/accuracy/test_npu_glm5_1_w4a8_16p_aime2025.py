import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import GLM_5_1_W4A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

GLM_5_1_SINGLE_NODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "2000",
}

GLM_5_1_SINGLE_NODE_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    280000,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.8,
    "--served-model-name",
    "glm-5",
    "--cuda-graph-max-bs",
    8,
    "--max-running-requests",
    32,
    "--quantization",
    "modelslim",
    "--speculative-draft-model-quantization",
    "unquant",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestNPUGLM5_1_W4A8_16P_AIME2025(TestAscendAccuracyMultiNodePdMixTestCaseBase):
    """Test NPU accuracy for GLM-5.1-w4a8 16p single node on AIME 2025"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = GLM_5_1_W4A8_MODEL_PATH
    other_args = GLM_5_1_SINGLE_NODE_OTHER_ARGS
    envs = GLM_5_1_SINGLE_NODE_ENVS
    accuracy = 0.8
    dataset_type = "aime2025"
    dataset_name = "aime2025_gen"
    batch_size = 64
    max_out_len = 8192

    def test_npu_glm5_1_w4a8_16p_aime2025(self):
        """Run NPU accuracy test for GLM-5.1-w4a8 single node on AIME 2025"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
