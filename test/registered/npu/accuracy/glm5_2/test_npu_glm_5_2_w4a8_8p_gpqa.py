import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_2_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

GLM_5_2_W4A8_8P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEPEP_HCCL_BUFFSIZE": "1000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TRANSFORMERS_VERBOSITY": "error",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_BS": "16",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
}

GLM_5_2_W4A8_8P_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--chunked-prefill-size",
    2048,
    "--max-prefill-tokens",
    32768,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.8,
    "--served-model-name",
    "glm5",
    "--cuda-graph-bs",
    4,
    "--max-running-requests",
    8,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    # 临时诊断项: 稀疏注意力 kernel 挂死后留 30min 窗口, 便于登录机器
    # 手动 py-spy dump 全部 scheduler 进程 (定位 glm5.2 共性问题.md 试验 1)
    "--watchdog-timeout",
    1800,
]


class TestNPUGLM5_2_W4A8_8P_GPQA(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for GLM-5.2-w4a8 8p single node on gpqa_diamond"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = GLM_5_2_W4A8_MODEL_PATH
    other_args = GLM_5_2_W4A8_8P_OTHER_ARGS
    envs = GLM_5_2_W4A8_8P_ENVS
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    eval_batch_size = 64
    generation_config = {"max_tokens": 65536, "temperature": 1.0}

    def test_npu_glm_5_2_w4a8_8p_gpqa(self):
        """Run NPU accuracy test for GLM-5.2-w4a8 8p single node on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
