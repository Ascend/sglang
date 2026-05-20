import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

# 注册CI任务
register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled=False,
)


# ====================== 基础配置 ======================
DEEPSEEK_R1_BASE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "HCCL_BUFFSIZE": "1600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TRANSFORMERS_VERBOSITY": "error",
}

DEEPSEEK_R1_BASE_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--dp-size",
    1,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.79,
    "--chunked-prefill-size",
    64000,
    "--context-length",
    66000,
    "--max-prefill-tokens",
    66000,
    "--max-total-tokens",
    66000,
    "--disable-radix-cache",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
]

# 全局变量存储基准测试的tpot值
BASELINE_TPOT = None


# ====================== 测试用例 ======================
class TestNPUDeepSeekR1_W8A8_Baseline(TestAscendPerformanceTestCaseBase):
    """DeepSeek R1 W8A8 NPU性能基准测试（原始参数）"""

    model = DEEPSEEK_R1_W8A8_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    other_args = DEEPSEEK_R1_BASE_ARGS
    envs = DEEPSEEK_R1_BASE_ENVS

    # 测试参数（可根据实际情况调整）
    max_concurrency = 64
    num_prompts = 256
    input_len = 2048
    output_len = 1024
    random_range_ratio = 1

    # 初始阈值（仅用于基准测试通过，实际对比在MLAPO用例中）
    tpot = 100
    output_token_throughput = 800

    def test_deepseek_r1_baseline_performance(self):
        """运行DeepSeek R1 W8A8基准性能测试并记录tpot"""
        global BASELINE_TPOT
        metrics = self.run_throughput_and_get_metrics()
        BASELINE_TPOT = float(metrics["mean_tpot"])
        self.assertIsNotNone(BASELINE_TPOT, "基准测试未获取到tpot数据")
        print(f"\n✅ 基准测试完成，原始参数tpot: {BASELINE_TPOT:.2f} ms")


class TestNPUDeepSeekR1_W8A8_MLAPO(TestAscendPerformanceTestCaseBase):
    """DeepSeek R1 W8A8 NPU性能测试（开启MLAPO优化）"""

    model = DEEPSEEK_R1_W8A8_MODEL_PATH
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    other_args = DEEPSEEK_R1_BASE_ARGS

    # 开启MLAPO优化的环境变量
    envs = {
        **DEEPSEEK_R1_BASE_ENVS,
        "SGLANG_NPU_USE_MLAPO": "1",
    }

    # 与基准测试保持相同的测试参数
    max_concurrency = 64
    num_prompts = 256
    input_len = 2048
    output_len = 1024
    random_range_ratio = 1

    def test_mlapo_optimization_effect(self):
        """测试MLAPO优化效果，断言tpot减少至少2ms"""
        global BASELINE_TPOT

        # 确保基准测试已完成
        self.assertIsNotNone(BASELINE_TPOT, "请先运行基准测试用例")

        # 运行MLAPO优化后的测试
        metrics = self.run_throughput_and_get_metrics()
        mlapo_tpot = float(metrics["mean_tpot"])

        print(f"\n📊 性能对比结果:")
        print(f"   原始参数tpot: {BASELINE_TPOT:.2f} ms")
        print(f"   MLAPO优化后tpot: {mlapo_tpot:.2f} ms")
        print(f"   tpot减少量: {BASELINE_TPOT - mlapo_tpot:.2f} ms")

        # 断言tpot减少至少2ms
        self.assertLess(
            mlapo_tpot,
            BASELINE_TPOT - 2,
            f"MLAPO优化效果不达标：tpot仅减少了{BASELINE_TPOT - mlapo_tpot:.2f}ms，要求至少减少2ms",
        )

        print(f"\n✅ MLAPO优化验证通过，tpot减少了{BASELINE_TPOT - mlapo_tpot:.2f} ms")


if __name__ == "__main__":
    unittest.main()
