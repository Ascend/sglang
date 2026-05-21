# 20260521 09:28:28 || DeepSeek R1 W8A8 MLAPO性能对比测试用例
import unittest
import time
from urllib.parse import urlparse

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH,
    TestAscendPerformanceTestCaseBase,
    run_aisbench,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import popen_launch_server, kill_process_tree

register_npu_ci(
    est_time=2400,
    suite="deepseek_mlapo",
    nightly=True,
    disabled=False,
)

# ====================== 基础配置 ======================
DEEPSEEK_R1_BASE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
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
    "w8a8_int8",
    "--served-model-name",
    "deepseek-r1",
]

# 测试参数
TEST_PARAMS = {
    "max_concurrency": 64,
    "num_prompts": 256,
    "input_len": 2048,
    "output_len": 1024,
    "random_range_ratio": 1,
    "benchmark_tool": BENCHMARK_TOOL_DEFAULT,
    "aisbench_dataset_type": AISBENCHMARK_DATASET_DEFAULT,
}


class TestNPUDeepSeekR1_W8A8_MLAPO(unittest.TestCase):
    """DeepSeek R1 W8A8 MLAPO优化前后性能对比测试"""
    
    # 类变量存储基准测试结果（同一测试类内共享）
    baseline_tpot = None
    base_url = "http://127.0.0.1:20166"
    parsed_url = urlparse(base_url)
    host = parsed_url.hostname
    port = parsed_url.port
    process = None

    def tearDown(self):
        """完全保留原有的进程清理逻辑，不做任何修改"""
        if self.process:
            try:
                kill_process_tree(self.process.pid)
            except Exception:
                pass
        self.process = None

    def _launch_server_and_run_test(self, extra_envs=None):
        """内部方法：启动服务器并返回性能指标"""
        # 合并环境变量
        env = DEEPSEEK_R1_BASE_ENVS.copy()
        if extra_envs:
            env.update(extra_envs)
        
        # 启动服务器
        self.process = popen_launch_server(
            DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH,
            self.base_url,
            timeout=1800,
            other_args=DEEPSEEK_R1_BASE_ARGS,
            env=env,
        )

        # 运行性能测试
        metrics = run_aisbench(
            host=self.host,
            port=str(self.port),
            model_path=DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH,
            dataset_type=TEST_PARAMS["aisbench_dataset_type"],
            dataset_path=None,
            input_len=TEST_PARAMS["input_len"],
            output_len=TEST_PARAMS["output_len"],
            max_concurrency=TEST_PARAMS["max_concurrency"],
            num_prompts=TEST_PARAMS["num_prompts"],
            random_range_ratio=TEST_PARAMS["random_range_ratio"],
        )

        # 关闭服务器
        self.tearDown()
        return metrics

    def test_01_baseline_performance(self):
        """第一步：运行原始参数基准测试"""
        print("\n" + "="*80)
        print("🔹 阶段1：原始参数基准测试")
        print("="*80)
        
        metrics = self._launch_server_and_run_test()
        self.__class__.baseline_tpot = float(metrics["mean_tpot"])
        
        print(f"\n✅ 基准测试完成：tpot = {self.__class__.baseline_tpot:.2f} ms")
        print("\n⏳ 等待20秒后开始MLAPO优化测试...")
        time.sleep(20)  # 测试间20秒延迟

    def test_02_mlapo_optimization(self):
        """第二步：运行MLAPO优化测试并验证效果"""
        self.assertIsNotNone(
            self.__class__.baseline_tpot,
            "基准测试未成功执行，无法进行对比"
        )

        print("\n" + "="*80)
        print("🔹 阶段2：MLAPO优化性能测试")
        print("="*80)
        
        # 开启MLAPO优化
        metrics = self._launch_server_and_run_test(
            extra_envs={"SGLANG_NPU_USE_MLAPO": "1"}
        )
        mlapo_tpot = float(metrics["mean_tpot"])
        
        # 性能对比
        tpot_reduction = self.__class__.baseline_tpot - mlapo_tpot
        print("\n" + "="*80)
        print("📊 最终性能对比结果")
        print("="*80)
        print(f"原始参数tpot: {self.__class__.baseline_tpot:.2f} ms")
        print(f"MLAPO优化后tpot: {mlapo_tpot:.2f} ms")
        print(f"tpot减少量: {tpot_reduction:.2f} ms")
        print(f"要求减少量: ≥ 2.00 ms")
        print("="*80)

        # 核心断言
        self.assertGreater(
            tpot_reduction,
            2.0,
            f"MLAPO优化效果不达标：tpot仅减少了{tpot_reduction:.2f}ms，要求至少减少2ms"
        )

        print(f"\n🎉 MLAPO优化验证通过！tpot减少了{tpot_reduction:.2f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
