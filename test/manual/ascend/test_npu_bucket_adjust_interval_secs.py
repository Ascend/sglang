import time
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)

# ====================== Base Configuration ======================
MODEL_CONFIG_BASE = {
    "model_path": DEEPSEEK_R1_W8A8_MODEL_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "HCCL_BUFFSIZE": "2800",
        "HAS_INDEX_K": "1",
        "SGLANG_DEEPEP_BF16_DISPATCH": "0",
        "SGLANG_NPU_USE_MLAPO": "0",
        "SGLANG_USE_AG_AFTER_QLORA": "0",
        "USE_MULTI_STREAM": "1",
        "ENABLE_MOE_NZ": "1",
        "PROFILING_MODE": "dynamic",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
        "HCCL_BUFFSIZE": "1024",
        "HAS_INDEX_K": "1",
        "SGLANG_DEEPEP_BF16_DISPATCH": "0",
        "SGLANG_NPU_USE_MLAPO": "0",
        "SGLANG_NPU_USE_MLAPROLOG": "0",
        "USE_MULTI_STREAM": "1",
        "ENABLE_FUSED_MOE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "TASK_QUEUE_ENABLE": "0",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "router_envs": {
        "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
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
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.8,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        16,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        327680,
        "--max-prefill-tokens",
        68000,
        "--max-total-tokens",
        68000,
        "--context-length",
        68000,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--disable-cuda-graph",
        "--disaggregation-transfer-backend",
        "ascend",
    ],
    "decode_args": [
        "--nnodes",
        "2",
        "--disaggregation-mode",
        "decode",
        "--tp-size",
        16,
        "--mem-fraction-static",
        0.8,
        "--quantization",
        "modelslim",
        "--max-running-requests",
        128,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        262144,
        "--max-prefill-tokens",
        68000,
        "--context-length",
        68000,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--disaggregation-transfer-backend",
        "ascend",
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--cuda-graph-max-bs",
        32,
        "--prefill-round-robin-balance",
        "--load-balance-method",
        ROUND_ROBIN,
    ],
    "router_args": [
        "--pd-disaggregation",
        "--prefill-policy",
        "bucket",
        "--balance-rel-threshold",
        1.0001,
        "--balance-abs-threshold",
        32,
        # --bucket-adjust-interval-secs will be added dynamically
    ],
}


class TestBucketAdjustIntervalSecsValidation(TestAscendPerfMultiNodePdSepTestCaseBase):
    """测试 --bucket-adjust-interval-secs 参数的合法性验证"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG_BASE
    dataset_name = "random"
    request_rate = 1
    max_concurrency = 1
    num_prompts = 1
    input_len = 32
    output_len = 1
    random_range_ratio = 1

    # 测试参数
    test_cases = [
        # (value, should_succeed, description)
        ("1", True, "合法值: 最小正整数"),
        ("4294967295", True, "合法值: 最大无符号32位整数"),
        ("0", False, "非法值: 0（小于最小值）"),
        ("4294967296", False, "非法值: 超过最大无符号32位整数"),
        ("5.1", False, "非法值: 浮点数"),
        ("abc", False, "非法值: 纯字母字符串"),
        ("@#$", False, "非法值: 特殊字符"),
    ]

    # 等待router启动的超时时间（秒）
    router_startup_timeout = 60
    # 检查间隔
    check_interval = 5

    def create_model_config_with_param(self, bucket_interval):
        """创建带有指定 bucket-adjust-interval-secs 参数的配置"""
        config = MODEL_CONFIG_BASE.copy()
        config["router_args"] = MODEL_CONFIG_BASE["router_args"].copy()
        config["router_args"].extend(
            [
                "--bucket-adjust-interval-secs",
                bucket_interval,
            ]
        )
        return config

    def is_router_server_running(self):
        """检查router服务器是否正常运行"""
        try:
            # 尝试连接到router服务
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("127.0.0.1", self.port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def test_bucket_adjust_interval_secs_validation(self):
        """测试 --bucket-adjust-interval-secs 参数的合法性验证"""
        print("=== 开始测试 --bucket-adjust-interval-secs 参数验证 ===\n")

        # 先拉起PD节点（只拉一次，后续复用）
        print("步骤1: 启动PD节点（prefill + decode）")
        self.__class__.model_config = self.create_model_config_with_param("1")
        self.start_pd_server()

        # 等待PD节点启动
        print("等待PD节点启动...")
        time.sleep(30)

        try:
            # 依次测试每个参数值
            for value, should_succeed, description in self.test_cases:
                print(f"\n{'='*60}")
                print(f"测试: {description}")
                print(f"参数值: '{value}'")
                print(f"期望结果: {'启动成功' if should_succeed else '启动失败'}")
                print("=" * 60)

                # 更新配置
                self.__class__.model_config = self.create_model_config_with_param(value)

                # 启动router并检查结果
                success = self._test_single_value()
                
                # 验证结果
                if should_succeed:
                    self.assertTrue(
                        success, msg=f"参数 '{value}' 应该启动成功，但实际失败"
                    )
                    print(f"✓ 验证通过: 服务启动成功")
                else:
                    self.assertFalse(
                        success, msg=f"参数 '{value}' 应该启动失败，但实际成功"
                    )
                    print(f"✓ 验证通过: 服务启动失败（预期行为）")

                # 清理当前router（为下一次测试做准备）
                self.stop_sglang_thread()
                time.sleep(5)  # 等待完全停止
                
        finally:
            # 最后清理PD节点
            print("\n清理PD节点...")
            self.stop_sglang_thread()

        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)

    def _test_single_value(self):
        """测试单个参数值的router启动情况"""
        try:
            self.start_router_server()

            # 等待并检查启动状态
            start_time = time.time()
            while time.time() - start_time < self.router_startup_timeout:
                time.sleep(self.check_interval)

                # 检查服务是否正常
                if self.is_router_server_running():
                    return True

                # 检查线程是否异常退出
                if not self.sglang_thread.is_alive():
                    return False

            # 超时判断
            return self.is_router_server_running()

        except Exception as e:
            print(f"启动过程发生异常: {e}")
            return False


if __name__ == "__main__":
    unittest.main()
