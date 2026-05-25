import os
import time
import unittest
from time import sleep

# import logger

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME, check_role, TestAscendMultiNodePdSepTestCaseBase
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)

# ConfigMap相关配置
CONFIGMAP_NAME = os.environ.get("KUBE_CONFIG_MAP")
NAMESPACE = os.environ.get("NAMESPACE")

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
        # "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        # "HCCL_SOCKET_IFNAME": NIC_NAME,
        # "GLOO_SOCKET_IFNAME": NIC_NAME,
        "TRANSFORMERS_VERBOSITY": "error",
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
        # "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        # "HCCL_SOCKET_IFNAME": NIC_NAME,
        # "GLOO_SOCKET_IFNAME": NIC_NAME,
        "TRANSFORMERS_VERBOSITY": "error",
    },
    "router_envs": {
        # "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        # "HCCL_SOCKET_IFNAME": NIC_NAME,
        # "GLOO_SOCKET_IFNAME": NIC_NAME,
        "TRANSFORMERS_VERBOSITY": "error",
    },
    "prefill_args": [
        "--disaggregation-mode",
        "prefill",
        "--nnodes",
        1,
        "--node-rank",
        "0",
        "--tp",
        16,
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.8,
        "--max-total-tokens",
        68000,
        "--context-length",
        68000,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        327680,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        16,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
        "--disable-cuda-graph",
    ],
    "decode_args": [
        "--disaggregation-mode",
        "decode",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--tp",
        16,
        "--moe-dense-tp-size",
        1,
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--watchdog-timeout",
        9000,
        "--mem-fraction-static",
        0.8,
        "--context-length",
        68000,
        "--disable-radix-cache",
        "--chunked-prefill-size",
        262144,
        "--max-prefill-tokens",
        68000,
        "--max-running-requests",
        128,
        "--cuda-graph-max-bs",
        32,
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--quantization",
        "modelslim",
        "--disaggregation-transfer-backend",
        "ascend",
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


def create_model_config_with_param(bucket_interval):
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


class TestBucketAdjustIntervalSecsValidation(TestAscendMultiNodePdSepTestCaseBase):
    """测试 --bucket-adjust-interval-secs 参数的合法性验证"""

    test_cases = [
        {"value": "1", "should_succeed": True, "description": "合法值: 最小正整数"},
        {"value": "4294967295", "should_succeed": True, "description": "合法值: 最大无符号32位整数"},
        {"value": "0", "should_succeed": False, "description": "非法值: 0（小于最小值）"},
        {"value": "4294967296", "should_succeed": False, "description": "非法值: 超过最大无符号32位整数"},
        {"value": "5.1", "should_succeed": False, "description": "非法值: 浮点数"},
        {"value": "abc", "should_succeed": False, "description": "非法值: 纯字母字符串"},
        {"value": "@#$", "should_succeed": False, "description": "非法值: 特殊字符"},
    ]

    @classmethod
    def setUpClass(cls):
        cls.degradation_tolerance = 0
        cls.model = DEEPSEEK_R1_W8A8_MODEL_PATH
        cls.config = MODEL_CONFIG_BASE.copy()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def is_router_server_running(self):
        """检查router服务器是否正常运行"""
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("127.0.0.1", self.port))
            sock.close()
            print(f"result is {result}")
            return result == 0
        except Exception:
            return False

    def print_test_case_info(self, test_case):
        """打印测试用例信息"""
        value = test_case["value"]
        should_succeed = test_case["should_succeed"]
        description = test_case["description"]
        print(f"\n{'=' * 60}")
        print(f"测试: {description}")
        print(f"参数值: '{value}'")
        print(f"期望结果: {'启动成功' if should_succeed else '启动失败'}")
        print("=" * 60)

    def kill_process_if_alive(self):
        try:
            kill_process_tree(self.process.pid)
            sleep(30)
        except Exception:
            # 忽略清理异常，可能进程已提前退出
            pass

    def test_bucket_adjust_interval_secs_validation(self):
        """测试 --bucket-adjust-interval-secs 参数的合法性验证"""
        print("=== 开始测试 --bucket-adjust-interval-secs 参数验证 ===\n")

        # self.print_test_case_info(self.test_cases[0])
        # self.assert_result(self.test_cases[0]["value"], self.is_router_server_running(), self.test_cases[0]["should_succeed"])
        # self.kill_process_if_alive()
        # time.sleep(5)  # 等待完全停止
        try:
            self.start_pd_server()

            for test_case in self.test_cases:
                self.print_test_case_info(test_case)

                value = test_case["value"]
                should_succeed = test_case["should_succeed"]

                self.__class__.model_config = create_model_config_with_param(value)

        finally:
            self.stop_sglang_thread()

        # 依次测试每个参数值
        for test_case in self.test_cases:
            # if test_case["value"] == self.initial_value:
            #     continue
            
            self.print_test_case_info(test_case)

            value = test_case["value"]
            should_succeed = test_case["should_succeed"]

            self.__class__.model_config = create_model_config_with_param(value)

            caught_exception = False
            try:
                self.start_pd_server()
                self.start_router_server()
            except Exception:
                caught_exception = True
            finally:
                self.stop_sglang_thread()

            self.assert_result(value, not caught_exception, should_succeed)

            time.sleep(5)  # 等待完全停止

        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)

    def assert_result(self, value, success, should_succeed):
        """断言测试结果"""
        if should_succeed:
            self.assertTrue(success, msg=f"参数 '{value}' 应该启动成功，但实际失败")
            print(f"✓ 验证通过: 服务启动成功")
        else:
            self.assertFalse(
                success, msg=f"参数 '{value}' 应该启动失败，但实际成功"
            )
            print(f"✓ 验证通过: 服务启动失败（预期行为）")


if __name__ == "__main__":
    unittest.main()
