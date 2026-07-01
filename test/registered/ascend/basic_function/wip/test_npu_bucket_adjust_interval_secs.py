import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    ACTIVE_TEST_CLASS,
    SERVICE_PORT,
    check_role,
    launch_router,
    wait_for_prefill_decode_exit,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="multi nodes testcase",
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


class TestBucketAdjustIntervalSecsValidation(TestAscendPerfMultiNodePdSepTestCaseBase):
    """测试 --bucket-adjust-interval-secs 参数的合法性验证"""

    model_config = MODEL_CONFIG_BASE
    test_cases = [
        {"value": "1", "should_succeed": True, "description": "合法值: 最小正整数"},
        {
            "value": "4294967295",
            "should_succeed": True,
            "description": "合法值: 最大无符号32位整数",
        },
        {
            "value": "0",
            "should_succeed": False,
            "description": "非法值: 0（小于最小值）",
        },
        {
            "value": "4294967296",
            "should_succeed": False,
            "description": "非法值: 超过最大无符号32位整数",
        },
        {"value": "5.1", "should_succeed": False, "description": "非法值: 浮点数"},
        {
            "value": "abc",
            "should_succeed": False,
            "description": "非法值: 纯字母字符串",
        },
        {"value": "@#$", "should_succeed": False, "description": "非法值: 特殊字符"},
    ]

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = (
            "router"
            if "router" in cls.hostname
            else "prefill" if "prefill" in cls.hostname else "decode"
        )
        logger.info(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_server()

    @classmethod
    @check_role(allowed_roles=["router"])
    def start_router_server(cls, model_config):
        wait_for_prefill_decode_exit(key=ACTIVE_TEST_CLASS, value=cls.__name__)
        logger.info("Starting router in thread...")

        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(launch_router, model_config)

        cls.process = future.result()

        health_check_url = f"{cls.base_url}/health"
        logger.info(f"Waiting for router to be ready at {health_check_url}")
        return cls._wait_server_ready(health_check_url)

    @classmethod
    def _wait_server_ready(cls, url, timeout=120):
        start = time.perf_counter()
        while True:
            if requests.get(url).status_code == 200:
                logger.info(f"Server {url} is ready")
                return True
            if time.perf_counter() - start > timeout:
                return False
            time.sleep(2)

    @classmethod
    @check_role(allowed_roles=["router"])
    def stop_router_server(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
                for _ in range(60):
                    if cls.process.poll() is not None:
                        logger.info("Process fully exited")
                        break
                    time.sleep(1)
                else:
                    logger.warning("Process did NOT exit in time")
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    @staticmethod
    def print_test_case_info(test_case):
        """打印测试用例信息"""
        value = test_case["value"]
        should_succeed = test_case["should_succeed"]
        description = test_case["description"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"测试: {description}")
        logger.info(f"参数值: '{value}'")
        logger.info(f"期望结果: {'启动成功' if should_succeed else '启动失败'}")
        logger.info("=" * 60)

    @check_role(allowed_roles=["router"])
    def validate_bucket_adjust_interval_secs(self, test_case):
        self.print_test_case_info(test_case)

        value = test_case["value"]
        should_succeed = test_case["should_succeed"]

        self.model_config = create_model_config_with_param(value)

        is_running = self.start_router_server(self.model_config)
        self.assert_result(value, is_running, should_succeed)

    @check_role(allowed_roles=["router"])
    def test_bucket_adjust_interval_secs_validation(self):
        """测试 --bucket-adjust-interval-secs 参数的合法性验证"""
        logger.info("=== 开始测试 --bucket-adjust-interval-secs 参数验证 ===\n")
        for test_case in self.test_cases:
            self.validate_bucket_adjust_interval_secs(test_case)
            self.stop_router_server()

    def assert_result(self, value, success, should_succeed):
        """断言测试结果"""
        if should_succeed:
            self.assertTrue(success, msg=f"参数 '{value}' 应该启动成功，但实际失败")
            logger.info(f"✓ 验证通过: 服务启动成功")
        else:
            self.assertFalse(success, msg=f"参数 '{value}' 应该启动失败，但实际成功")
            logger.info(f"✓ 验证通过: 服务启动失败（预期行为）")


if __name__ == "__main__":
    unittest.main()
