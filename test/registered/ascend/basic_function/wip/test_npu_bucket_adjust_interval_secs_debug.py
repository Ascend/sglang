import copy
import logging
import os
import re
import subprocess
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    SERVER_INITIALIZATION_DELAY,
    SERVICE_PORT,
    TestAscendMultiNodePdSepTestCaseBase,
    check_role,
    launch_router,
    wait_server_ready,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN, TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="multi nodes testcase",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

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
        "TRANSFORMERS_VERBOSITY": "error",
    },
    "router_envs": {
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
    config = copy.deepcopy(MODEL_CONFIG_BASE)
    config["router_args"].extend(
        [
            "--bucket-adjust-interval-secs",
            bucket_interval,
        ]
    )
    return config


class TestBucketAdjustIntervalSecsValidation(TestAscendPerfMultiNodePdSepTestCaseBase):
    """测试 --bucket-adjust-interval-secs 参数的合法性验证"""
    model_config = create_model_config_with_param("5")

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


    def is_router_server_running(self, timeout=30):
        """检查router服务器是否正常运行，通过HTTP请求检测"""
        url = f"http://127.0.0.1:{self.port}/health"
        start_time = time.perf_counter()
        check_interval = 2

        while True:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Router server at {url} is ready!")
                    return True
                else:
                    print(f"Router server returned status code: {response.status_code}")
            except Exception as e:
                print(f"Router server not ready yet: {e}")

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > timeout:
                print(f"Router server failed to start within {timeout}s")
                return False
            time.sleep(check_interval)

    @staticmethod
    def print_test_case_info(test_case):
        """打印测试用例信息"""
        value = test_case["value"]
        should_succeed = test_case["should_succeed"]
        description = test_case["description"]
        print(f"\n{'=' * 60}")
        print(f"测试: {description}")
        print(f"参数值: '{value}'")
        print(f"期望结果: {'启动成功' if should_succeed else '启动失败'}")
        print("=" * 60)

    def _find_router_pid_by_port(self):
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.split("\n"):
            if f":{SERVICE_PORT}" in line:
                match = re.search(r"pid=(\d+)", line)
                if match:
                    return int(match.group(1))
        return None

    def _start_router_server(self):
        cls = self.__class__
        cls.sglang_thread = threading.Thread(
            target=launch_router, args=(cls.model_config,)
        )
        cls.sglang_thread.daemon = True
        cls.sglang_thread.start()

        health_check_url = f"{cls.base_url}/health"
        wait_server_ready(health_check_url)

        time.sleep(SERVER_INITIALIZATION_DELAY)

        cls._router_pid = self._find_router_pid_by_port()
        if cls._router_pid is None:
            raise RuntimeError(
                f"Failed to find router process PID on port {SERVICE_PORT}"
            )

    def _stop_router_server(self):
        cls = self.__class__
        router_pid = getattr(cls, "_router_pid", None)
        if router_pid is not None:
            try:
                kill_process_tree(router_pid)
            except Exception:
                pass
            cls._router_pid = None

        if cls.sglang_thread is not None:
            if cls.sglang_thread.is_alive():
                cls.stop_event.set()
                cls.sglang_thread.join(timeout=5)
            cls.sglang_thread = None

        time.sleep(5)

    @check_role(allowed_roles=["router"])
    def test_bucket_adjust_interval_secs_validation(self):
        is_running = self.is_router_server_running(timeout=60)
        self.assert_result("5", is_running, True)



if __name__ == "__main__":
    unittest.main()
