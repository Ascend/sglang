import copy
import time
import unittest
import threading

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.e2e.test_npu_multi_node_utils import check_role, LOCAL_TIMEOUT
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    SERVICE_PORT,
    check_role,
    launch_pd_mix_node,
    launch_pd_separation_node,
    launch_router,
    wait_server_ready,
)

register_npu_ci(
    est_time=3600, suite="", nightly=True, disabled="multi modes test cases"
)

MODEL_CONFIG = {
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
        "--bucket-adjust-interval-secs",
        5,
    ],
}


def check_server_ready(url, timeout=LOCAL_TIMEOUT):
    """Wait for the server to be ready.

    Args:
        url (str): Server URL to check.
        timeout (int): Timeout in seconds.

    Raises:
        RuntimeError: If server fails to start within timeout.
    """
    start_time = time.perf_counter()
    check_interval = 10

    while True:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return True
        except Exception as e:
            # logger.error(f"Server {url} request error: {e}, retrying...")
            pass

        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > timeout:
            return False
        time.sleep(check_interval)

def create_model_config_with_param(bucket_interval):
    """创建带有指定 bucket-adjust-interval-secs 参数的配置"""
    config = copy.deepcopy(MODEL_CONFIG)
    config["router_args"].extend(
        [
            "--bucket-adjust-interval-secs",
            bucket_interval,
        ]
    )
    return config


class TestNPUBucketAdjustIntervalSecsConcurrency(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """Testcase：After configuring the --prefill-policy parameter as bucket for the PD classification scenario and
    setting the relevant parameters, verify that the router service can stably support 2048 concurrent requests

    [Test Category] Parameter
    [Test Target] --prefill-policy; --bucket-adjust-interval-secs; --balance-abs-threshold; --balance-rel-threshold
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 40
    max_concurrency = 2048
    num_prompts = 2048
    input_len = 300
    output_len = 20
    random_range_ratio = 1

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

    @classmethod
    @check_role(allowed_roles=["router"])
    def _start_router_server(cls):
        sglang_thread = threading.Thread(target=launch_router, args=(cls.model_config,))
        sglang_thread.daemon = True
        sglang_thread.start()


    @check_role(allowed_roles=["router"])
    def test_throughput(self):
        health_check_url = f"{self.base_url}/health"
        if_ready = check_server_ready(health_check_url)
        print(f"{if_ready=}")

        self._stop_router_server()

        if_ready = check_server_ready(health_check_url)
        print(f"{if_ready=}")

        self.__class__.model_config = create_model_config_with_param("0")

        try:
            self._start_router_server()
            is_running = check_server_ready(health_check_url)
            print(f"{is_running=}")
        finally:
            self._stop_router_server()


if __name__ == "__main__":
    unittest.main()
