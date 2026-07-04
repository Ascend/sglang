import json
import math
import os
import re
import unittest

import requests

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    ROUND_ROBIN,
    TestAscendPerfMultiNodePdSepTestCaseBase,
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
PROMETHEUS_PORT = 29000

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
        "--prometheus-host",
        "0.0.0.0",
        "--prometheus-port",
        PROMETHEUS_PORT,
    ],
}


class TestNPUBalance(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 40
    max_concurrency = 30
    num_prompts = 120
    input_len = 300
    output_len = 20
    random_range_ratio = 1
    seed = 1
    tolerance_ratio = 0.01

    def get_router_metrics(self):
        """获取Router节点的metrics（使用prometheus端口29000）"""
        router_host = os.environ.get("POD_IP", "127.0.0.1")
        router_prometheus_port = PROMETHEUS_PORT
        print(f"Querying router metrics from: {router_host}:{router_prometheus_port}")

        try:
            response = requests.get(
                f"http://{router_host}:{router_prometheus_port}/metrics", timeout=30
            )
            if response.status_code == 200:
                print(
                    f"Successfully fetched router metrics (length: {len(response.text)})"
                )
                return json.loads(response.text)
            else:
                print(
                    f"Failed to get router metrics, status code: {response.status_code}"
                )
        except Exception as e:
            print(f"Error fetching router metrics: {e}")

        return None

    def parse_worker_requests(self, metrics):
        """解析各worker节点处理的请求数量

        Args:
            metrics (dict): Router返回的metrics字典

        Returns:
            tuple: (prefill_requests, decode_requests)
                prefill_requests: dict, key为worker地址，value为处理请求数
                decode_requests: dict, key为worker地址，value为处理请求数
        """

        prefill_requests = {}
        decode_requests = {}

        total_requests = metrics[
            'smg_http_requests_total{method="POST",path="/v1/chat/completions"}'
        ]

        for key, value in metrics.items():
            # 匹配 smg_worker_cb_outcomes_total{worker="http://xxx:8000",outcome="success"}
            match = re.search(
                r'smg_worker_cb_outcomes_total\{worker="([^"]+)",outcome="success"}',
                key,
            )
            if match:
                worker_url = match.group(1)
                if value == total_requests:
                    decode_requests[worker_url] = value
                else:
                    prefill_requests[worker_url] = value

        return prefill_requests, decode_requests

    def assert_load_balance(self, requests_after, requests_before, tolerance_ratio):
        """断言P节点处理请求负载均衡

        Args:
            requests_after (dict): 测试后各P节点处理请求数
            requests_before (dict, optional): 测试前各P节点处理请求数. Defaults to None.
        """
        if requests_before is not None:
            # 检查key是否一致
            after_keys = set(requests_after.keys())
            before_keys = set(requests_before.keys())

            if after_keys != before_keys:
                raise ValueError(
                    f"requests_after和requests_before的key不一致！"
                    f"\nafter_keys: {after_keys}"
                    f"\nbefore_keys: {before_keys}"
                )

            # 计算差值
            handled_request = {
                key: requests_after[key] - requests_before[key] for key in after_keys
            }
        else:
            handled_request = requests_after

        total = sum(handled_request.values())
        count = len(handled_request)
        avg = total / count if count > 0 else 0
        tolerance_abs = max(math.ceil(avg * tolerance_ratio), 1)

        print(f"  - P节点平均处理请求数: {avg:.0f}")
        print(f"  - 负载均衡容忍偏差: ±{tolerance_ratio * 100:.0f}%")
        print(f"  - 负载均衡容忍偏差绝对值: ±{tolerance_abs} 请求")

        max_deviation_abs = 0
        unbalanced_workers = []

        for worker, req_count in handled_request.items():
            deviation_abs = abs(req_count - avg)

            max_deviation_abs = max(max_deviation_abs, deviation_abs)

            if deviation_abs > tolerance_abs:
                unbalanced_workers.append((worker, req_count, deviation_abs))
            print(
                f"    - {worker}: {req_count:.0f} 请求, 绝对偏差: {deviation_abs:.1f} 请求"
            )

        try:
            assert (
                max_deviation_abs <= tolerance_abs
            ), f"P节点负载不均衡，最大绝对偏差{max_deviation_abs:.1f}请求超过容忍阈值{tolerance_abs}请求"
            print(
                f"  - ✓ 断言通过：P节点负载均衡（最大绝对偏差{max_deviation_abs:.1f}请求 ≤ 容忍阈值{tolerance_abs}请求）"
            )
        except AssertionError as e:
            print(f"  - ✗ 断言失败：{e}")
            if unbalanced_workers:
                print("    不均衡节点详情:")
                for worker, req_count, deviation_abs in unbalanced_workers:
                    print(
                        f"      - {worker}: {req_count:.0f} 请求, 绝对偏差: {deviation_abs:.1f} 请求"
                    )
            raise

    def test_throughput_with_prefill_stats(self):
        router_metrics_before = self.get_router_metrics()
        print(f"{router_metrics_before=}")
        prefill_requests_before, decode_requests_before = self.parse_worker_requests(
            router_metrics_before
        )
        self.run_throughput()
        router_metrics_after = self.get_router_metrics()
        print(f"{router_metrics_after=}")
        prefill_requests_after, decode_requests_after = self.parse_worker_requests(
            router_metrics_after
        )

        self.assert_load_balance(
            prefill_requests_after, prefill_requests_before, self.tolerance_ratio
        )


if __name__ == "__main__":
    unittest.main()
