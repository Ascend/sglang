import os
import unittest

import requests

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
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
        "HCCL_SOCKET_IFNAME": "enp23s0f3",
        "GLOO_SOCKET_IFNAME": "enp23s0f3",
    },
    "router_envs": {
        "ASCEND_MF_STORE_URL": "tcp://192.168.0.60:24667",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "prefill_args": [
        "--disaggregation-mode",
        "prefill",
        "--nnodes",
        3,
        "--tp",
        48,
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


class TestManualDeploy(TestAscendPerfMultiNodePdSepTestCaseBase):
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

    @staticmethod
    def query_configmap(configmap_name, namespace):
        """从Kubernetes ConfigMap获取节点IP信息"""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "configmap",
                    configmap_name,
                    "-n",
                    namespace,
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
        except Exception as e:
            print(f"Failed to query ConfigMap: {e}")
        return None

    @staticmethod
    def get_prefill_ips_from_configmap():
        """从ConfigMap自动获取所有P节点的IP地址"""
        if not CONFIGMAP_NAME or not NAMESPACE:
            print("Warning: KUBE_CONFIG_MAP or NAMESPACE environment variable not set")
            # 尝试从环境变量获取
            prefill_ips = os.environ.get("PREFILL_IPS", "")
            if prefill_ips:
                return [ip.strip() for ip in prefill_ips.split(",") if ip.strip()]
            return []

        configmap = TestManualDeploy.query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or "data" not in configmap:
            print("Warning: ConfigMap data not available")
            return []

        prefill_ips = []
        for pod_name, pod_ip in configmap["data"].items():
            if pod_name.lower().endswith("prefill-0") or "prefill" in pod_name.lower():
                prefill_ips.append(pod_ip)
                print(f"Found P node: {pod_name} = {pod_ip}")
        return prefill_ips

    @staticmethod
    def get_prefill_metrics(prefill_ip, port=8000):
        """获取单个P节点的统计信息"""
        try:
            response = requests.get(f"http://{prefill_ip}:{port}/metrics", timeout=10)
            if response.status_code == 200:
                return TestManualDeploy.parse_metrics(response.text)
        except Exception as e:
            print(f"Failed to get metrics from {prefill_ip}:{port}: {e}")
        return None

    @staticmethod
    def parse_metrics(metrics_text):
        """解析Prometheus格式的metrics"""
        parsed = {}
        for line in metrics_text.split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                try:
                    value = float(parts[1])
                    parsed[name] = value
                except ValueError:
                    pass
        return parsed

    def collect_prefill_metrics(self):
        """收集所有P节点的metrics并打印统计"""
        prefill_ips = self.get_prefill_ips_from_configmap()
        if not prefill_ips:
            print("Warning: No P nodes found, trying localhost")
            prefill_ips = ["127.0.0.1"]

        metrics = {}
        for ip in prefill_ips:
            m = self.get_prefill_metrics(ip)
            if m:
                metrics[ip] = m
                print(f"\nP节点 {ip} 统计:")
                print(f"  - 请求数: {m.get('sglang_prefill_requests_total', 0):.0f}")
                print(f"  - Tokens数: {m.get('sglang_prefill_tokens_total', 0):.0f}")
                print(
                    f"  - 平均延迟: {m.get('sglang_prefill_latency_seconds', 0):.4f}s"
                )
        return metrics

    def test_throughput_with_prefill_stats(self):
        """测试吞吐量并统计每个P节点的请求数和tokens数"""
        # 获取P节点IP
        prefill_ips = self.get_prefill_ips_from_configmap()
        if not prefill_ips:
            print("Warning: No P nodes found from ConfigMap, using fallback")
            prefill_ips = ["127.0.0.1"]

        print("=== 测试开始前的P节点统计 ===")
        initial_metrics = self.collect_prefill_metrics()

        # 运行主测试
        print("\n=== 开始运行吞吐量测试 ===")
        self.run_throughput()

        print("\n=== 测试结束后的P节点统计 ===")
        final_metrics = self.collect_prefill_metrics()

        # 计算增量
        print("\n=== 测试期间的增量统计 ===")
        total_requests = 0
        total_tokens = 0
        for ip in prefill_ips:
            initial = initial_metrics.get(ip, {})
            final = final_metrics.get(ip, {})
            req_diff = final.get('sglang_prefill_requests_total', 0) - initial.get(
                "sglang_prefill_requests_total", 0
            )
            tok_diff = final.get('sglang_prefill_tokens_total', 0) - initial.get(
                "sglang_prefill_tokens_total", 0
            )
            total_requests += req_diff
            total_tokens += tok_diff
            print(f"\nP节点 {ip}:")
            print(f"  - 处理请求数: {req_diff:.0f}")
            print(f"  - 处理Tokens数: {tok_diff:.0f}")

        print(f"\n=== 总计 ===")
        print(
            f"所有P节点共处理: {total_requests:.0f} 个请求, {total_tokens:.0f} 个tokens"
        )

    def test_throughput(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
