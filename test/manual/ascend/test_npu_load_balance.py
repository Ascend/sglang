import math
import os
import unittest

import requests

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
        # "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8192",
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
        # "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "8192",
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
        # "HCCL_SOCKET_IFNAME": "enp23s0f3",
        # "GLOO_SOCKET_IFNAME": "enp23s0f3",
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
        # "--max-dispatch-tokens",
        # 1024,
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
        "29000",
    ],
}


class TestManualDeploy(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    request_rate = 40
    max_concurrency = 30
    num_prompts = 30
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
        router_prefill_ips = TestManualDeploy.get_prefill_ips_from_router()
        if router_prefill_ips:
            return router_prefill_ips

        if CONFIGMAP_NAME and NAMESPACE:
            configmap = TestManualDeploy.query_configmap(CONFIGMAP_NAME, NAMESPACE)
            if configmap and "data" in configmap:
                prefill_ips = []
                for pod_name, pod_ip in configmap["data"].items():
                    if pod_name.lower().endswith("prefill-0") or "prefill" in pod_name.lower():
                        prefill_ips.append(pod_ip)
                        print(f"Found P node from ConfigMap: {pod_name} = {pod_ip}")
                if prefill_ips:
                    return prefill_ips

        return []

    @staticmethod
    def get_prefill_ips_from_router():
        """从Router节点获取已注册的P节点IP地址"""
        from sglang.test.ascend.e2e.test_npu_multi_node_utils import SERVICE_PORT
        
        # 获取router地址
        router_host = os.environ.get("POD_IP", "127.0.0.1")
        router_port = SERVICE_PORT
        
        # 如果当前节点不是router，尝试从环境变量获取router地址
        hostname = os.environ.get("HOSTNAME", "")
        if not hostname or "router" not in hostname.lower():
            # 尝试从其他环境变量获取router地址
            router_host = os.environ.get("ROUTER_IP", router_host)
            router_port = int(os.environ.get("ROUTER_PORT", router_port))

        print(f"Trying to get P nodes from router: {router_host}:{router_port}")
        
        # 方法1: 从router的health接口获取
        health_url = f"http://{router_host}:{router_port}/health"
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                # 解析health响应，尝试提取prefill节点信息
                health_data = response.text
                print(f"Router health response received (length: {len(health_data)})")
                # 尝试从响应中提取IP地址
                prefill_ips = TestManualDeploy.extract_ips_from_router_response(health_data)
                if prefill_ips:
                    print(f"Found P nodes from router health: {prefill_ips}")
                    return prefill_ips
        except Exception as e:
            print(f"Failed to query router health: {e}")

        # 方法2: 从router的metrics接口获取
        metrics_url = f"http://{router_host}:{router_port}/metrics"
        try:
            response = requests.get(metrics_url, timeout=10)
            if response.status_code == 200:
                metrics_data = response.text
                print(f"Router metrics response received (length: {len(metrics_data)})")
                prefill_ips = TestManualDeploy.extract_ips_from_router_response(metrics_data)
                if prefill_ips:
                    print(f"Found P nodes from router metrics: {prefill_ips}")
                    return prefill_ips
        except Exception as e:
            print(f"Failed to query router metrics: {e}")

        # 方法3: 从router的配置接口获取（如果存在）
        config_urls = [
            f"http://{router_host}:{router_port}/config",
            f"http://{router_host}:{router_port}/api/v1/config"
        ]
        for config_url in config_urls:
            try:
                response = requests.get(config_url, timeout=10)
                if response.status_code == 200:
                    config_data = response.text
                    prefill_ips = TestManualDeploy.extract_ips_from_router_response(config_data)
                    if prefill_ips:
                        print(f"Found P nodes from router config: {prefill_ips}")
                        return prefill_ips
            except Exception as e:
                print(f"Failed to query router config ({config_url}): {e}")

        print("No P nodes found from router")
        return []

    @staticmethod
    def extract_ips_from_router_response(response_text):
        """从Router响应中提取P节点IP地址"""
        import re
        
        prefill_ips = []
        
        # 模式1: 匹配完整的URL格式 http://ip:port
        url_pattern = r'http://(\d+\.\d+\.\d+\.\d+):8000'
        matches = re.findall(url_pattern, response_text)
        if matches:
            prefill_ips.extend(matches)
        
        # 模式2: 匹配IP地址后跟prefill相关的字符串
        ip_pattern = r'(\d+\.\d+\.\d+\.\d+).*?prefill'
        matches = re.findall(ip_pattern, response_text, re.IGNORECASE)
        if matches:
            prefill_ips.extend(matches)
        
        # 模式3: 匹配IP地址列表格式
        ip_list_pattern = r'prefill.*?\[([^\]]+)\]'
        list_match = re.search(ip_list_pattern, response_text, re.IGNORECASE | re.DOTALL)
        if list_match:
            ip_content = list_match.group(1)
            individual_ips = re.findall(r'\d+\.\d+\.\d+\.\d+', ip_content)
            prefill_ips.extend(individual_ips)
        
        # 去重并验证IP格式
        valid_ips = []
        for ip in set(prefill_ips):
            if TestManualDeploy.is_valid_ip(ip):
                valid_ips.append(ip)
        
        return valid_ips

    @staticmethod
    def is_valid_ip(ip):
        """验证IP地址格式是否有效"""
        import re
        ip_pattern = r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$'
        match = re.match(ip_pattern, ip)
        if match:
            # 验证每个部分的范围
            parts = [int(p) for p in match.groups()]
            return all(0 <= p <= 255 for p in parts)
        return False

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

    def get_router_metrics(self):
        """获取Router节点的metrics（使用prometheus端口29000）"""
        from sglang.test.ascend.e2e.test_npu_multi_node_utils import SERVICE_PORT
        
        # 获取router地址
        router_host = os.environ.get("POD_IP", "127.0.0.1")
        # router的prometheus端口（从配置中读取）
        router_prometheus_port = 29000
        
        # 如果当前节点不是router，尝试从环境变量获取router地址
        hostname = os.environ.get("HOSTNAME", "")
        if not hostname or "router" not in hostname.lower():
            router_host = os.environ.get("ROUTER_IP", router_host)
        
        print(f"Querying router metrics from: {router_host}:{router_prometheus_port}")
        
        try:
            response = requests.get(f"http://{router_host}:{router_prometheus_port}/metrics", timeout=30)
            if response.status_code == 200:
                print(f"Successfully fetched router metrics (length: {len(response.text)})")
                return self.parse_metrics(response.text)
            else:
                print(f"Failed to get router metrics, status code: {response.status_code}")
        except Exception as e:
            print(f"Error fetching router metrics: {e}")
        
        return None

    def print_router_metrics(self, metrics):
        """打印Router节点的关键metrics"""
        print("\nRouter节点统计:")
        print(f"  - metric返回值数量: {len(metrics)}")
        print(f"  - metric:\n{metrics}")
        
        # 打印一些关键的router指标（如果存在）
        for key, value in metrics.items():
            if any(kw in key.lower() for kw in ["router_requests", "http_requests_total{method=\"post\""]):
                print(f"  - {key}: {value}")

    def parse_worker_requests(self, metrics):
        """解析各worker节点处理的请求数量
        
        Args:
            metrics (dict): Router返回的metrics字典
        
        Returns:
            tuple: (prefill_requests, decode_requests)
                prefill_requests: dict, key为worker地址，value为处理请求数
                decode_requests: dict, key为worker地址，value为处理请求数
        """
        import re
        
        prefill_requests = {}
        decode_requests = {}
        
        for key, value in metrics.items():
            # 匹配 smg_worker_cb_outcomes_total{worker="http://xxx:8000",outcome="success"}
            match = re.search(r'smg_worker_cb_outcomes_total\{worker="([^"]+)",outcome="success"\}', key)
            if match:
                worker_url = match.group(1)
                # 判断是prefill还是decode节点
                # 从smg_worker_health指标判断
                for health_key in metrics.keys():
                    if worker_url in health_key and "worker_type=\"prefill\"" in health_key:
                        prefill_requests[worker_url] = value
                        break
                    elif worker_url in health_key and "worker_type=\"decode\"" in health_key:
                        decode_requests[worker_url] = value
                        break
        
        # 如果通过health指标无法判断，则通过smg_worker_selection_total辅助判断
        if not prefill_requests and not decode_requests:
            # 直接统计所有success的请求
            for key, value in metrics.items():
                match = re.search(r'smg_worker_cb_outcomes_total\{worker="([^"]+)",outcome="success"\}', key)
                if match:
                    worker_url = match.group(1)
                    # 默认假设为prefill节点（如果无法区分）
                    prefill_requests[worker_url] = value
        
        return prefill_requests, decode_requests

    def assert_prefill_decode_equal(self, p_total, d_total):
        """断言P节点和D节点处理请求总数相等"""
        print(f"  - 验证P节点总计({p_total:.0f}) == D节点总计({d_total:.0f})")
        try:
            assert abs(p_total - d_total) <= 1, f"P节点处理请求数({p_total})与D节点({d_total})不相等"
            print("  - ✓ 断言通过：P节点和D节点处理请求总数相等")
        except AssertionError as e:
            print(f"  - ✗ 断言失败：{e}")
            raise

    def assert_prefill_load_balance(self, prefill_requests, tolerance_ratio=0.1):
        """断言P节点处理请求负载均衡
        
        Args:
            prefill_requests (dict): 各P节点处理请求数
            tolerance_ratio (float): 容忍偏差比例，默认为10%
        """
        if not prefill_requests:
            return
        
        total = sum(prefill_requests.values())
        count = len(prefill_requests)
        avg = total / count if count > 0 else 0
        tolerance_abs = max(math.ceil(avg * tolerance_ratio), 1)
        
        
        
        print(f"  - P节点平均处理请求数: {avg:.0f}")
        print(f"  - 负载均衡容忍偏差: ±{tolerance_ratio*100:.0f}%")
        print(f"  - 负载均衡容忍偏差绝对值: ±{tolerance_abs} 请求")
        
        max_deviation_abs = 0
        unbalanced_workers = []
        
        for worker, req_count in prefill_requests.items():
            deviation_abs = abs(req_count - avg)
            
            max_deviation_abs = max(max_deviation_abs, deviation_abs)
            
            if deviation_abs > tolerance_abs:
                unbalanced_workers.append((worker, req_count, deviation_abs))
            print(f"    - {worker}: {req_count:.0f} 请求, 绝对偏差: {deviation_abs:.1f} 请求")
        
        try:
            assert max_deviation_abs <= tolerance_abs, \
                f"P节点负载不均衡，最大绝对偏差{max_deviation_abs:.1f}请求超过容忍阈值{tolerance_abs}请求"
            print(f"  - ✓ 断言通过：P节点负载均衡（最大绝对偏差{max_deviation_abs:.1f}请求 ≤ 容忍阈值{tolerance_abs}请求）")
        except AssertionError as e:
            print(f"  - ✗ 断言失败：{e}")
            if unbalanced_workers:
                print("    不均衡节点详情:")
                for worker, req_count, deviation_abs in unbalanced_workers:
                    print(f"      - {worker}: {req_count:.0f} 请求, 绝对偏差: {deviation_abs:.1f} 请求")
            raise

    def collect_prefill_metrics(self, prefill_ips=None):
        """收集所有P节点的metrics并打印统计
        
        Args:
            prefill_ips (list, optional): P节点IP列表. 如果为None，会自动获取.
        
        Returns:
            dict: 各P节点的metrics字典
        """
        # 如果未传入IP列表，自动获取
        if prefill_ips is None:
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
                print(f"  - metric返回值: {m}")
                print(f"  - 请求数: {m.get('sglang_prefill_requests_total', 0):.0f}")
                print(f"  - Tokens数: {m.get('sglang_prefill_tokens_total', 0):.0f}")
                print(
                    f"  - 平均延迟: {m.get('sglang_prefill_latency_seconds', 0):.4f}s"
                )
        return metrics

    def test_throughput_with_prefill_stats(self):
        # """测试吞吐量并统计每个P节点的请求数和tokens数"""
        # # 获取P节点IP（一次性获取，复用）
        # prefill_ips = self.get_prefill_ips_from_configmap()
        # print(f"发现P节点IP: {prefill_ips}")

        # print("=== 测试开始前的P节点统计 ===")
        # initial_metrics = self.collect_prefill_metrics(prefill_ips)

        # # 运行主测试
        # print("\n=== 开始运行吞吐量测试 ===")
        self.run_throughput()

        # 获取Router节点的metrics
        print("\n=== 获取Router节点的Metrics ===")
        router_metrics = self.get_router_metrics()
        # if router_metrics:
        #     self.print_router_metrics(router_metrics)
        prefill_requests, decode_requests = self.parse_worker_requests(router_metrics)
        print(f"  - P节点处理请求数: {prefill_requests}")
        print(f"  - D节点处理请求数: {decode_requests}")
        self.assert_prefill_load_balance(prefill_requests)


        print("\n=== 测试结束后的P节点统计 ===")
        final_metrics = self.collect_prefill_metrics(prefill_ips)

        # 计算增量
        print("\n=== 测试期间的增量统计 ===")
        total_requests = 0
        total_tokens = 0
        for ip in prefill_ips:
            initial = initial_metrics.get(ip, {})
            final = final_metrics.get(ip, {})
            req_diff = final.get("sglang_prefill_requests_total", 0) - initial.get(
                "sglang_prefill_requests_total", 0
            )
            tok_diff = final.get("sglang_prefill_tokens_total", 0) - initial.get(
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


if __name__ == "__main__":
    unittest.main()
