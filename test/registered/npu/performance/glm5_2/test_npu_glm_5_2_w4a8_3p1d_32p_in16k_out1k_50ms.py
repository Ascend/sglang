import logging
import subprocess
import unittest

logger = logging.getLogger(__name__)

from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    CONFIGMAP_NAME,
    NAMESPACE,
    NIC_NAME,
    query_configmap,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_2_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_2_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    # "ZBAL_HCCL_OP": "send,recv",
    # "HCCL_BUFFSIZE": "128",
    # "SGLANG_ZBAL_LOCAL_MEM_SIZE": "61184",
    # "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
    "DEEP_USE_ALLTOALL_MODE": "1",
}

GLM_5_2_PD_SEP_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_BUFFSIZE": "300",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "40",
    "TASK_QUEUE_ENABLE": "0",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--trust-remote-code",
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.72,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    16,
    "--served-model-name",
    "glm-5",
    "--chunked-prefill-size",
    4096,
    "--max-prefill-tokens",
    180000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--disable-shared-experts-fusion",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
    # "--speculative-draft-model-quantization",
    # "unquant",
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--attn-cp-size",
    4,
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--pp-size",
    4,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
]

GLM_5_2_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--trust-remote-code",
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--ep-size",
    16,
    "--mem-fraction-static",
    0.97,
    "--max-running-requests",
    128,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--served-model-name",
    "glm-5",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--cuda-graph-max-bs-decode",
    8,
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--context-length",
    180000,
    "--tokenizer-worker-num",
    8,
    "--disable-shared-experts-fusion",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
    # "--speculative-draft-model-quantization",
    # "unquant",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]

GLM_5_2_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_2_W4A8_MODEL_PATH,
    "prefill_args": GLM_5_2_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_2_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_2_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_2_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUGLM_5_2_W4A8_PD_SEP_In16k_Out1k(TestNpuPerfMultiNodePdSepTestCaseBase):
    """Test NPU performance for GLM-5.2-w4a8 PD separation 3p1d in16k out1k"""

    model_config = GLM_5_2_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    max_concurrency = 128
    num_prompts = 512
    input_len = 16000
    output_len = 1000
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 1694

    def _flush_prefill_cache(self):
        """Flush cache on all prefill nodes via /flush_cache endpoint."""
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            logger.warning("ConfigMap is empty, skip flush_cache")
            return
        for pod_name, pod_ip in configmap.data.items():
            if "prefill" in pod_name:
                url = f"http://{pod_ip}:9999/flush_cache"
                try:
                    subprocess.run(
                        [
                            "curl",
                            "--location",
                            url,
                            "--header",
                            "Content-Type: application/json",
                        ],
                        capture_output=True,
                        timeout=30,
                        check=True,
                    )
                    logger.info(f"flush_cache succeeded on {pod_name} ({pod_ip})")
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"flush_cache failed on {pod_name} ({pod_ip}): "
                        f"exit_code={e.returncode}, stderr={e.stderr.decode() if e.stderr else 'N/A'}"
                    )
                except subprocess.TimeoutExpired as e:
                    logger.warning(
                        f"flush_cache timeout on {pod_name} ({pod_ip}): {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"flush_cache unexpected error on {pod_name} ({pod_ip}): {e}"
                    )

    def run_throughput(self):
        """Override: flush prefill cache and retry on first failure."""
        try:
            return TestNpuPerfMultiNodePdSepTestCaseBase.run_throughput(self)
        except (AssertionError, Exception):
            self._flush_prefill_cache()
            return TestNpuPerfMultiNodePdSepTestCaseBase.run_throughput(self)

    def test_npu_glm_5_2_w4a8_pd_sep_in16k_out1k(self):
        """Run NPU performance test for GLM-5.2-w4a8 PD separation 3p1d"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
