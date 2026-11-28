import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K3_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

# Deployment of the reference case (test_npu_kimi_k3_w4a8_32p_mix_in128k_out1_prefill.py):
# 4 nodes, tp-size 16 x pp-size 4.
KIMI_K3_W4A8_32P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "HCCL_BUFFSIZE": "2000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "2",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_FIAS_V2_BSND": "1",
    "SGLANG_PP_LAYER_PARTITION": "24,24,24,21",
}

KIMI_K3_W4A8_32P_PREFILL_ARGS = [
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--nnodes",
    4,
    "--tokenizer-path",
    KIMI_K3_W4A8_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    16,
    "--pp-size",
    4,
    "--pp-async-batch-depth",
    0,
    "--disable-overlap-schedule",
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--enable-shared-experts-attn-tp",
    "--enable-dense-mlp-attn-tp",
    "--mem-fraction-static",
    0.78,
    "--chunked-prefill-size",
    16384,
    "--max-total-tokens",
    250000,
    "--disable-radix-cache",
    "--cuda-graph-backend-decode",
    "disabled",
    "--max-running-requests",
    8,
    "--reasoning-parser",
    "kimi_k3",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-custom-all-reduce",
    "--watchdog-timeout",
    9000,
]

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_PREFILL_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3W4A832PMixIn128kOut1PrefillBaseline(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """Cold 131072 in / 1 out pure prefill over the reference deployment,
    benchmarked once with `python -m sglang.bench_serving` (random dataset, 10
    prompts, C1 with 1 warmup request) exactly like the deepseek_v4_flash case.

    This case only records the TTFT, TPOT and throughput data. Its pair
    test_npu_kimi_k3_w4a8_32p_mix_in128k_out1_prefill_dynamic_chunking.py runs the
    same workload with --enable-dynamic-chunking; the two deployments are separate
    multi-node launches, so the TTFT comparison happens between the metrics the two
    cases record."""

    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_name = "random"
    input_len = 131072
    output_len = 1
    num_prompts = 10
    max_concurrency = 1
    warmup_requests = 1
    random_range_ratio = 1
    request_rate = float("inf")

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1_prefill_baseline(self):
        """Benchmark the reference config and record its metrics."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
