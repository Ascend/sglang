import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    EVALSCOPE,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH, TestNpuPerformanceTestCaseBase, AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT, AISBENCHMARK,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=7200,
    suite="nightly-acc-4-npu-a5-",
    nightly=True,
)

DEEPSEEK_V4_FLASH_FP8_4P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "1",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    # HCCL deepep
    "DEEPEP_HCCL_BUFFSIZE": "2560",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    # DeepEP long-sequence normal combine (ROUND * TOKENS >= chunked_prefill_size / tp * dp)
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    # dsv4
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    # performance
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "ASCEND_LAUNCH_BLOCKING": "1",
    # Dspark
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    # "SGLANG_DSPARK_A5_EXTRA_OPS_SO": "/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so",
    # "ASCEND_CUSTOM_OPP_PATH": "/usr/local/Ascend/extra-ops/vendors/custom_transformer/:$ASCEND_CUSTOM_OPP_PATH",
    # "LD_LIBRARY_PATH": "/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH",
}

DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    4,
    "--trust-remote-code",
    "--attention-backend",
    "dsv4",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.73,
    "--max-running-requests",
    32,
    "--chunked-prefill-size",
    131072,
    "--max-prefill-tokens",
    131072,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    "--kv-cache-dtype",
    "auto",
    "--enable-dp-lm-head",
    "--disable-radix-cache",
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--reasoning-parser",
    "deepseek-v4",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    "--speculative-dspark-block-size",
    5,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "fp8",
]

DEEPSEEK_V4_FLASH_FP8_4P_GENERATION_CONFIG_MAX = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}
    },
}


class TestNPUDeepSeekV4FlashFP84PGPQA(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for DeepSeek-V4-Flash FP8 4p DSPARK GPQA."""

    benchmark_tool = EVALSCOPE
    model = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_FP8_4P_ENVS
    accuracy = 0.874
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = DEEPSEEK_V4_FLASH_FP8_4P_GENERATION_CONFIG_MAX
    eval_batch_size = 128
    stream = True
    timeout = 6000
    seed = 1

    def test_npu_deepseek_v4_flash_fp8_4p_gpqa(self):
        """Run NPU accuracy test for DeepSeek-V4-Flash FP8 4p DSPARK GPQA."""
        self.run_accuracy()

class TestNPUDeepSeekV4FlashDSPARK(TestNpuPerformanceTestCaseBase):
    """Test DSPARK for Deepseek-V4 in the 950 environment, acceptance rate greater than 0.5"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    # dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_FP8_4P_ENVS
    dataset_name = "random"
    input_len = 8000
    output_len = 1000
    num_prompts = 160
    max_concurrency = 160
    random_range_ratio = 1
    warmup_requests = 32
    request_rate = float("inf")
    max_attempts = 3
    accept_rate = 0.5

    # @classmethod
    # def setUpClass(cls):
    #     """Launch server via `python3 -m sglang.launch_server` instead of `sglang serve`."""
    #     cls._setup_per_case_output()
    #     cls.base_url = DEFAULT_URL_FOR_TEST
    #     env = os.environ.copy()
    #     if cls.envs:
    #         env.update(cls.envs)
    #
    #     _, host, port = cls.base_url.split(":")
    #     command = [
    #         "python3",
    #         "-m",
    #         "sglang.launch_server",
    #         "--model-path",
    #         cls.model,
    #         *[str(x) for x in cls.other_args],
    #         "--host",
    #         host[2:],
    #         "--port",
    #         port,
    #     ]
    #     cls.process = subprocess.Popen(command, env=env)
    #     wait_for_server(cls.base_url, timeout=cls.timeout, process=cls.process)

    def test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms(self):
        """Run NPU performance test for DeepSeek-V4-Flash W8A8 8p in8k out1k."""
        self.run_throughput()

class TestNPUDeepSeekV4FlashAisBench(TestNpuPerformanceTestCaseBase):
    """Test DSPARK for Deepseek-V4 in the 950 environment, acceptance rate greater than 0.5"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    # dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_FP8_4P_ENVS
    dataset_name = "random"
    input_len = 8000
    output_len = 1000
    num_prompts = 10
    max_concurrency = 160
    random_range_ratio = 1
    warmup_requests = 32
    request_rate = float("inf")
    max_attempts = 3
    accept_rate = 0.5

    def test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms(self):
        """Run NPU performance test for DeepSeek-V4-Flash W8A8 8p in8k out1k."""
        self.run_throughput()

class TestNPUKimiK3W4A832PMixIn128kOut1kDecodePrefix(
    TestNpuPerformanceTestCaseBase
):
    """Pure decode low latency: 128k shared prefix in / 1024 out, C12,
    mean TPOT <= 30 ms (dev measured 23.2-25.9 ms at C12 on this shape)."""

    model_config = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
    # AISBENCH RETRY of the pure-decode case (user-requested): identical
    # GSM8K natural-text rows through the PLAIN aisbench perf path - no
    # prefix-tool chain involved. Proven fallback if this fails: revert to
    # benchmark_tool = BENCHSERVING + dataset_name = "gsm8k-shared-prefix"
    # (run 41001026946, Mean TPOT 22.10 ms SUCCESS).
    benchmark_tool = AISBENCHMARK
    dataset_type = "gsm8k-shared-prefix"
    dataset_path = None
    input_len = 131072
    output_len = 1024
    num_prompts = 90
    max_concurrency = 60
    random_range_ratio = 1
    request_rate = 0.9
    dp = 16
    # repeat_rate is GSP-only; the gsm8k-shared-prefix generator fixes the
    # prefix at ~99% of input_len and gives each row its own short question.
    repeat_rate = None
    # One warmup request per DP rank primes the shared prefix everywhere
    # before the measured pass (bench_serving caps warmup output at 32).
    pop_sglang_is_in_ci_for_gsp = True
    # Greedy decoding: with the default temperature 1.0 the DSPARK draft
    # accept length collapsed to ~1.6 tokens (dev measured 3.94) and TPOT
    # landed at 63.7 ms vs the dev 25.9 ms. Dev's perf passes always pin
    # temperature to 0.
    temperature = 0
    tpot = 30.0

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1k_decode_prefix(self):
        """Run the 128k pure-decode low-latency TPOT acceptance test."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
