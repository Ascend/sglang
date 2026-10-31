import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

# NOTE (known blocker, verify before running): in the current codebase
# `--enable-dsa-cache-layer-split` is rejected for DeepseekV4ForCausalLM by
# the generic arch check in server_args.py (is_deepseek_dsa only covers
# V3/V32/GLM/Longcat archs). DeepSeek-V4 has its own prefill-CP support
# (validate_deepseek_v4_cp: interleave-only, dp_size == 1, tp_size <= 8).
# Make sure the deployed sglang version extends the layer-split arch check
# to DeepseekV4 before enabling this testcase.

# Common environment variables shared by prefill/decode nodes, ported from
# scripts_shell/pd/pro_1p1d(2+2)/dsv4_pro_pd.sh.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
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
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

# Prefill node environment variables for the LayerSplit deployment.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_PREFILL_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    # memory fabric for PD KV transfer
    "MF_HYBM_USE_VMM_SEGMENT": "1",
    "ASCEND_MF_TRANSFER_PROTOCOL": "device_urma",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
}

# Decode node environment variables for the LayerSplit deployment.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_DECODE_ENVS = {
    **DEEPSEEK_V4_PRO_W4A8_PD_SEP_COMMON_ENVS,
    "DEEPEP_HCCL_BUFFSIZE": "900",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "30",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    # MTP (DSPARK)
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    # dspark correctness-first setup
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
}

# Prefill node (1 node x 8 NPUs, TP8, prefill-CP attn-cp 8) launch args.
# V4 prefill CP constraints (validate_deepseek_v4_cp): dp_size == 1 and
# tp_size <= 8, so attn-cp 8 forces a single-node TP8 prefill topology.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--disaggregation-bootstrap-port",
    8998,
    "--tp-size",
    8,
    "--nnodes",
    1,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.83,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    2048000,
    "--chunked-prefill-size",
    65536,
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--context-length",
    133120,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-cuda-graph",
    # LayerSplit (requirement DSV4_Pro_LayerSplit_0):
    # --enable-dsa-cache-layer-split --attn-cp-size 8 --cp-strategy interleave
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--attn-cp-size",
    8,
    "--enable-dsa-cache-layer-split",
]

# Decode node (2 nodes x 8 NPUs, TP16 DP4) launch arguments.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    16,
    "--nnodes",
    2,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    256,
    "--mem-fraction-static",
    0.86,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    2048000,
    "--chunked-prefill-size",
    16384,
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--context-length",
    133120,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    "--tokenizer-worker-num",
    8,
    # DSPARK speculative decoding with the bundled draft weights.
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
]

# Model config for DSV4-Pro W4A8 LayerSplit PD-Sep deployment.
DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}

# Generation config for Think High mode (thinking=true, reasoning_effort=high).
DEEPSEEK_V4_PRO_W4A8_GENERATION_CONFIG_HIGH = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}


class TestNPUDeepSeekV4ProW4A8PDSepLayerSplitGPQAHigh(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for DSV4-Pro-0813 W4A8 PD-Sep prefill LayerSplit.

    Requirement: DSV4_Pro_LayerSplit_0 (1P1D PD separation, prefill with
    --enable-dsa-cache-layer-split --attn-cp-size 8 --cp-strategy interleave,
    GPQA accuracy within tolerance).
    """

    model_config = DEEPSEEK_V4_PRO_W4A8_PD_SEP_LS_MODEL_CONFIG
    # TODO: calibrate the baseline on the first successful run.
    accuracy = 0.85
    datasets = ["gpqa_diamond"]
    generation_config = DEEPSEEK_V4_PRO_W4A8_GENERATION_CONFIG_HIGH
    eval_batch_size = 64

    def test_npu_deepseek_v4_pro_w4a8_pd_sep_layersplit_gpqa_high(self):
        """Run NPU accuracy test for DSV4-Pro W4A8 PD-Sep LayerSplit GPQA."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
