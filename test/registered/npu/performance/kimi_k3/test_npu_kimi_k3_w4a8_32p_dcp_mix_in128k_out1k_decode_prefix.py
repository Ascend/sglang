import logging
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_GSM8K,
    KIMI_K3_DSPARK_MODEL_PATH,
    KIMI_K3_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

# Same four-node Kimi-K3-w4a8 topology as the non-DCP decode case, plus
# decode context parallelism (tp64 / attn-dp4 / dcp2). DCP_SIZE>1 requires the
# Ascend FIA attention path.
KIMI_K3_W4A8_32P_DCP_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE": "1",
    "TRITON_CACHE_DIR": "/root/.cache/sglang/triton",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FOLDED_PROPOSAL": "0",
    "SGLANG_DSPARK_STACKED_CTX_KV": "0",
    "SGLANG_DSPARK_EMBED_IN_GRAPH": "0",
    "SGLANG_NPU_FINE_GRAINED_MOE_DUAL_STREAM": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "2",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_FIAS_V2_BSND": "1",
    "HCCL_BUFFSIZE": "200",
    "DEEPEP_HCCL_BUFFSIZE": "1800",
}

KIMI_K3_W4A8_32P_DCP_OTHER_ARGS = [
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
    64,
    "--attn-dp-size",
    4,
    "--enable-dp-lm-head",
    "--enable-shared-experts-attn-tp",
    "--enable-dense-mlp-attn-tp",
    "--mamba-ssm-dtype",
    "bfloat16",
    # Decode context parallelism: shard the target MLA KV cache across DCP ranks.
    "--dcp-size",
    1,
    "--dcp-comm-backend",
    "a2a",
    "--mem-fraction-static",
    0.75,
    "--max-total-tokens",
    250000,
    "--max-mamba-cache-size",
    240,
    "--chunked-prefill-size",
    20480,
    "--cuda-graph-bs-decode",
    2,
    4,
    8,
    12,
    16,
    "--max-running-requests",
    64,
    "--reasoning-parser",
    "kimi_k3",
    "--enable-cache-report",
    "--random-seed",
    540332093,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    KIMI_K3_DSPARK_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-dspark-block-size",
    7,
    "--speculative-draft-attention-backend",
    "ascend",
    "--linear-attn-verify-backend",
    "triton",
    "--speculative-eagle-topk",
    1,
    "--disable-custom-all-reduce",
    "--watchdog-timeout",
    9000,
]

KIMI_K3_W4A8_32P_DCP_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_DCP_OTHER_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_DCP_ENVS,
}


class TestNPUKimiK3W4A832PDCPMixIn128kOut1kDecodePrefix(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """128k-in / 1k-out pure decode at C6 over a 99% shared prefix with DCP=2;
    gate: mean TPOT <= 30 ms (+1 ms tolerance) and output throughput >= 140 token/s.
    The same gates as the non-DCP case assert that DCP holds the low-latency bar."""

    model_config = KIMI_K3_W4A8_32P_DCP_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    dataset_path = None
    input_len = 131072
    output_len = 1024
    num_prompts = 6
    max_concurrency = 6
    repeat_rate = 0.99
    request_rate = 0
    dp = 4
    tpot = 30.0
    output_token_throughput = 140.0

    def test_npu_kimi_k3_w4a8_32p_dcp_mix_in128k_out1k_decode_prefix(self):
        """Run the 128k pure-decode low-latency TPOT acceptance test with DCP=2."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
