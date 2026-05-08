import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_5_397B_W8A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-pd-sep-2-node",
    nightly=True,
)

PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}

DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "6",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--node-rank",
    0,
    "--mem-fraction-static",
    0.75,
    "--max-running-requests",
    4,
    "--chunked-prefill-size",
    32768,
    "--max-prefill-tokens",
    32768,
    "--max-total-tokens",
    150000,
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
]

DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--tp-size",
    16,
    "--nnodes",
    1,
    "--mem-fraction-static",
    0.75,
    "--max-total-tokens",
    100000,
    "--max-running-requests",
    4,
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
    12,
    16,
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
]

MODEL_CONFIG = {
    "model_path": QWEN3_5_397B_W8A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUQwen3_5_397B_W8A8_1P1D_16P_GPQA(
    TestAscendAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for Qwen3.5-397B-W8A8 1p1d_16p on GPQA"""

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    accuracy = 0.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    max_concurrency = 4
    num_prompts = 448
    output_len = 1024
    generation_kwargs = '{"temperature": 0}'

    def test_npu_qwen3_5_397b_w8a8_1p1d_16p_gpqa(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
