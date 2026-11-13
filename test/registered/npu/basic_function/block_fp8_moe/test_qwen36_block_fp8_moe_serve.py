import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3600, suite="full-2-npu-a3", nightly=True)

SERVER_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM": "1",
    "ASCEND_USE_FIA": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
}

SERVER_ARGS = [
    "--tp-size",
    "2",
    "--nnodes",
    "1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    "-1",
    "--max-total-tokens",
    "600000",
    "--max-prefill-tokens",
    "65536",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests",
    "10",
    "--max-mamba-cache-size",
    "20",
    "--mem-fraction-static",
    "0.65",
    "--cuda-graph-bs-decode",
    "2",
    "4",
    "8",
    "12",
    "14",
    "16",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]

ASCEND_OPS_ARGS = [
    "--moe-a2a-backend",
    "ascend_ops",
    "--moe-ascend-ops-config",
    '{"prefill": "alltoallv", "decode": "low_latency"}',
    "--ep-size",
    "2",
]


class _Qwen36BlockFp8(GSM8KAscendMixin):
    """Common Qwen3.6-35B-A3B-FP8 server args and GSM8K protocol; not a CustomTestCase so unittest does not collect it."""

    model = QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH
    other_args = SERVER_ARGS
    # Eval protocol differs from the mixin default (completion/512).
    api = "chat"
    max_tokens = 4096
    # 200-question smoke subset: 5 shots + 195 scored.
    num_questions = 200
    num_threads = 16
    chat_template_kwargs = {"enable_thinking": False}


class TestE2E1NativeBlockFp8(_Qwen36BlockFp8, CustomTestCase):
    """Testcase: Verify that the inference accuracy of Qwen/Qwen3.6-35B-A3B-FP8 on the GSM8K dataset is no less than 0.95 with the native Block FP8 mode (no requantization).

    [Test Category] Accuracy
    [Test Target] SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=0
    [Test Config] TP2, npu graph enabled, NEXTN speculative decoding, GSM8K 200-question subset, chat api with thinking disabled
    """

    accuracy = 0.95
    env = {**SERVER_ENVS, "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": "0"}


class TestE2E2RequantMxfp8(_Qwen36BlockFp8, CustomTestCase):
    """Testcase: Verify that the inference accuracy of Qwen/Qwen3.6-35B-A3B-FP8 on the GSM8K dataset is no less than 0.94 with the requantized MXFP8 mode (MXFP8 is a second lossy conversion, so the bar is lower).

    [Test Category] Accuracy
    [Test Target] SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=1
    [Test Config] TP2, npu graph enabled, NEXTN speculative decoding, GSM8K 200-question subset, chat api with thinking disabled
    """

    accuracy = 0.94
    env = {**SERVER_ENVS, "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": "1"}


class TestE2E8AscendOpsNative(_Qwen36BlockFp8, CustomTestCase):
    """Testcase: Verify that the inference accuracy of Qwen/Qwen3.6-35B-A3B-FP8 on the GSM8K dataset is no less than 0.95 with the ascend_ops MoE backend in the native Block FP8 mode (cross-regression guard for the shared MoE dispatch code).

    [Test Category] Cross Regression
    [Test Target] --moe-a2a-backend ascend_ops + SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=0
    [Test Config] TP2 EP2, npu graph enabled, NEXTN speculative decoding, GSM8K 200-question subset, chat api with thinking disabled
    """

    other_args = SERVER_ARGS + ASCEND_OPS_ARGS
    accuracy = 0.95
    env = {**SERVER_ENVS, "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": "0"}


class TestE2E9AscendOpsMxfp8(_Qwen36BlockFp8, CustomTestCase):
    """Testcase: Verify that the inference accuracy of Qwen/Qwen3.6-35B-A3B-FP8 on the GSM8K dataset is no less than 0.94 with the ascend_ops MoE backend in the requantized MXFP8 mode (cross-regression guard for the shared MoE dispatch code).

    [Test Category] Cross Regression
    [Test Target] --moe-a2a-backend ascend_ops + SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8=1
    [Test Config] TP2 EP2, npu graph enabled, NEXTN speculative decoding, GSM8K 200-question subset, chat api with thinking disabled
    """

    other_args = SERVER_ARGS + ASCEND_OPS_ARGS
    accuracy = 0.94
    env = {**SERVER_ENVS, "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": "1"}


if __name__ == "__main__":
    unittest.main()
