import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import QWEN3_VL_8B_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=9000, suite="full-acc-4-npu-a3", nightly=True)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

OTHER_ARGS = [
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--trust-remote-code",
    "--tp-size",
    4,
    "--mm-attention-backend",
    "ascend_attn",
    "--disable-radix-cache",
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    -1,
    "--enable-multimodal",
    "--cuda-graph-bs-decode",
    8,
    16,
    24,
    32,
    48,
    58,
    64,
    "--cuda-graph-bs-prefill",
    8,
    16,
    24,
    32,
    48,
    58,
    64,
    "--prefill-delayer-max-delay-passes",
    100,
    "--sampling-backend",
    "ascend",
    "--mem-fraction-static",
    0.85,
]

# Generation config follows the manually-verified A5 setup.
# chat_template_kwargs passthrough via evalscope generation_config is
# unverified; drop it if the first run errors out.
# MMLU answers are short (multiple choice); a 16k cap trims long-tail
# generations. Batch 64 fits the KV pool comfortably (extreme case
# 64 x 18k tokens ~= 1.15M vs ~5.2M pool across 4 cards).
MMLU_GENERATION_CONFIG = {
    "max_tokens": 16384,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 40,
    "presence_penalty": 2.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}

# MMMU keeps the full 32k budget for long generations; batch 32 bounds
# vision-prefill activation peaks.
MMMU_GENERATION_CONFIG = {
    "max_tokens": 32768,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 40,
    "presence_penalty": 2.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}


class TestQwen3VL8B_BF16_4P_MMLU_MMMU(TestNpuAccuracyTestCaseBase):
    """Qwen3-VL-8B-Instruct bf16 TP4 accuracy: full MMLU + full MMMU on A3.

    Baselines: MMLU threshold 0.75 (public score 0.807,
    https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct); MMMU threshold 0.58
    (public score 0.696, lower bound covers evalscope-vs-VLMEvalKit scoring
    delta; local A5 evalscope measured 0.58+).
    """

    model = QWEN3_VL_8B_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    eval_batch_size = 16

    def test_mmlu(self):
        self.datasets = ["mmlu"]
        self.accuracy = 0.75
        self.eval_batch_size = 64
        self.generation_config = MMLU_GENERATION_CONFIG
        self.run_accuracy()

    def test_mmmu(self):
        self.datasets = ["mmmu"]
        self.accuracy = 0.58
        self.eval_batch_size = 32
        self.generation_config = MMMU_GENERATION_CONFIG
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
