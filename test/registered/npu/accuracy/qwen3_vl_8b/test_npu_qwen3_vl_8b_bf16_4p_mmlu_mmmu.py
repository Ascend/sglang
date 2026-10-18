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

# MMLU answers are short multiple choice; 16k cap trims long-tail generations.
MMLU_GENERATION_CONFIG = {
    "max_tokens": 16384,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 40,
    "presence_penalty": 2.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}

# MMMU needs the 32k budget for long generations; batch 32 bounds vision peaks.
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
    """Qwen3-VL-8B-Instruct bf16 TP4 accuracy: full MMLU + full MMMU on A3; thresholds sit below public scores (MMLU 0.807, MMMU 0.696) to absorb evalscope scoring variance."""

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
