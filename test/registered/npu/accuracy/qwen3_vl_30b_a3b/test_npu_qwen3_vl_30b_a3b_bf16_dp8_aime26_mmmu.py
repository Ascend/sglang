import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import QWEN3_VL_30B_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci

# Temporary registration: this case is exercised via single-test-npu.yml first.
# Before promoting to a nightly suite, re-register with the target suite name.
register_npu_ci(
    est_time=27000,
    suite="",
    nightly=True,
    disabled="pending accuracy validation",
)

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

OTHER_ARGS = [
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--trust-remote-code",
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--mm-attention-backend",
    "ascend_attn",
    "--disable-radix-cache",
    "--enable-multimodal",
    "--mem-fraction-static",
    0.7,
]

# Generation config follows the manually-verified A5 setup.
# chat_template_kwargs passthrough via evalscope generation_config is
# unverified; drop it if the first run errors out.
AIME26_GENERATION_CONFIG = {
    "max_tokens": 40000,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": True},
}

MMMU_GENERATION_CONFIG = {
    "max_tokens": 40000,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": True},
}


class TestQwen3VL30BA3B_BF16_DP8_AIME26_MMMU(TestNpuAccuracyTestCaseBase):
    """Qwen3-VL-30B-A3B-Instruct bf16 DP8 accuracy: AIME26 + full MMMU on A3.

    Baselines: AIME26 has no official public score (to be re-anchored after
    the first GPU baseline run; 0.6 is a conservative placeholder with the
    built-in 2/30-question fluctuation tolerance). MMMU threshold 0.65
    (public score 0.742, https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct,
    lower bound covers evalscope-vs-VLMEvalKit scoring delta).
    """

    model = QWEN3_VL_30B_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    eval_batch_size = 16

    def test_aime26(self):
        self.datasets = ["aime26"]
        self.accuracy = 0.6
        self.generation_config = AIME26_GENERATION_CONFIG
        self.run_accuracy()

    def test_mmmu(self):
        self.datasets = ["mmmu"]
        self.accuracy = 0.65
        self.generation_config = MMMU_GENERATION_CONFIG
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
