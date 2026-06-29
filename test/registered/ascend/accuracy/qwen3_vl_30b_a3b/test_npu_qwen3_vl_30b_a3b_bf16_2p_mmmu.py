import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_VL_30B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="full-4-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)
register_npu_ci(
    est_time=3600,
    suite="stage-b-test-4-npu-a3",
    nightly=True,
    disabled="accuracy testcase",
)

ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--mm-attention-backend",
    "ascend_attn",
    "--device",
    "npu",
    "--tp-size",
    2,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "qwen3",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--enable-multimodal",
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    102400,
    "--max-running-requests",
    512,
]


class TestNPUQWEN3_VL_30B_A3B_mmmu(TestAscendAccuracyTestCaseBase):
    model = QWEN3_VL_30B_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.676
    datasets = ["mmmu"]
    generation_config = {
        "max_tokens": 40000,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    eval_batch_size = 64

    def test_mmmu(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
