import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_30B_A3B_HUB_MODEL_PATH,
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

QWEN3_30B_A3B_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

QWEN3_30B_A3B_OTHER_ARGS = [
    "--tp-size",
    2,
    "--dp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "qwen3",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--enable-dp-attention",
]


class TestNPUQwen3_30B_A3B_gpqa(TestAscendAccuracyTestCaseBase):
    model = QWEN3_30B_A3B_HUB_MODEL_PATH
    envs = QWEN3_30B_A3B_ENVS
    other_args = QWEN3_30B_A3B_OTHER_ARGS
    accuracy = 0.578
    datasets = ["gpqa_diamond"]
    eval_batch_size = 64
    generation_config = {
        "max_tokens": 40000,
        "temperature": 0.0,
    }

    def test_gpqa(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
