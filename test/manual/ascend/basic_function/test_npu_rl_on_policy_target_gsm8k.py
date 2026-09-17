import os
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_4B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=200,
    suite="full-1-npu-a3",
    nightly=True,
    disabled="Dependency operator blue zone not available",
)

# When using the Cann9.1.0 version image, it is necessary to modify the environment variables
ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "LD_LIBRARY_PATH": (
        "/usr/local/Ascend/cann-9.0.0/opp/vendors/batch_invariant/op_api/lib/:"
        f"{os.environ.get('LD_LIBRARY_PATH', '')}"
    ),
}

OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    1,
    "--chunked-prefill-size",
    -1,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.8,
    "--enable-deterministic-inference",
    "--rl-on-policy-target",
    "fsdp",
    "--disable-cuda-graph",
]


class TestNPUQwen3_4B_1P_GSM8K(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for Qwen3-4B on GSM8K with rl-on-policy-target=fsdp.

    The shell script enables deterministic inference and rl-on-policy-target
    for RL training consistency. This test verifies GSM8K accuracy under
    these settings.

    [Test Category] Parameter
    [Test Target] --rl-on-policy-target
    """

    model = QWEN3_4B_WEIGHTS_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.80
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 2048,
        "stream": True,
        "timeout": 600,
    }
    eval_batch_size = 64
    limit = 256

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
