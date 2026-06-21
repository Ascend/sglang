import unittest

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_deterministic_utils import TestDeterministicBase

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendDeterministic(TestDeterministicBase):
    """Test deterministic inference with Ascend attention backend.

    [Test Category] Core
    [Test Target] deterministic inference, attention backend
    """

    @classmethod
    def get_server_args(cls):
        args = [
            "--model-path",
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            "--trust-remote-code",
            "--cuda-graph-max-bs",
            "32",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        return args


if __name__ == "__main__":
    unittest.main()
