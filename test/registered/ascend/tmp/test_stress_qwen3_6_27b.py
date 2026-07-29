"""Stress test for Qwen3-235B model."""

import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_6_27B_W8A8_MODEL_PATH,
)
from sglang.test.ascend.test_npu_stress_utils import NpuStressTestRunner
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

# Register for CI - estimated 45 minutes
register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)

QWEN3_6_27B_3K5_1K5_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "130",
    "ASCEND_USE_FIA": "1",
}

QWEN3_6_27B_3K5_1K5_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    60000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    64,
    "--max-mamba-cache-size",
    74,
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs",
    2,
    8,
    16,
    32,
    40,
    45,
    50,
    54,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
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
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]


class TestStressQwen3627B(CustomTestCase):
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    random_input_len = 3500
    random_output_len = 1500
    output_file = "stress_test_Qwen3_6_27B.jsonl"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.runner = NpuStressTestRunner(
            test_name="Qwen3.6-27B Stress Test",
            base_url=cls.base_url,
        )

    def test_stress_qwen3_6_27b(self):
        try:
            success = self.runner.run_stress_test_for_model(
                model_path=self.model,
                random_input_len=self.random_input_len,
                random_output_len=self.random_output_len,
                output_file=self.output_file,
                server_args=QWEN3_6_27B_3K5_1K5_OTHER_ARGS,
                env=QWEN3_6_27B_3K5_1K5_ENVS,
            )

            self.assertTrue(success, f"Stress test failed for {self.model}")

        finally:
            self.runner.write_final_report()


if __name__ == "__main__":
    unittest.main()
