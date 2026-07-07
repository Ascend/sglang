"""NPU GLM-5.1-w4a8 prefill context-parallel (CP) accuracy test (8-NPU).

Adapted from test_deepseek_v4_pro_fp4_cp.py for NPU with GLM-5.1-w4a8 model.
Enables prefill CP via ``--enable-prefill-cp --cp-strategy interleave`` over the
ascend backend.

Registry: full-8-npu-a3 suite
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import GLM_5_1_W4A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(
    est_time=3600, suite="debug-full-8-npu-a3", nightly=True
)

GLM5_1_MODEL = GLM_5_1_W4A8_MODEL_PATH
SERVER_LAUNCH_TIMEOUT = 3600

COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_DP_USE_GATHERV": "1",
    "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
    "AITER_BF16_FP8_MOE_BOUND": "0",
}

FP4_ENV_VARS = {
    "SGLANG_DSV4_FP4_EXPERTS": "true",
}


class TestGLM5_1PrefillCPInterleave(CustomTestCase):
    """GLM-5.1-w4a8 prefill CP, interleave (round-robin-split), tp=8."""

    @classmethod
    def setUpClass(cls):
        cls.model = GLM5_1_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST

        env = os.environ.copy()
        env.update(COMMON_ENV_VARS)
        env.update(FP4_ENV_VARS)

        other_args = [
            "--trust-remote-code",
            "--tp", "8",
            "--dp", "1",
            # "--enable-prefill-cp",
            # "--cp-strategy", "interleave",
            "--enable-nsa-prefill-context-parallel",
            "--nsa-prefill-cp-mode", "in-seq-split",
            "--attn-cp-size", "8",
            "--disable-radix-cache",
            "--attention-backend", "ascend",
            "--max-running-requests", "256",
            "--page-size", "256",
            "--mem-fraction-static", "0.90",
            "--swa-full-tokens-ratio", "0.1",
            "--chunked-prefill-size", "8192",
            "--disable-shared-experts-fusion",
            "--tool-call-parser", "glm47",
            "--reasoning-parser", "glm45",
            "--quantization", "modelslim",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1319,
            num_threads=32,
            num_shots=5,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_a_gsm8k (glm5.1-w4a8-cp-interleave)\n"
                f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], 0.92)


if __name__ == "__main__":
    unittest.main()
