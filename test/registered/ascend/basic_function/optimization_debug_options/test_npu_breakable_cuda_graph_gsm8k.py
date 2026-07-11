"""Breakable CUDA-graph (BCG) prefill capture accuracy test on NPU.

Exercises the breakable (BCG) CUDA-graph prefill capture path on
Kimi-K2.6-W4A8 so the BCG code runs in NPU CI:

  * runner_backend/breakable_cuda_graph_backend.py
        BCG must capture correctly with ascend backend.
  * The prefill is routed through the BCG path
        (is_in_breakable_cuda_graph()).

These branches are gated and default-off, so the default full-cuda-graph CI
tests never touch them. This test launches the server with
``--cuda-graph-backend-prefill breakable`` and asserts GSM8K accuracy,
turning "did the capture succeed and stay correct?" into a CI signal.
"""

import os
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import KIMI_K2_6_W4A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(est_time=3600, suite="debug-full-16-npu-a3", nightly=True)

MODEL_PATH = KIMI_K2_6_W4A8_MODEL_PATH
SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
ACCURACY_THRESHOLD = 0.9121


@dataclass
class CaptureConfig:
    """A prefill cuda-graph capture backend variant to validate."""

    variant: str
    # Extra server args that select the prefill capture backend.
    capture_args: List[str]
    env_vars: Dict[str, str] = field(default_factory=dict)


# Common args: TP4, ascend prefill/decode, fp8 KV, allreduce fusion, 8192 chunked prefill.
COMMON_ARGS: List[str] = [
    "--tensor-parallel-size",
    "16",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.765",
    "--disable-radix-cache",
    "--prefill-attention-backend",
    "ascend",
    "--decode-attention-backend",
    "ascend",
    "--kv-cache-dtype",
    "auto",
    "--max-running-requests",
    "1024",
    "--enable-aiter-allreduce-fusion",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]


def get_capture_configs() -> List[CaptureConfig]:
    return [
        # BCG: breakable prefill capture.
        CaptureConfig(
            variant="bcg",
            capture_args=[
                "--cuda-graph-backend-prefill",
                "breakable",
                "--cuda-graph-backend-decode",
                "full",
            ],
        ),
    ]


class TestNpuBreakableCudaGraphGsm8k(CustomTestCase):
    """Testcase: Validate BCG prefill capture accuracy on NPU.

    [Test Category] Parameter
    [Test Target] --cuda-graph-backend-prefill; --cuda-graph-backend-decode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.configs = get_capture_configs()

    def _run_variant(self, config: CaptureConfig) -> float:
        env = os.environ.copy()
        env["SGLANG_AITER_MLA_PERSIST"] = "1"
        for key, value in config.env_vars.items():
            env[key] = value

        other_args = list(COMMON_ARGS) + list(config.capture_args)
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )
        try:
            requests.get(self.base_url + "/flush_cache")
            args = SimpleNamespace(
                num_shots=8,
                data_path=None,
                num_questions=GSM8K_NUM_QUESTIONS,
                parallel=GSM8K_NUM_QUESTIONS,
                max_new_tokens=512,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            print(f"[{config.variant}] {metrics=}")
            return metrics["accuracy"]
        finally:
            kill_process_tree(process.pid)

    def test_bcg_gsm8k(self):
        summary = "### Kimi-K2.6-W4A8 BCG capture (NPU, TP4)\n\n"
        summary += "| Capture backend | Accuracy | Threshold | Status |\n"
        summary += "| --------------- | -------- | --------- | ------ |\n"

        failures = []
        for config in self.configs:
            with self.subTest(variant=config.variant):
                acc = self._run_variant(config)
                passed = acc >= ACCURACY_THRESHOLD
                status = "PASS" if passed else "FAIL"
                summary += (
                    f"| {config.variant} | {acc:.3f} | "
                    f"{ACCURACY_THRESHOLD} | {status} |\n"
                )
                if not passed:
                    failures.append((config.variant, acc))

        if is_in_ci():
            write_github_step_summary(summary)

        self.assertEqual(
            failures,
            [],
            f"BCG accuracy below {ACCURACY_THRESHOLD}: {failures}",
        )


if __name__ == "__main__":
    unittest.main()
