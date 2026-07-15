"""Full decode CUDA-graph capture accuracy test on NPU.

Exercises the ``--cuda-graph-backend-decode full`` path on
Kimi-K2.6-W4A8 with modelslim quantization to verify that full decode
graph capture does not degrade accuracy on NPU.
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
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(est_time=3600, suite="full-16-npu-a3", nightly=True)

MODEL_PATH = KIMI_K2_6_W4A8_MODEL_PATH
SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))
ACCURACY_THRESHOLD = 0.9121


@dataclass
class CaptureConfig:
    """A prefill cuda-graph capture backend variant to validate."""

    variant: str
    # Extra server args that select the prefill capture backend.
    capture_args: List[str]
    env_vars: Dict[str, str] = field(default_factory=dict)


# Common args: TP16, ascend backend, modelslim quantization, 8192 chunked prefill.
COMMON_ARGS: List[str] = [
    "--tensor-parallel-size",
    "16",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--quantization",
    "modelslim",
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
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "4",
    "8",
    "--enable-dp-attention",
    "--dp-size",
    "2",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]


def get_capture_configs() -> List[CaptureConfig]:
    return [
        # Full decode graph capture.
        CaptureConfig(
            variant="bcg",
            capture_args=[
                "--cuda-graph-backend-prefill",
                "disabled",
                "--cuda-graph-backend-decode",
                "full",
            ],
            env_vars={
                "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
                "HCCL_BUFFSIZE": "1200",
                "HCCL_OP_EXPANSION_MODE": "AIV",
            },
        ),
    ]


class TestNpuFullDecodeGraphGsm8k(CustomTestCase):
    """Testcase: Validate full decode CUDA-graph accuracy on NPU.

    [Test Category] Parameter
    [Test Target] --cuda-graph-backend-decode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.configs = get_capture_configs()

    def _run_variant(self, config: CaptureConfig) -> float:
        env = os.environ.copy()
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
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                num_shots=8,
                num_examples=GSM8K_NUM_QUESTIONS,
                num_threads=128,
                max_tokens=512,
                base_url=self.base_url,
            )
            metrics = run_eval(args)
            print(f"[{config.variant}] {metrics=}")
            return metrics["score"]
        finally:
            kill_process_tree(process.pid)

    def test_full_decode_graph_gsm8k(self):
        summary = "### Kimi-K2.6-W4A8 full decode graph (NPU, TP16)\n\n"
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
            f"Full decode graph accuracy below {ACCURACY_THRESHOLD}: {failures}",
        )


if __name__ == "__main__":
    unittest.main()
