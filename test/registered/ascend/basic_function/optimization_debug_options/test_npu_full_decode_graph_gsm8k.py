"""Full decode CUDA-graph capture accuracy test on NPU.

Exercises the ``--cuda-graph-backend-decode full`` path on
Qwen3-30B-A3B to verify that full decode graph capture does not
degrade accuracy on NPU.
"""

import os
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_npu_ci(est_time=500, suite="debug-full-2-npu-a3", nightly=True)

MODEL_PATH = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))
ACCURACY_THRESHOLD = 0.90


@dataclass
class CaptureConfig:
    """A prefill cuda-graph capture backend variant to validate."""

    variant: str
    # Extra server args that select the prefill capture backend.
    capture_args: List[str]
    env_vars: Dict[str, str] = field(default_factory=dict)


# Common args: TP2, ascend backend, 8192 chunked prefill.
COMMON_ARGS: List[str] = [
    "--tensor-parallel-size",
    "2",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
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
        summary = "### Qwen3-30B-A3B full decode graph (NPU, TP2)\n\n"
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
