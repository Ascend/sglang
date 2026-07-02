import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=500, suite="debug-full-2-npu-a3", nightly=True)

# Prompt long enough to guarantee CP split is triggered:
# ZigzagCPStrategy.can_apply requires num_tokens >= 2*cp_size=4.
# This prompt (~150 tokens) ensures CP is exercised on every request.
_LONG_PROMPT = (
    "Explain the water cycle in detail. The water cycle describes how water "
    "evaporates from the surface of the earth, rises into the atmosphere, cools "
    "and condenses into clouds, and falls back to the surface as precipitation. "
    "The water that falls to earth then evaporates again, continuing the cycle. "
    "This process involves several key stages: evaporation, transpiration, "
    "condensation, precipitation, infiltration, and runoff. Each of these "
    "plays an important role in the continuous movement of water."
)


class _CPTestBase(CustomTestCase):
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def _open_logs(cls):
        out_log = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log = open("./cache_err_log.txt", "w+", encoding="utf-8")
        return out_log, err_log

    @classmethod
    def _close_logs(cls, out_log, err_log):
        out_log.close()
        err_log.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    @classmethod
    def _launch_with_args(cls, extra_args):
        out_log, err_log = cls._open_logs()
        proc = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=extra_args,
            return_stdout_stderr=(out_log, err_log),
        )
        return proc, out_log, err_log

    @classmethod
    def _stop(cls, proc, out_log, err_log):
        kill_process_tree(proc.pid)
        cls._close_logs(out_log, err_log)

    def _request(self, text, max_new_tokens=32):
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )


class TestNpuCPStrategyContrastive(_CPTestBase):
    """Testcase: verify zigzag and interleave strategies produce identical
    correct outputs under TP=2 — two different code paths MUST converge.

    Uses both a short factual prompt (correctness check) and a long prompt
    (~150 tokens, guarantees CP split triggers) for contrastive comparison.

    [Test Category] Parameter / Contrastive
    [Test Target] --cp-strategy; --enable-prefill-cp
    """

    base_extra_args = [
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--tp-size", "2",
        "--attn-cp-size", "2",
        "--enable-prefill-cp",
    ]

    def _launch(self, strategy):
        args = list(self.base_extra_args) + ["--cp-strategy", strategy]
        return self._launch_with_args(args)

    def test_cp_strategy_zigzag_mmlu(self):
        # --- zigzag ---
        proc1, out1, err1 = self._launch("zigzag")
        try:
            r_short = self._request("The capital of France is")
            self.assertEqual(r_short.status_code, 200)
            zigzag_short = r_short.text
            self.assertIn("Paris", zigzag_short)

            r_long = self._request(_LONG_PROMPT)
            self.assertEqual(r_long.status_code, 200)
            zigzag_long = r_long.text
        finally:
            self._stop(proc1, out1, err1)

        # --- interleave ---
        proc2, out2, err2 = self._launch("interleave")
        try:
            r_short = self._request("The capital of France is")
            self.assertEqual(r_short.status_code, 200)
            interleave_short = r_short.text
            self.assertIn("Paris", interleave_short)

            r_long = self._request(_LONG_PROMPT)
            self.assertEqual(r_long.status_code, 200)
            interleave_long = r_long.text
        finally:
            self._stop(proc2, out2, err2)

        self.assertEqual(
            zigzag_short,
            interleave_short,
            f"short: zigzag={zigzag_short!r} != interleave={interleave_short!r}",
        )
        self.assertEqual(
            zigzag_long,
            interleave_long,
            f"long: zigzag and interleave outputs diverge "
            f"(len zigzag={len(zigzag_long)}, len interleave={len(interleave_long)})",
        )


class TestNpuCPStrategyAliases(_CPTestBase):
    """Testcase: verify deprecated CP aliases forward correctly and produce
    valid output on NPU. Each alias maps to a canonical --cp-strategy value
    via _handle_legacy_cp_arguments.

    [Test Category] Parameter / Error Path
    [Test Target] --enable-prefill-context-parallel; --enable-dsa-prefill-context-parallel; --dsa-prefill-cp-mode
    """

    base_extra_args = [
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--tp-size", "2",
        "--attn-cp-size", "2",
    ]

    def _run_and_check(self, extra_args):
        args = list(self.base_extra_args) + extra_args
        proc, out, err = self._launch_with_args(args)
        try:
            r_short = self._request("The capital of France is")
            self.assertEqual(r_short.status_code, 200)
            self.assertIn("Paris", r_short.text)

            r_long = self._request(_LONG_PROMPT)
            self.assertEqual(r_long.status_code, 200)
            self.assertGreater(len(r_long.text), 10)
        finally:
            self._stop(proc, out, err)

    def test_enable_prefill_cp_alias(self):
        """--enable-prefill-context-parallel → --enable-prefill-cp + zigzag."""
        self._run_and_check(["--enable-prefill-context-parallel"])

    def test_enable_dsa_prefill_cp_alias(self):
        """--enable-dsa-prefill-context-parallel → interleave.
        On non-DSA models (Llama) the general CP path is used."""
        self._run_and_check(["--enable-dsa-prefill-context-parallel"])

    def test_dsa_cp_mode_alias(self):
        """in-seq-split → zigzag via legacy mode→strategy mapping."""
        self._run_and_check(
            ["--enable-prefill-cp", "--dsa-prefill-cp-mode", "in-seq-split"]
        )


if __name__ == "__main__":
    unittest.main()
