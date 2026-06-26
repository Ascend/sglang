"""NPU adaptation of the EAGLE3 variant from test_hicache_variants.py.

The GPU `test_hicache_variants.py` defines four HiCache variants:
  * TestHiCacheStandard  - already covered by test_npu_hicache.py / test_npu_hicache_mha.py
  * TestHiCacheMLA       - already covered by test_npu_hicache_mla.py
  * TestHiCachePage      - already covered by test_npu_hicache_page.py
  * TestHiCacheEagle     - ** NOT covered by any existing NPU test ** -> this file

This file therefore ports ONLY `TestHiCacheEagle`, i.e. the EAGLE3
speculative-decoding + HiCache combination.

Why EAGLE3 lives in its own file (instead of being appended to
`test_npu_hicache.py` as a `test_004`):
  * `test_npu_hicache.py` is registered with `suite="full-1-npu-a3"`,
    i.e. it runs on a 1-NPU runner.
  * EAGLE3 on NPU requires `--tp-size 4` (see
    `test_npu_speculative_attention_mode.py`), so it must run on a
    4-NPU runner. Placing it in `test_npu_hicache.py` would make the
    1-NPU job fail at server launch.
  * Hence a dedicated file registered with `suite="stage-b-test-4-npu-a3"`.

[Test Category] Functional
[Test Target] HiCache + EAGLE3 speculative decoding coexistence

Key observation points ported from the GPU `TestHiCacheEagle`:
  * The server boots successfully with both `--enable-hierarchical-cache`
    and `--speculative-algorithm EAGLE3` enabled at the same time.
  * MMLU accuracy does not regress (GPU threshold 0.72; relaxed to 0.65
    here to match the threshold used by the existing NPU HiCache tests,
    since NPU uses a different target model than GPU).

NPU adaptation notes (see report for the full rationale):
  * Target model: QWEN3_8B (GPU used Llama-3.1-8B-Instruct; NPU has no
    Llama-3.1-8B EAGLE3 draft, so we pick the closest NPU-mature EAGLE3
    pair: Qwen3-8B + Qwen3-8B-EAGLE3, both shipped in
    `test_ascend_utils.py`).
  * EAGLE3 spec args follow `test_npu_speculative_attention_mode.py`:
    `--tp-size 4 --mem-fraction-static 0.7 --dtype bfloat16
    --speculative-num-steps 4 --speculative-eagle-topk 1
    --speculative-num-draft-tokens 5` and the
    `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` / `SGLANG_ENABLE_SPEC_V2=1`
    env vars. Note: `--disable-radix-cache` (used by the spec-only NPU
    test) is intentionally dropped here because HiCache's device tier
    IS the radix cache; the two flags are mutually exclusive.
  * HiCache args: `--enable-hierarchical-cache --hicache-ratio 1.2`
    (GPU used `--hicache-ratio 1.2 --mem-fraction-static 0.7`; the
    `--page-size 128 / --attention-backend ascend / --disable-cuda-graph`
    NPU essentials are added).
  * Evaluation uses `sglang.test.run_eval.run_eval` with `eval_name=mmlu`
    (NPU convention, see `test_npu_hicache_page.py`) instead of GPU's
    `MMLUMixin`.
  * Risk: EAGLE3 + HiCache coexistence on NPU is not yet validated by any
    existing test. If the combination is unsupported, the server launch
    will fail and the PR will surface it loudly so we can either fix the
    runtime or skip the test with a clear reason in a follow-up commit.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=600,
    suite="stage-b-test-4-npu-a3",
    nightly=False,
)


class TestHiCacheEagle(CustomTestCase):
    """HiCache + EAGLE3 speculative decoding: verify they coexist without
    regressing MMLU accuracy.

    [Test Category] Functional
    [Test Target] --enable-hierarchical-cache + --speculative-algorithm EAGLE3
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    mmlu_score_threshold = 0.65

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--speculative-draft-model-quantization",
                "unquant",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                QWEN3_8B_EAGLE3_WEIGHTS_PATH,
                "--speculative-num-steps",
                "4",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "5",
                "--speculative-attention-mode",
                "decode",
                "--tp-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--disable-cuda-graph",
                "--dtype",
                "bfloat16",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "1.2",
                "--page-size",
                "128",
            ],
            env={
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "TRANSFORMERS_VERBOSITY": "error",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        """MMLU score must clear the threshold with HiCache + EAGLE3 both on."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        score = float(metrics["score"])
        self.assertGreaterEqual(
            score,
            self.mmlu_score_threshold,
            f"MMLU score {score} below threshold {self.mmlu_score_threshold}",
        )


if __name__ == "__main__":
    unittest.main()
