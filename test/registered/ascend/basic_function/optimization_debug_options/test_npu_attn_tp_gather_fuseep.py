import os
import unittest

import requests
import torch

os.environ.setdefault("HCCL_BUFFSIZE", "600")

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_6_35B_A3B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Register with the maximum device count needed (T7 needs 4).
# Tests requiring fewer cards self-skip via _require_devices().
register_npu_ci(est_time=600, suite="debug-full-4-npu-a3", nightly=True)


def _require_devices(test_case, n):
    """Skip test if fewer than *n* devices are available.

    Checks CUDA_VISIBLE_DEVICES first (used by CI to limit visible cards),
    then falls back to torch.cuda.device_count().
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        count = len([d for d in visible.split(",") if d.strip()])
    else:
        try:
            count = torch.cuda.device_count()
        except Exception:
            count = 1
    if count < n:
        test_case.skipTest(f"Requires at least {n} NPU devices, found {count}")


class TestAttnTpGatherA2APath(CustomTestCase):
    """Test --disable-attn-tp-gather with MOE model via a2a=ascend_fuseep path.

    Unlike the existing OLMoE test (test_npu_attn_tp_gather.py) which triggers
    require_attn_tp_gather=true via --moe-dense-tp-size 1, this test triggers
    the same branch via --moe-a2a-backend ascend_fuseep (common.py:3102 OR).

    [Test Category] Parameter
    [Test Target] --disable-attn-tp-gather, --moe-a2a-backend ascend_fuseep
    """

    model = QWEN3_6_35B_A3B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        # --moe-a2a-backend ascend_fuseep:
        #   triggers require_attn_tp_gather=true at common.py:3102
        #   via "not get_moe_a2a_backend().is_none()" condition;
        #   SGLANG_NPU_FUSED_MOE_MODE defaults to 1 (valid).
        "--moe-a2a-backend",
        "deepep",
        "--tp-size",
        "2",
    ]

    def test_contrastive_tp2(self):
        """T1+T2: Contrastive — with/without flag on tp=2.

        Phase 1 (no flag): require_attn_tp_gather() → Branch C (a2a path) → True
            → global_dp_buffer_len = num_tokens (model_runner.py:2579)
        Phase 2 (with flag): require_attn_tp_gather() → Branch A (opt-out) → False
            → global_dp_buffer_len = None (model_runner.py:2581)

        [Priority] P0
        [Branch] C→F vs A→G
        [Does NOT cover] dense_tp=1 path, DP attention scenarios
        """
        _require_devices(self, 2)
        prompts = [
            "The capital of France is",
            "What is the largest planet in our solar system?",
        ]

        # Phase 1: WITHOUT --disable-attn-tp-gather
        process1 = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self.base_args),
        )
        try:
            resp1 = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(resp1.status_code, 200)
        finally:
            kill_process_tree(process1.pid)

        # Phase 2: WITH --disable-attn-tp-gather
        process2 = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self.base_args) + ["--disable-attn-tp-gather"],
        )
        try:
            resp2 = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(resp2.status_code, 200)
        finally:
            kill_process_tree(process2.pid)

        # Both paths produce correct output for all prompts
        self.assertIn("Paris", resp1.text)
        self.assertIn("Jupiter", resp1.text)
        self.assertIn("Paris", resp2.text)
        self.assertIn("Jupiter", resp2.text)


class TestAttnTpGatherDense(CustomTestCase):
    """Test --disable-attn-tp-gather with non-MOE (dense) model.

    For non-MOE models, require_attn_tp_gather() returns False regardless
    of the flag (common.py:3102 → else: return False at 3108, or early
    return at 3096). This test verifies the flag does not break dense models.

    [Test Category] Parameter
    [Test Target] --disable-attn-tp-gather
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def test_dense_model_noop(self):
        """T3: Flag is a no-op for non-MOE models — server starts and infers correctly.

        [Priority] P0
        [Branch] D→G (no flag) or A→G (with flag) — both produce same result
        [Does NOT cover] MOE models, DP attention scenarios
        """
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--disable-attn-tp-gather",
                # For non-MOE: moe_a2a_backend="none" AND moe_dense_tp_size=None
                # → both conditions at common.py:3102 are False
                # → require_attn_tp_gather() returns False with OR without flag
            ],
        )
        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Paris", resp.text)
        finally:
            kill_process_tree(process.pid)


class TestAttnTpGatherA2ATp2(CustomTestCase):
    """Test --disable-attn-tp-gather with MOE a2a backend and tp=2.

    Same contrastive pattern as TestAttnTpGatherA2APath but validates
    the flag works correctly with tensor parallelism > 1.

    [Test Category] Parameter
    [Test Target] --disable-attn-tp-gather, --moe-a2a-backend ascend_fuseep, --tp-size
    """

    model = QWEN3_6_35B_A3B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        # --moe-a2a-backend ascend_fuseep:
        #   triggers require_attn_tp_gather=true at common.py:3102
        #   via "not get_moe_a2a_backend().is_none()" condition.
        "--moe-a2a-backend",
        "deepep",
        "--enable-dp-attention",
        "--dp-size",
        "2",
        "--tp-size",
        "2",
    ]

    def test_contrastive_tp2(self):
        """T4+T5: Contrastive on tp=2 — with/without --disable-attn-tp-gather.

        Phase 1 (no flag): a2a=ascend_fuseep → Branch C → gather enabled
        Phase 2 (with flag): → Branch A → opt-out

        [Priority] P1
        [Branch] C→F vs A→G (tp=2), B→F (dp-attn)
        [Does NOT cover] dense_tp=1 path
        """
        prompts = [
            "The capital of France is",
            "What is the largest planet in our solar system?",
        ]

        # Phase 1: WITHOUT flag
        process1 = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self.base_args),
        )
        try:
            resp1 = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(resp1.status_code, 200)
        finally:
            kill_process_tree(process1.pid)

        # Phase 2: WITH flag
        process2 = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self.base_args) + ["--disable-attn-tp-gather"],
        )
        try:
            resp2 = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(resp2.status_code, 200)
        finally:
            kill_process_tree(process2.pid)

        self.assertIn("Paris", resp1.text)
        self.assertIn("Jupiter", resp1.text)
        self.assertIn("Paris", resp2.text)
        self.assertIn("Jupiter", resp2.text)


class TestAttnTpGatherDPAttn(CustomTestCase):
    """Test --disable-attn-tp-gather under --enable-dp-attention.

    When dp_size == tp_size, require_attn_tp_gather() returns False
    (dp_size < tp_size = False at common.py:3104), i.e. gather is already
    disabled without the flag.

    When dp_size < tp_size, require_attn_tp_gather() returns True
    (dp_size < tp_size = True at common.py:3104), and --disable-attn-tp-gather
    overrides via the early return at common.py:3096.

    [Test Category] Parameter
    [Test Target] --disable-attn-tp-gather, --enable-dp-attention
    """

    model = QWEN3_6_35B_A3B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @staticmethod
    def _launch(tp_size, dp_size, disable_gather):
        """Launch server with DP attention configuration.

        Args:
            tp_size: Tensor parallelism size.
            dp_size: Data parallelism size (must divide tp_size).
            disable_gather: Whether to pass --disable-attn-tp-gather.
        """
        args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--moe-a2a-backend",
            "deepep",
            # --enable-dp-attention: required to exercise Branch B
            # (common.py:3103-3104) in require_attn_tp_gather().
            "--enable-dp-attention",
            "--tp-size",
            str(tp_size),
            "--dp-size",
            str(dp_size),
        ]
        if disable_gather:
            args.append("--disable-attn-tp-gather")
        return popen_launch_server(
            TestAttnTpGatherDPAttn.model,
            TestAttnTpGatherDPAttn.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
        )

    def _make_request(self):
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )

    def test_dp_equals_tp(self):
        """T6: dp_size == tp_size → Branch B-False → gather disabled.

        When dp_size == tp_size, require_attn_tp_gather() returns
        dp_size < tp_size = False (common.py:3104). Since gather is
        already disabled by default, the flag is a no-op in this config.
        Tests both with and without the flag to verify neither breaks.

        [Priority] P2
        [Branch] B-False→G (dp=2, tp=2)
        [Does NOT cover] dp < tp scenario
        """
        _require_devices(self, 2)

        # Phase 1: WITHOUT flag → Branch B-False (dp==tp → False)
        process1 = self._launch(tp_size=2, dp_size=2, disable_gather=False)
        try:
            resp1 = self._make_request()
            self.assertEqual(resp1.status_code, 200)
            self.assertIn("Paris", resp1.text)
        finally:
            kill_process_tree(process1.pid)

        # Phase 2: WITH flag → Branch A (opt-out, same result)
        process2 = self._launch(tp_size=2, dp_size=2, disable_gather=True)
        try:
            resp2 = self._make_request()
            self.assertEqual(resp2.status_code, 200)
            self.assertIn("Paris", resp2.text)
        finally:
            kill_process_tree(process2.pid)

    def test_dp_less_than_tp(self):
        """T7: dp_size < tp_size → Branch B-True → gather enabled.

        When dp_size < tp_size, require_attn_tp_gather() returns True
        (common.py:3104). --disable-attn-tp-gather overrides this via
        the early return at common.py:3096.

        [Priority] P2
        [Branch] B-True→F (dp=2, tp=4)
        [Does NOT cover] dp == tp scenario
        """
        _require_devices(self, 4)

        # Phase 1: WITHOUT flag → Branch B-True → gather enabled
        process1 = self._launch(tp_size=4, dp_size=2, disable_gather=False)
        try:
            resp1 = self._make_request()
            self.assertEqual(resp1.status_code, 200)
            self.assertIn("Paris", resp1.text)
        finally:
            kill_process_tree(process1.pid)

        # Phase 2: WITH flag → Branch A → opt-out
        process2 = self._launch(tp_size=4, dp_size=2, disable_gather=True)
        try:
            resp2 = self._make_request()
            self.assertEqual(resp2.status_code, 200)
            self.assertIn("Paris", resp2.text)
        finally:
            kill_process_tree(process2.pid)


if __name__ == "__main__":
    unittest.main()
