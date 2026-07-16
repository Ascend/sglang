import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class _Eagle3ParityBase(Eagle3Base):
    """Shared knobs for EAGLE3 parity variants; no test methods."""

    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)
    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128


def _greedy(url, text, max_new_tokens=48):
    return requests.post(
        url + "/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
    ).json()["text"]


class SpecParityKitNPU:
    """Lossless output parity vs a non-spec reference on NPU.

    Sequential (NOT concurrent): launch a non-spec reference server on the
    standard port, capture greedy outputs, tear it down, THEN let the fixture
    launch the spec server. Only one model is resident at a time -- two 8B
    servers don't fit on one NPU. Mix this kit FIRST in the bases so its
    setUpClass runs before the fixture's:  ``class T(SpecParityKit, Eagle3Base)``.
    """

    parity_prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "The three primary colors are",
        "def fibonacci(n):",
    ]

    @classmethod
    def setUpClass(cls):
        ref_url = DEFAULT_URL_FOR_TEST
        ref_proc = popen_launch_server(
            cls.model,
            ref_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--base-gpu-id",
                1,
                "--mem-fraction-static",
                "0.8",  # ref alone -> full GPU available
                "--attention-backend",
                cls.attention_backend,
                "--page-size",
                "128",
                "--dtype",
                cls.dtype,
                *(["--trust-remote-code"] if cls.trust_remote_code else []),
            ],
        )
        try:
            cls.parity_ref_outputs = {
                p: _greedy(ref_url, p) for p in cls.parity_prompts
            }
        finally:
            kill_process_tree(ref_proc.pid, wait_timeout=60)
        # Now the spec server (same port; ref is gone).
        super().setUpClass()

    def test_parity_vs_reference(self):
        """Spec decode greedy output must equal the non-spec reference."""
        for prompt in self.parity_prompts:
            spec_out = _greedy(self.base_url, prompt)
            self.assertEqual(
                spec_out,
                self.parity_ref_outputs[prompt],
                f"spec != ref for prompt {prompt!r}",
            )


class TestEagle3ParityNPU(SpecParityKitNPU, _Eagle3ParityBase):
    """Test Case: Verify EAGLE3 speculative decoding greedy output matches non-speculative reference on NPU.

    [Test Category] Functional
    [Test Target] --speculative-algorithm
    """

    disable_overlap = False


if __name__ == "__main__":
    unittest.main()
