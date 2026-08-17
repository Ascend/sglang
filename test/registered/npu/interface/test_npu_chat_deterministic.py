import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

PROMPT = "Write a short poem about autumn."


@unittest.skip(
    "The batch_invariant_ops package (registers torch.ops.batch_invariant_ops."
    "npu_* kernels) is an RL-requirement operator that is not open-sourced "
    "and unavailable in the public CI image; the seed path requires manual "
    "testing. Re-enable once the package is available in CI."
)
class TestChatSeedDeterministic(CustomTestCase):
    """Testcase: seed behavior under --enable-deterministic-inference.

    With the flag on, the sampling seed locks the random sequence. temperature=2.0
    maximizes randomness, so identical outputs across two requests can only be
    explained by the seed taking effect.

    [Test Category] Interface
    [Test Target] seed
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
                "--enable-deterministic-inference",
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _chat(self, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=2.0,
            max_tokens=64,
            **kwargs,
        )

    def test_seed_same_seed_identical(self):
        first = self._chat(seed=42).choices[0].message.content
        second = self._chat(seed=42).choices[0].message.content
        self.assertEqual(
            first,
            second,
            "same seed at temperature=2.0 should produce identical output",
        )

    def test_seed_different_seed_differs(self):
        first = self._chat(seed=42).choices[0].message.content
        second = self._chat(seed=43).choices[0].message.content
        self.assertNotEqual(
            first,
            second,
            "different seeds should produce different random sequences",
        )

    def test_seed_stream_consistent(self):
        non_stream_text = self._chat(seed=42).choices[0].message.content
        stream = self._chat(seed=42, stream=True)
        stream_text = "".join(
            chunk.choices[0].delta.content
            for chunk in stream
            if chunk.choices and chunk.choices[0].delta.content
        )
        self.assertEqual(
            non_stream_text,
            stream_text,
            "stream and non-stream with the same seed should agree",
        )

    def test_seed_n2_behavior(self):
        """n=2 sub-requests inherit the same seed — document the actual behavior."""
        response = self._chat(seed=42, n=2)
        texts = [choice.message.content for choice in response.choices]
        self.assertEqual(len(texts), 2)
        # Sub-requests share the parent seed; whether outputs are identical
        # depends on engine seed-offset handling — record both possibilities
        # rather than asserting one, and require both to be non-empty.
        for text in texts:
            self.assertTrue(text, "each choice should produce content")


if __name__ == "__main__":
    unittest.main()
