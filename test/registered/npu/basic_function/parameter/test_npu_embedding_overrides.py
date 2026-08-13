import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class TestEmbeddingOverrides(CustomTestCase):
    """Testcase: embed_override_token_id / embed_overrides on /v1/embeddings.

    The override swaps the INPUT embedding at the placeholder position
    (schedule_batch.py:2258-2276); the model still processes the sequence
    normally. Functional verification: swapping the override vector changes the
    output embedding — a no-op override mechanism would leave it unchanged.

    [Test Category] Parameter
    [Test Target] embed_override_token_id / embed_overrides
    """

    @classmethod
    def setUpClass(cls):
        cls.model = GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--is-embedding",
                "--attention-backend",
                "ascend",
            ],
        )
        cls.base_url += "/v1"
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url)

        # Discover hidden_size via a normal embedding.
        response = cls.client.embeddings.create(model=cls.model, input="Hello")
        cls.hidden_size = len(response.data[0].embedding)
        # A placeholder token unlikely to appear in normal text.
        cls.placeholder = 15339

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _unit_vector(self, position):
        vec = [0.0] * self.hidden_size
        vec[position] = 1.0
        return vec

    def _embed_with_override(self, input_ids, override):
        return self.client.embeddings.create(
            model=self.model,
            input=input_ids,
            extra_body={
                "embed_override_token_id": self.placeholder,
                "embed_overrides": [override],
            },
        )

    def test_token_id_without_overrides_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.embeddings.create(
                model=self.model,
                input="Hello",
                extra_body={"embed_override_token_id": self.placeholder},
            )
        self.assertIn(
            "embed_override_token_id requires embed_overrides", str(ctx.exception)
        )

    def test_overrides_without_token_id_400(self):
        with self.assertRaises(openai.BadRequestError) as ctx:
            self.client.embeddings.create(
                model=self.model,
                input="Hello",
                extra_body={"embed_overrides": [self._unit_vector(0)]},
            )
        self.assertIn(
            "embed_override_token_id is required", str(ctx.exception)
        )

    def test_override_takes_effect(self):
        """Single placeholder token: the output depends entirely on the override."""
        input_ids = [self.placeholder]
        ev = self._embed_with_override(input_ids, self._unit_vector(0)).data[0].embedding
        ew = self._embed_with_override(input_ids, self._unit_vector(1)).data[0].embedding
        self.assertLess(
            cosine(ev, ew),
            0.999,
            "swapping the override vector must change the output embedding",
        )

    def test_override_changes_output_with_context(self):
        """With real context tokens, the override still measurably changes output."""
        # NOTE: context tokens must NOT include the placeholder id, otherwise
        # the placeholder scan counts extra occurrences and rejects the request.
        input_ids = [314, 703, 284, self.placeholder]
        baseline = self.client.embeddings.create(
            model=self.model, input=input_ids
        ).data[0].embedding
        overridden = self._embed_with_override(
            input_ids, self._unit_vector(0)
        ).data[0].embedding
        self.assertLess(
            cosine(baseline, overridden),
            0.999,
            "the override should measurably change the pooled output",
        )

    def test_count_mismatch_400(self):
        """Placeholder appears twice but only one override entry → 400."""
        with self.assertRaises(openai.BadRequestError) as ctx:
            self._embed_with_override(
                [self.placeholder, self.placeholder], self._unit_vector(0)
            )
        self.assertIn("occurrences", str(ctx.exception))

    def test_batch_none_entry_skipped(self):
        """Batch with [override, None]: the None entry keeps the normal path."""
        # NOTE: the second batch item must not contain the placeholder id.
        input_ids = [[self.placeholder], [314, 703]]
        response = self.client.embeddings.create(
            model=self.model,
            input=input_ids,
            extra_body={
                "embed_override_token_id": self.placeholder,
                "embed_overrides": [[self._unit_vector(0)], None],
            },
        )
        self.assertEqual(len(response.data), 2)
        for item in response.data:
            self.assertTrue(len(item.embedding) > 0)

    def test_wrong_dimension_error(self):
        """Override vector length != hidden_size has no explicit validation —
        record the actual behavior (expected non-200)."""
        with self.assertRaises(openai.APIError):
            self._embed_with_override([self.placeholder], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
