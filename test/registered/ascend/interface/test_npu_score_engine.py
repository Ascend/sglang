import json
import os
import unittest

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_0_6B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=85, suite="full-1-npu-a3", nightly=True)

_CAUSAL_LM_MODEL = os.environ.get("TEST_MODEL_NAME", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH)
_SEQCLS_MODEL = os.environ.get(
    "TEST_CLASSIFICATION_BASE_MODEL", QWEN3_0_6B_WEIGHTS_PATH
)


class TestCausalLMScoring(CustomTestCase):
    """Test CausalLM scoring via Engine on NPU.

    [Test Category] Feature
    [Test Target] Engine.score CausalLM
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_CAUSAL_LM_MODEL,
            attention_backend="ascend",
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_score_batch_sizes(self):
        label_token_ids = [1, 2, 3]
        for n in [1, 2, 4, 8]:
            with self.subTest(n=n):
                scores = self.engine.score(
                    query="The test was",
                    items=[f"test {i}" for i in range(n)],
                    label_token_ids=label_token_ids,
                    apply_softmax=True,
                ).scores
                self.assertEqual(len(scores), n)
                for row in scores:
                    self.assertEqual(len(row), len(label_token_ids))
                    self.assertTrue(all(isinstance(v, float) for v in row))
                    self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_empty_items(self):
        result = self.engine.score(
            query="Test query", items=[], label_token_ids=[1, 2], apply_softmax=True
        )
        self.assertEqual(len(result.scores), 0)
        self.assertEqual(result.prompt_tokens, 0)

    def test_score_without_softmax(self):
        scores = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad", "Neutral"],
            label_token_ids=[1, 2, 3],
            apply_softmax=False,
        ).scores
        self.assertEqual(len(scores), 3)
        for row in scores:
            self.assertEqual(len(row), 3)
            for v in row:
                self.assertIsInstance(v, (int, float))

    def test_score_varying_label_token_sets(self):
        for n_labels in [1, 2, 4, 8]:
            with self.subTest(n_labels=n_labels):
                scores = self.engine.score(
                    query="Choose:",
                    items=["Option A", "Option B"],
                    label_token_ids=list(range(1, n_labels + 1)),
                    apply_softmax=True,
                ).scores
                self.assertEqual(len(scores), 2)
                for row in scores:
                    self.assertEqual(len(row), n_labels)
                    self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_unicode(self):
        scores = self.engine.score(
            query="选择最佳选项：",
            items=["选项A", "选项B", "选项C"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        ).scores
        self.assertEqual(len(scores), 3)
        for row in scores:
            self.assertAlmostEqual(sum(row), 1.0, places=6)

    def test_score_deterministic(self):
        kwargs = dict(query="Choose:", items=["A", "B", "C"], label_token_ids=[1, 2, 3])
        scores_a = self.engine.score(**kwargs).scores
        scores_b = self.engine.score(**kwargs).scores
        self.assertEqual(len(scores_a), len(scores_b))
        for row_a, row_b in zip(scores_a, scores_b):
            self.assertEqual(len(row_a), len(row_b))
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=5)


class TestSeqClsScoring(CustomTestCase):
    """Test SequenceClassification scoring via Engine on NPU.

    [Test Category] Feature
    [Test Target] Engine.score SeqCls
    """

    NUM_LABELS = 2

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            attention_backend="ascend",
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": cls.NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_score_shape(self):
        scores = self.engine.score(
            query="Rate each option:",
            items=["Option A", "Option B"],
            apply_softmax=True,
        ).scores
        self.assertEqual(len(scores), 2)
        for i, row in enumerate(scores):
            self.assertEqual(len(row), self.NUM_LABELS)
            self.assertAlmostEqual(sum(row), 1.0, places=5)
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_score_single_item_edge_case(self):
        scores = self.engine.score(
            query="Evaluate:", items=["Only item"], apply_softmax=True
        ).scores
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(scores[0]), self.NUM_LABELS)
        self.assertAlmostEqual(sum(scores[0]), 1.0, places=5)

    def test_score_without_softmax(self):
        scores = self.engine.score(
            query="Evaluate:", items=["Alpha", "Beta"], apply_softmax=False
        ).scores
        self.assertEqual(len(scores), 2)
        for row in scores:
            self.assertEqual(len(row), self.NUM_LABELS)
            for v in row:
                self.assertIsInstance(v, (int, float))

    def test_score_deterministic(self):
        kwargs = dict(query="Evaluate:", items=["alpha", "beta", "gamma"])
        scores1 = self.engine.score(**kwargs).scores
        scores2 = self.engine.score(**kwargs).scores
        self.assertEqual(len(scores1), len(scores2))
        for s1, s2 in zip(scores1, scores2):
            for v1, v2 in zip(s1, s2):
                self.assertAlmostEqual(v1, v2, places=1)


if __name__ == "__main__":
    unittest.main()
