import json
import unittest

import requests
import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=100, suite="full-1-npu-a3", nightly=True)

_SEQCLS_MODEL = QWEN3_0_6B_WEIGHTS_PATH
_NUM_LABELS = 4


class TestPooledHiddenStatesEngine(CustomTestCase):
    """Test return_pooled_hidden_states through the Engine Python API on NPU.

    [Test Category] Feature
    [Test Target] return_pooled_hidden_states
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(
            model_path=_SEQCLS_MODEL,
            disable_radix_cache=True,
            json_model_override_args=json.dumps(
                {
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "num_labels": _NUM_LABELS,
                }
            ),
            mem_fraction_static=0.15,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()
        torch.cuda.empty_cache()

    def test_phs_returned_when_requested(self):
        result = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad"],
            return_pooled_hidden_states=True,
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        self.assertEqual(len(result.pooled_hidden_states), 2)
        for phs in result.pooled_hidden_states:
            self.assertIsInstance(phs, torch.Tensor)
            self.assertEqual(phs.dim(), 1)
            self.assertGreater(phs.shape[0], 0)

    def test_phs_none_when_not_requested(self):
        result = self.engine.score(
            query="Rate each:",
            items=["Good", "Bad"],
            return_pooled_hidden_states=False,
        )
        self.assertIsNone(result.pooled_hidden_states)

    def test_phs_shape_is_consistent(self):
        result = self.engine.score(
            query="Evaluate:",
            items=["Alpha", "Beta", "Gamma"],
            return_pooled_hidden_states=True,
        )
        self.assertIsNotNone(result.pooled_hidden_states)
        dims = {phs.shape[0] for phs in result.pooled_hidden_states}
        self.assertEqual(len(dims), 1, "All PHS vectors must share the same hidden dim")
        self.assertGreater(dims.pop(), 0)

    def test_phs_count_matches_items(self):
        for n in [1, 3, 5]:
            with self.subTest(n=n):
                result = self.engine.score(
                    query="Classify:",
                    items=[f"Item {i}" for i in range(n)],
                    return_pooled_hidden_states=True,
                )
                self.assertIsNotNone(result.pooled_hidden_states)
                self.assertEqual(len(result.pooled_hidden_states), n)

    def test_phs_on_cpu(self):
        result = self.engine.score(
            query="Check device:",
            items=["Test"],
            return_pooled_hidden_states=True,
        )
        for phs in result.pooled_hidden_states:
            self.assertEqual(str(phs.device), "cpu")

    def test_phs_deterministic(self):
        kwargs = dict(
            query="Evaluate:", items=["A", "B"], return_pooled_hidden_states=True
        )
        phs1 = self.engine.score(**kwargs).pooled_hidden_states
        phs2 = self.engine.score(**kwargs).pooled_hidden_states
        for t1, t2 in zip(phs1, phs2):
            self.assertTrue(
                torch.allclose(t1, t2, atol=1e-5),
                "Pooled hidden states differ across identical requests",
            )

    def test_scores_unaffected_by_phs_flag(self):
        kwargs = dict(query="Rate:", items=["X", "Y", "Z"], apply_softmax=True)
        scores_without = self.engine.score(
            **kwargs, return_pooled_hidden_states=False
        ).scores
        scores_with = self.engine.score(
            **kwargs, return_pooled_hidden_states=True
        ).scores
        self.assertEqual(len(scores_without), len(scores_with))
        for row_a, row_b in zip(scores_without, scores_with):
            for a, b in zip(row_a, row_b):
                self.assertAlmostEqual(a, b, places=2)


class TestPooledHiddenStatesHTTP(CustomTestCase):
    """Test HTTP integration: /v1/score with return_pooled_hidden_states on NPU.

    [Test Category] Feature
    [Test Target] return_pooled_hidden_states HTTP API
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _SEQCLS_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",
                "--attention-backend",
                "ascend",
                "--json-model-override-args",
                json.dumps(
                    {
                        "architectures": ["Qwen3ForSequenceClassification"],
                        "num_labels": _NUM_LABELS,
                    }
                ),
                "--mem-fraction-static",
                "0.15",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(self.base_url + "/v1/score", json=payload)

    def test_phs_in_response_json(self):
        resp = self._post(
            {
                "query": "Rate each:",
                "items": ["Good", "Bad"],
                "return_pooled_hidden_states": True,
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        phs = body.get("pooled_hidden_states")
        self.assertIsNotNone(phs)
        self.assertEqual(len(phs), 2)
        for item_phs in phs:
            self.assertIsInstance(item_phs, list)
            self.assertGreater(len(item_phs), 0)
            for v in item_phs:
                self.assertIsInstance(v, float)

    def test_phs_absent_when_not_requested(self):
        resp = self._post(
            {
                "query": "Rate each:",
                "items": ["Good"],
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body.get("pooled_hidden_states"))

    def test_phs_matches_item_count(self):
        items = ["A", "B", "C", "D"]
        resp = self._post(
            {
                "query": "Classify:",
                "items": items,
                "return_pooled_hidden_states": True,
                "model": self.model,
            }
        )
        self.assertEqual(resp.status_code, 200)
        phs = resp.json()["pooled_hidden_states"]
        self.assertEqual(len(phs), len(items))


if __name__ == "__main__":
    unittest.main()
