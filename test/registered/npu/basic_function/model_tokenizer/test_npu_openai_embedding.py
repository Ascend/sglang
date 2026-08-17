import json
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

register_npu_ci(est_time=50, suite="full-1-npu-a3", nightly=True)


class TestOpenAIEmbedding(CustomTestCase):
    """
    Testcase：Verify the correctness of the embeddings function of gte_Qwen2-1.5B-instruct
    when client.embeddings.create API of openai is called for different inputs

    [Test Category] Parameter
    [Test Target] --is-embedding
    """

    @classmethod
    def setUpClass(cls):
        cls.model = GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Configure embedding-specific args
        other_args = [
            "--is-embedding",
            "--enable-metrics",
            "--attention-backend",
            "ascend",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_embedding_single(self):
        """Test single embedding request"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input="Hello world")
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

    def test_embedding_batch(self):
        """Test batch embedding request"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model, input=["Hello world", "Test text"]
        )
        self.assertEqual(len(response.data), 2)
        self.assertTrue(len(response.data[0].embedding) > 0)
        self.assertTrue(len(response.data[1].embedding) > 0)

    def test_embedding_single_batch_str(self):
        """Test embedding with a List[str] and length equals to 1"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input=["Hello world"])
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

    def test_embedding_single_int_list(self):
        """Test embedding with a List[int] or List[List[int]]]"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model,
            input=[[15339, 314, 703, 284, 612, 262, 10658, 10188, 286, 2061]],
        )
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

        response = client.embeddings.create(
            model=self.model,
            input=[15339, 314, 703, 284, 612, 262, 10658, 10188, 286, 2061],
        )
        self.assertEqual(len(response.data), 1)
        self.assertTrue(len(response.data[0].embedding) > 0)

    def test_empty_string_embedding(self):
        """Test embedding an empty string."""

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Text embedding example with empty string
        text = ""
        # Expect a BadRequestError for empty input
        with self.assertRaises(openai.BadRequestError) as cm:
            client.embeddings.create(
                model=self.model,
                input=text,
            )
        # check the status code
        self.assertEqual(cm.exception.status_code, 400)

    def test_embedding_with_dimensions_parameter(self):
        """Test that non-Matryoshka models reject dimensions parameter."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test that specifying dimensions fails for non-Matryoshka models
        with self.assertRaises(openai.BadRequestError) as cm:
            client.embeddings.create(
                model=self.model, input="Hello world", dimensions=512
            )

        self.assertEqual(cm.exception.status_code, 400)

    def test_model_param_not_used_for_selection(self):
        """The request's model name is not validated or used for model
        selection — the response echoes the server's model path."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        normal = client.embeddings.create(model=self.model, input="Hello world")
        bogus = client.embeddings.create(
            model="nonexistent-model-name", input="Hello world"
        )

        self.assertEqual(
            normal.model,
            bogus.model,
            "response.model must echo the server model path regardless of the request model",
        )


class TestMatryoshkaEmbeddingModel(CustomTestCase):
    """Test class for Model that supports Matryoshka embedding functionality, using OpenAI API."""

    @classmethod
    def setUpClass(cls):
        cls.model = GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.matryoshka_dims = [128, 256, 512, 768, 1024]

        # Configure embedding-specific args with Matryoshka support via json_model_override_args
        matryoshka_config = {
            "is_matryoshka": True,
            "matryoshka_dimensions": cls.matryoshka_dims,
        }
        other_args = [
            "--is-embedding",
            "--enable-metrics",
            "--json-model-override-args",
            json.dumps(matryoshka_config),
            "--attention-backend",
            "ascend",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)

    def test_matryoshka_embedding_valid_dimensions(self):
        """Test Matryoshka embedding with valid dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test with various valid dimensions
        for dimensions in self.matryoshka_dims:
            with self.subTest(dimensions=dimensions):
                response = client.embeddings.create(
                    model=self.model, input="Hello world", dimensions=dimensions
                )
                self.assertEqual(len(response.data), 1)
                self.assertEqual(len(response.data[0].embedding), dimensions)

    def test_matryoshka_embedding_batch_same_dimensions(self):
        """Test Matryoshka embedding with batch input and same dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(
            model=self.model,
            input=["Hello world", "Test text", "Another example"],
            dimensions=256,
        )

        self.assertEqual(len(response.data), 3)
        for embedding_data in response.data:
            self.assertEqual(len(embedding_data.embedding), 256)

    def test_matryoshka_embedding_no_dimensions(self):
        """Test embedding without specifying dimensions (should use full size)."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(model=self.model, input="Hello world")

        self.assertEqual(len(response.data), 1)

        # Should return full embedding size when no dimensions specified
        self.assertEqual(len(response.data[0].embedding), 1536)

    def test_matryoshka_embedding_invalid_dimensions(self):
        """Test Matryoshka embedding with invalid dimensions."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        for dimensions in [100, 0, -1, 10000]:
            with self.assertRaises(openai.BadRequestError) as cm:
                client.embeddings.create(
                    model=self.model,
                    input="Hello world",
                    dimensions=dimensions,
                )
            self.assertEqual(cm.exception.status_code, 400)

    def test_matryoshka_embedding_l2_norm(self):
        """Value-level: the truncated vector is L2-normalized AFTER truncation
        (pooler.py:192-199) — the result must be a unit vector."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.embeddings.create(
            model=self.model, input="Hello world", dimensions=256
        )
        embedding = response.data[0].embedding
        norm = sum(x * x for x in embedding) ** 0.5
        self.assertAlmostEqual(
            norm,
            1.0,
            places=3,
            msg="truncated embedding must be L2-normalized to unit norm",
        )

    def test_matryoshka_embedding_semantic_preserved(self):
        """Value-level: semantic relations survive truncation — similar texts
        stay closer than dissimilar texts in the 256-dim space."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        def embed(text):
            return (
                client.embeddings.create(model=self.model, input=text, dimensions=256)
                .data[0]
                .embedding
            )

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb)

        a1 = embed("The weather is sunny today")
        a2 = embed("It is a bright and sunny day")
        b = embed("Quantum physics explains subatomic particles")

        self.assertGreater(
            cosine(a1, a2),
            cosine(a1, b),
            "similar texts must be closer than dissimilar texts after truncation",
        )


if __name__ == "__main__":
    unittest.main()
