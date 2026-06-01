import multiprocessing as mp
import unittest

from sglang.srt.entrypoints.openai.protocol import EmbeddingRequest
from sglang.srt.managers.io_struct import EmbeddingReqInput, TokenizedEmbeddingReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=150, suite="nightly-2-npu-a3", nightly=True)


class TestNPUEmbeddingLoRASupport(unittest.TestCase):
    """Test LoRA support in embedding request structures on NPU.

    [Test Category] Feature
    [Test Target] EmbeddingReqInput LoRA fields, embedding LoRA normalization
    """

    def test_embedding_lora_fields(self):
        req = EmbeddingReqInput(
            text=["Hello", "World"], lora_path="my-adapter", lora_id=["id1", "id2"]
        )
        self.assertIsNotNone(req.lora_path)
        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, ["my-adapter", "my-adapter"])
        self.assertEqual(req[0].lora_path, "my-adapter")
        self.assertEqual(req[1].lora_id, "id2")

        req = EmbeddingReqInput(text=["Hello", "World", "Test"], lora_path=["adapter1"])
        with self.assertRaises(ValueError):
            req.normalize_batch_and_arguments()

        tokenized = TokenizedEmbeddingReqInput(
            input_text="Hello",
            input_ids=[1, 2, 3],
            image_inputs={},
            token_type_ids=[],
            sampling_params=SamplingParams(),
            lora_id="my-lora-id",
        )
        self.assertEqual(tokenized.lora_id, "my-lora-id")
        self.assertEqual(
            EmbeddingRequest(
                input="Hello", model="test", lora_path="adapter"
            ).lora_path,
            "adapter",
        )

    def test_embedding_lora_path_normalization(self):
        req = EmbeddingReqInput(
            text=["Text1", "Text2", "Text3"], lora_path="single-adapter"
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(
            req.lora_path, ["single-adapter", "single-adapter", "single-adapter"]
        )

    def test_embedding_lora_id_normalization(self):
        req = EmbeddingReqInput(
            text=["A", "B"], lora_path="adapter", lora_id="shared-id"
        )
        req.normalize_batch_and_arguments()
        # lora_id is kept as the original string value
        self.assertEqual(req.lora_id, "shared-id")

    def test_embedding_request_without_lora(self):
        req = EmbeddingReqInput(text=["Hello", "World"], lora_path=None)
        self.assertIsNone(req.lora_path)
        req.normalize_batch_and_arguments()
        self.assertIsNone(req.lora_path)

    def test_embedding_lora_mixed_batch(self):
        req = EmbeddingReqInput(
            text=["Text1", "Text2"], lora_path=["adapter1", "adapter2"]
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req.lora_path, ["adapter1", "adapter2"])


class TestNPUEmbeddingLoRAValidation(unittest.TestCase):
    """Test embedding LoRA field validation on NPU.

    [Test Category] Feature
    [Test Target] embedding request LoRA fields, validation
    """

    def test_embedding_request_lora_validation(self):
        req = EmbeddingReqInput(text=["Test"], lora_path="valid-adapter")
        self.assertIsNotNone(req.lora_path)

    def test_embedding_lora_path_length_mismatch(self):
        req = EmbeddingReqInput(
            text=["A", "B", "C"], lora_path=["adapter1", "adapter2"]
        )
        with self.assertRaises(ValueError):
            req.normalize_batch_and_arguments()

    def test_embedding_request_structure(self):
        req = EmbeddingRequest(
            input="Test input", model="test-model", lora_path="test-adapter"
        )
        self.assertEqual(req.input, "Test input")
        self.assertEqual(req.model, "test-model")
        self.assertEqual(req.lora_path, "test-adapter")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main()
