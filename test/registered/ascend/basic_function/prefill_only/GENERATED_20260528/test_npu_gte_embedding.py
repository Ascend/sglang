import multiprocessing as mp
import unittest

import torch

from sglang.test.ascend.test_ascend_utils import GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, get_similarities

register_npu_ci(est_time=136, suite="nightly-1-npu-a3", nightly=True)

MODEL_TO_CONFIG = {
    GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH: (1, 1e-5),
}
MODELS = [(key, *MODEL_TO_CONFIG[key]) for key in MODEL_TO_CONFIG]

TORCH_DTYPES = [torch.float16]


class TestGTEEmbedding(CustomTestCase):
    """Test GTE-Qwen2-1.5B embedding model on NPU.

    [Test Category] Model
    [Test Target] Alibaba-NLP/gte-Qwen2-1.5B-instruct
    """

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        prefill_tolerance,
    ) -> None:
        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as hf_runner:
            hf_outputs = hf_runner.forward(prompts)

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            attention_backend="ascend",
        ) as srt_runner:
            srt_outputs = srt_runner.forward(prompts)

        for i in range(len(prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))

            if len(prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < prefill_tolerance
                ), "embeddings are not all close"

    def test_prefill_logits(self):
        for model, tp_size, prefill_tolerance in MODELS:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS, model, tp_size, torch_dtype, prefill_tolerance
                )


if __name__ == "__main__":
    unittest.main()
