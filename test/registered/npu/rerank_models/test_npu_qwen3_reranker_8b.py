import gc
import multiprocessing as mp
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import get_device
from sglang.test.ascend.test_ascend_utils import QWEN3_RERANKER_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import TEST_RERANK_QUERY_DOCS, SRTRunner
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)

MODELS = [
    (QWEN3_RERANKER_8B_WEIGHTS_PATH, 1, 1e-2),
]
ATTENTION_BACKEND = ["ascend"]
TORCH_DTYPES = [torch.bfloat16]

# Qwen3-Reranker is a generative (CausalLM) reranker, not a
# SequenceClassification cross-encoder: score = sigmoid(yes_logit - no_logit)
# from the next-token logits of the "yes"/"no" tokens. It must be scored via
# engine.score() with label_token_ids=[yes, no]; the SRTRunner "cross_encoder"
# path returns full-vocabulary logits per pair (a list, not a scalar).

DEFAULT_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def build_rerank_prompts(tokenizer, query, documents, instruct=DEFAULT_INSTRUCT):
    """Render one Qwen3-Reranker chat prompt per (query, document) pair."""
    prompts = []
    for doc in documents:
        messages = [
            {"role": "system", "content": RERANKER_SYSTEM},
            {
                "role": "user",
                "content": f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}",
            },
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return prompts


class TestQwen3Reranker8B(CustomTestCase):
    """Validate Qwen3-Reranker-8B scores from SGLang against HuggingFace.

    Qwen3-Reranker is a generative (CausalLM) reranker, so it is scored through
    the decoder logprob path (engine.score with yes/no label tokens).

    [Test Category] Model
    [Test Target] Qwen/Qwen3-Reranker-8B
    """

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _hf_scores(self, model_path, tokenizer, prompts, yes_id, no_id, torch_dtype):
        """Reference scores computed with HuggingFace (last-token yes/no logits)."""
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(get_device())
        model.eval()
        scores = []
        try:
            with torch.no_grad():
                for prompt in prompts:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    logits = model(**inputs).logits[:, -1, :]
                    # softmax over [yes, no] -> p(yes) == sigmoid(yes - no)
                    probs = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)
                    scores.append(probs[0, 0].item())
        finally:
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        return scores

    def _srt_scores(
        self,
        model_path,
        prompts,
        yes_id,
        no_id,
        tp_size,
        torch_dtype,
        attention_backend,
    ):
        """SGLang scores via engine.score() (decoder logprob path)."""
        # model_type="generation" makes engine.score use next-token logprobs.
        with SRTRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=tp_size,
            attention_backend=attention_backend,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
        ) as srt_runner:
            result = srt_runner.engine.score(
                query="",
                items=prompts,
                label_token_ids=[yes_id, no_id],
                apply_softmax=True,
            )
        # p_yes (apply_softmax over [yes, no]) is the Qwen3-Reranker score.
        return [row[0] for row in result.scores]

    def assert_scores_close(
        self,
        query_docs,
        model_path,
        tp_size,
        torch_dtype,
        score_tolerance,
        attention_backend,
    ) -> None:
        query = query_docs["query"]
        documents = query_docs["documents"]

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
        prompts = build_rerank_prompts(tokenizer, query, documents)

        # Run HF first, then SRT, to release NPU memory between backends.
        hf_scores = self._hf_scores(
            model_path, tokenizer, prompts, yes_id, no_id, torch_dtype
        )
        srt_scores = self._srt_scores(
            model_path, prompts, yes_id, no_id, tp_size, torch_dtype, attention_backend
        )

        print(f"[query] {query}")
        for i, prompt in enumerate(prompts):
            print(f"  [prompt {i}] {prompt[:120]}...")
        print(f"hf_scores:  {hf_scores}")
        print(f"srt_scores: {srt_scores}")

        self.assertEqual(len(hf_scores), len(srt_scores), "score count mismatch")
        for i, (h, s) in enumerate(zip(hf_scores, srt_scores)):
            diff = abs(h - s)
            print(f"  [{i}] hf={h:.6f} srt={s:.6f} diff={diff:.6e}")
            self.assertLess(
                diff,
                score_tolerance,
                f"score {i}: |hf {h} - srt {s}| = {diff} >= {score_tolerance}",
            )

    def test_rerank_scores(self):
        for model, tp_size, score_tolerance in MODELS:
            for attention_backend in ATTENTION_BACKEND:
                for torch_dtype in TORCH_DTYPES:
                    for query_docs in TEST_RERANK_QUERY_DOCS:
                        with self.subTest(model=model, query=query_docs["query"]):
                            self.assert_scores_close(
                                query_docs,
                                model,
                                tp_size,
                                torch_dtype,
                                score_tolerance,
                                attention_backend,
                            )


if __name__ == "__main__":
    unittest.main()
