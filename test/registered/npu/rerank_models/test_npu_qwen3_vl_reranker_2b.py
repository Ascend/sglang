import gc
import multiprocessing as mp
import os
import unittest

import requests
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from sglang.srt.utils import get_device
from sglang.test.ascend.test_ascend_utils import (
    IMAGES_023_PATH,
    QWEN3_VL_RERANKER_2B_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_multimodal_utils import launch_server
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import TEST_RERANK_QUERY_DOCS, SRTRunner
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)

MODELS = [
    (QWEN3_VL_RERANKER_2B_WEIGHTS_PATH, 1, 1e-2),
]
ATTENTION_BACKEND = ["ascend"]
TORCH_DTYPES = [torch.bfloat16]

IMAGES = IMAGES_023_PATH

# Qwen3-VL-Reranker is a *generative* reranker (a CausalLM, not a
# SequenceClassification cross-encoder). It scores a (query, document) pair by
# reading the next-token logits of the "yes"/"no" tokens after a chat-template
# prompt, then normalising: score = sigmoid(yes_logit - no_logit).
#
# The correct SGLang path is engine.score() with label_token_ids=[yes, no],
# mirroring serving_rerank's "text_decoder" backend.

DEFAULT_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def build_rerank_prompts(tokenizer, query, documents, instruct=DEFAULT_INSTRUCT):
    """Render one Qwen3-VL-Reranker chat prompt per (query, document) pair."""
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


class TestQwen3VLReranker2B(CustomTestCase):
    """Validate Qwen3-VL-Reranker-2B scores from SGLang against HuggingFace.

    Unlike BAAI/bge-reranker (a true SequenceClassification cross-encoder),
    Qwen3-VL-Reranker is a generative (CausalLM) reranker. It must be scored
    through the decoder logprob path (engine.score with yes/no label tokens),
    not the cross-encoder embedding path.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-VL-Reranker-2B
    """

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _hf_scores(self, model_path, processor, prompts, yes_id, no_id, torch_dtype):
        """Reference scores computed with HuggingFace (last-token yes/no logits)."""
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(get_device())
        model.eval()
        scores = []
        try:
            with torch.no_grad():
                for prompt in prompts:
                    inputs = processor(text=[prompt], return_tensors="pt").to(
                        model.device
                    )
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
        # model_type="generation" -> is_embedding=False, so the engine runs in
        # generation mode and engine.score uses next-token logprobs.
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
        # apply_softmax=True over [yes, no] logprobs -> [p_yes, p_no] (sums to 1).
        # p_yes == p_yes / (p_yes + p_no) == the Qwen3-VL-Reranker score.
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

        processor = AutoProcessor.from_pretrained(model_path)
        yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
        no_id = processor.tokenizer.convert_tokens_to_ids("no")
        prompts = build_rerank_prompts(processor.tokenizer, query, documents)

        # HF first, then SRT: load them sequentially so NPU memory is released
        # between the two backends.
        hf_scores = self._hf_scores(
            model_path, processor, prompts, yes_id, no_id, torch_dtype
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


class TestQwen3VLReranker2BMultimodal(CustomTestCase):
    """Validate Qwen3-VL-Reranker-2B multimodal (text+image) reranking.

    Compares HuggingFace scores against SGLang /v1/rerank API scores
    for both text-only and text+image inputs.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-VL-Reranker-2B
    """

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _hf_multimodal_scores(self, model_path, processor, torch_dtype, query, documents):
        """HF scores for multimodal reranking (text + image)."""
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(get_device())
        model.eval()
        scores = []
        try:
            with torch.no_grad():
                for doc in documents:
                    # Build messages with image if doc is a list (multimodal content)
                    if isinstance(doc, list):
                        content = doc
                    else:
                        content = [{"type": "text", "text": doc}]

                    messages = [
                        {"role": "system", "content": RERANKER_SYSTEM},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"<Instruct>: {DEFAULT_INSTRUCT}\n<Query>: {query}\n<Document>: "},
                                *content,
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    inputs = processor(
                        text=[prompt],
                        images=[IMAGES],
                        return_tensors="pt",
                    ).to(model.device)
                    logits = model(**inputs).logits[:, -1, :]
                    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
                    no_id = processor.tokenizer.convert_tokens_to_ids("no")
                    probs = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)
                    scores.append(probs[0, 0].item())
        finally:
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        return scores

    def _srt_multimodal_scores(self, base_url, query, documents):
        """SRT scores via /v1/rerank HTTP API."""
        payload = {
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        response = requests.post(
            f"{base_url}/v1/rerank",
            json=payload,
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, f"Rerank API failed: {response.text}")
        results = response.json()
        if isinstance(results, dict) and "message" in results:
            self.fail(f"Rerank API error: {results['message']}")
        # Sort by index to match HF order
        results.sort(key=lambda r: r["index"])
        return [r["score"] for r in results]

    def assert_multimodal_scores_close(
        self, model_path, torch_dtype, score_tolerance, query, documents
    ):
        processor = AutoProcessor.from_pretrained(model_path)

        hf_scores = self._hf_multimodal_scores(
            model_path, processor, torch_dtype, query, documents
        )

        # Launch SRT server with VL reranker chat template
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..",
            "examples", "chat_template", "qwen3_vl_reranker.jinja",
        )
        template_path = os.path.abspath(template_path)

        process, base_url = launch_server(
            model_path,
            extra_args=[
                "--chat-template", template_path,
                "--disable-radix-cache",
                "--tp-size", "1",
            ],
        )
        try:
            srt_scores = self._srt_multimodal_scores(base_url, query, documents)
        finally:
            process.terminate()
            process.wait(timeout=30)

        print(f"[query] {query}")
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

    def test_multimodal_rerank(self):
        for model, tp_size, score_tolerance in MODELS:
            for torch_dtype in TORCH_DTYPES:
                # Text-only: backward compatibility
                with self.subTest(model=model, mode="text"):
                    self.assert_multimodal_scores_close(
                        model,
                        torch_dtype,
                        score_tolerance,
                        query="How many people live in Berlin?",
                        documents=[
                            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
                            "Berlin is well known for its museums.",
                        ],
                    )

                # Text + image: multimodal reranking
                with self.subTest(model=model, mode="image"):
                    self.assert_multimodal_scores_close(
                        model,
                        torch_dtype,
                        score_tolerance,
                        query="What is shown in the image?",
                        documents=[
                            [
                                {"type": "image_url", "image_url": {"url": IMAGES}},
                            ],
                            "A busy city street with cars and pedestrians.",
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
