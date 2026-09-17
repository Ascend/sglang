import gc
import math
import multiprocessing as mp
import os
import unittest

import jinja2
import requests
import torch
from jinja2.sandbox import ImmutableSandboxedEnvironment
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

# The image document lands at mid confidence (score ~ 0.25), where the sigmoid
# is steepest and NPU-bf16 vs HF eager-bf16 noise is largest; the image path
# also adds vision-encoder projections. Relax the tolerance vs text-only cases.
MULTIMODAL_IMAGE_SCORE_TOLERANCE = 2e-2

IMAGES = IMAGES_023_PATH

# Qwen3-VL-Reranker is a generative (CausalLM) reranker: score = sigmoid(yes_logit
# - no_logit) from the next-token logits of the "yes"/"no" tokens after a
# chat-template prompt. Score via engine.score() with label_token_ids=[yes, no],
# mirroring serving_rerank's "text_decoder" backend.

DEFAULT_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANKER_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


def _load_vl_reranker_template():
    """Load the shared qwen3_vl_reranker.jinja used by the SRT /v1/rerank server.

    Ensures the HF reference and SGLang score byte-identical prompts.
    """
    template_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "examples",
        "chat_template",
        "qwen3_vl_reranker.jinja",
    )
    with open(os.path.abspath(template_path), encoding="utf-8") as f:
        template_text = f.read()
    env = ImmutableSandboxedEnvironment(
        loader=jinja2.BaseLoader(),
        autoescape=False,
        undefined=jinja2.Undefined,
    )
    return env.from_string(template_text)


def _to_template_content(content):
    """Normalize query/document into the template content-part list.

    Mirrors serving_rerank._content_to_template_list so the HF reference and
    the SRT server render byte-identical prompts.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    result = []
    for part in content:
        if isinstance(part, dict):
            part_type = part.get("type")
            if part_type == "text":
                result.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                result.append({"type": "image"})
            elif part_type == "video_url":
                result.append({"type": "video"})
            else:
                result.append(part)
        else:
            result.append(part)
    return result


def render_vl_reranker_prompt(query, document, instruct=DEFAULT_INSTRUCT):
    """Render one (query, document) reranker prompt with the shared template."""
    return _load_vl_reranker_template().render(
        query=_to_template_content(query),
        document=_to_template_content(document),
        instruct=instruct,
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

    Qwen3-VL-Reranker is a generative (CausalLM) reranker, so it is scored
    through the decoder logprob path (engine.score with yes/no label tokens).

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
        # p_yes (apply_softmax over [yes, no]) is the Qwen3-VL-Reranker score.
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

        # Run HF first, then SRT, to release NPU memory between backends.
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

    def _hf_multimodal_scores(
        self, model_path, processor, torch_dtype, query, documents
    ):
        """HF scores for multimodal reranking (text + image).

        Prompts use the same qwen3_vl_reranker.jinja template as the SRT server.
        """
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch_dtype
        ).to(get_device())
        model.eval()
        scores = []
        try:
            with torch.no_grad():
                for i, doc in enumerate(documents):
                    prompt = render_vl_reranker_prompt(query, doc)
                    # Pass images only for multimodal (list) documents.
                    images = [IMAGES] if isinstance(doc, list) else None
                    inputs = processor(
                        text=[prompt],
                        images=images,
                        return_tensors="pt",
                    ).to(model.device)
                    logits = model(**inputs).logits[:, -1, :]
                    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
                    no_id = processor.tokenizer.convert_tokens_to_ids("no")
                    probs = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)
                    score = probs[0, 0].item()
                    scores.append(score)
                    # Log provenance (image vs text) per doc so a failure can
                    # be traced to the exact image/document driving the score.
                    doc_tag = "image" if images is not None else "text"
                    print(
                        f"  [HF {i}] {doc_tag}: score={score:.6f} "
                        f"p_yes={probs[0, 0].item():.6f} p_no={probs[0, 1].item():.6f}"
                    )
        finally:
            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        return scores

    def _srt_multimodal_scores(self, base_url, processor, query, documents):
        """SRT scores via the native /generate endpoint with token_ids_logprob.

        ``token_ids_logprob=[yes, no]`` computes full-vocabulary logits for
        exactly those tokens (same mechanism as ``engine.score``), avoiding the
        score collapse seen with the /v1/rerank top-k logprob path.
        """
        yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
        no_id = processor.tokenizer.convert_tokens_to_ids("no")
        scores = []
        for i, doc in enumerate(documents):
            prompt = render_vl_reranker_prompt(query, doc)
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0},
                "return_logprob": True,
                "token_ids_logprob": [yes_id, no_id],
                "logprob_start_len": 0,
            }
            # Image parts become URL strings in image_data for the rendered
            # <|vision_start|><|image_pad|><|vision_end|> placeholder.
            has_image = False
            if isinstance(doc, list):
                image_data = [
                    part["image_url"]["url"]
                    for part in doc
                    if isinstance(part, dict)
                    and part.get("type") == "image_url"
                    and part.get("image_url")
                ]
                if image_data:
                    payload["image_data"] = image_data
                    has_image = True
            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                timeout=120,
            )
            self.assertEqual(
                response.status_code, 200, f"/generate failed: {response.text}"
            )
            meta = response.json()["meta_info"]
            # With max_new_tokens=0 (prefill-only), the yes/no logprobs come
            # from the full-context prefill forward, matching engine.score.
            p_yes = 0.0
            p_no = 0.0
            for item in (meta.get("output_token_ids_logprobs") or [[]])[0]:
                logprob, token_id = item[0], item[1]
                if token_id == yes_id:
                    p_yes = math.exp(logprob)
                elif token_id == no_id:
                    p_no = math.exp(logprob)
            denom = p_yes + p_no
            score = p_yes / denom if denom > 0 else 0.0
            scores.append(score)
            # Log provenance (image vs text) per doc so a failure can be
            # traced to the exact image/document driving the score.
            doc_tag = "image" if has_image else "text"
            print(
                f"  [SRT {i}] {doc_tag}: score={score:.6f} "
                f"p_yes={p_yes:.6f} p_no={p_no:.6f}"
            )
        return scores

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
            "..",
            "..",
            "..",
            "..",
            "examples",
            "chat_template",
            "qwen3_vl_reranker.jinja",
        )
        template_path = os.path.abspath(template_path)

        process, base_url = launch_server(
            model_path,
            extra_args=[
                "--chat-template",
                template_path,
                "--disable-radix-cache",
                "--tp-size",
                "1",
            ],
        )
        try:
            srt_scores = self._srt_multimodal_scores(
                base_url, processor, query, documents
            )
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
                            "The Eiffel Tower is located in Paris, France and stands 330 meters tall.",
                        ],
                    )

                # Text + image: multimodal reranking
                with self.subTest(model=model, mode="image"):
                    self.assert_multimodal_scores_close(
                        model,
                        torch_dtype,
                        MULTIMODAL_IMAGE_SCORE_TOLERANCE,
                        # The image document cannot be pushed to an extreme
                        # confidence score, so relax the tolerance here.
                        query="Does the image show a city street with cars and pedestrians?",
                        documents=[
                            [
                                {"type": "image_url", "image_url": {"url": IMAGES}},
                            ],
                            "The Eiffel Tower is located in Paris, France and stands 330 meters tall.",
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
