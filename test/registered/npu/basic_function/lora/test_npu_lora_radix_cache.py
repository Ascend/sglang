import multiprocessing as mp
import os
import unittest

from sglang import Engine
from sglang.test.ascend.lora_utils import CI_MULTI_LORA_MODELS
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

os.environ["SGLANG_ENABLE_FAST_INPUT_LOGPROBS"] = "0"
os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"

PROMPTS = [
    "AI is a field of computer science focused on",
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids.
    ### Question:
    What do you know about llamas?
    ### Answer:
    """,
]

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)

# On Ascend the KV cache is paged in 128-token pages. With
# `logprob_start_len=-1` the matcher may reuse at most len(input_ids) - 1
# tokens, so a prompt of 129 tokens reports exactly 128 cached tokens when it
# is sent a second time.
PAGE_TOKENS = 128
LONG_PROMPT_TOKENS = PAGE_TOKENS + 1
LOGPROB_TOLERANCE = 0.3
PORT_CACHE_OFF = 31100
PORT_CACHE_ON = 31101


class TestLoRARadixCache(CustomTestCase):
    """Validate that LoRA requests reuse the radix cache correctly.

    The reference is a cache-disabled SRT engine. On the cache-enabled engine we
    compare the next-token logprob of the same prompt, adapter and target token
    id, then verify that a repeated long prompt really hits 128 cached tokens
    and that a different adapter does not reuse another adapter's prefix.

    The old test compared 32 tokens of independent HF/SRT free generation. Once
    the two engines pick different tokens the histories diverge, so later
    logprobs no longer describe the same input; that comparison is not a valid
    cache oracle. It also sent `logprob_start_len=0`, which caps the reusable
    prefix at zero and made every "radix cache" request a cache miss.
    """

    def _start_engine(self, model_case, disable_radix_cache, port):
        return Engine(
            model_path=model_case.base,
            tp_size=model_case.tp_size,
            dtype="bfloat16",
            port=port,
            mem_fraction_static=0.88,
            attention_backend="ascend",
            lora_paths=[adaptor.name for adaptor in model_case.adaptors],
            max_loras_per_batch=model_case.max_loras_per_batch,
            max_loaded_loras=model_case.max_loaded_loras,
            lora_backend="csgmv",
            disable_radix_cache=disable_radix_cache,
        )

    def _generate(self, engine, prompt, lora_path, target_token_id=None):
        # A list of token ids must go through `input_ids`; `prompt` is text.
        prompt_kwargs = (
            {"input_ids": prompt} if isinstance(prompt, list) else {"prompt": prompt}
        )
        return engine.generate(
            **prompt_kwargs,
            lora_path=lora_path,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
            return_logprob=True,
            # -1 keeps the full prompt matchable; 0 would cap the prefix at 0.
            logprob_start_len=-1,
            token_ids_logprob=(
                [target_token_id] if target_token_id is not None else None
            ),
        )

    def _first_logprob(self, response):
        logprob, token_id, *_ = response["meta_info"]["output_token_logprobs"][0]
        return logprob, token_id

    def _assert_cached_tokens(self, response, expected, label):
        cached = response["meta_info"]["cached_tokens"]
        self.assertEqual(
            cached,
            expected,
            f"{label}: expected {expected} cached tokens, got {cached}",
        )

    def _assert_same_target_logprob(self, reference, response, label):
        ref_logprob, ref_token_id = reference
        token_logprobs = response["meta_info"]["output_token_ids_logprobs"]
        self.assertTrue(
            token_logprobs and token_logprobs[0],
            f"{label}: no token_ids_logprob returned for token {ref_token_id}",
        )
        hit_logprob = token_logprobs[0][0][0]
        diff = abs(ref_logprob - hit_logprob)
        print(
            f"{label}: ref={ref_logprob:.6f} hit={hit_logprob:.6f} "
            f"diff={diff:.6f} cached={response['meta_info']['cached_tokens']}"
        )
        self.assertLess(
            diff,
            LOGPROB_TOLERANCE,
            f"{label}: next-token logprob for fixed token {ref_token_id} "
            f"differs by {diff} (ref={ref_logprob}, cache-on={hit_logprob}, "
            f"tolerance={LOGPROB_TOLERANCE})",
        )

    def _build_long_input_ids(self, tokenizer, prompt):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        filler_ids = tokenizer.encode(
            " The quick brown fox jumps over the lazy dog.",
            add_special_tokens=False,
        )
        self.assertTrue(prompt_ids)
        self.assertTrue(filler_ids)
        input_ids = list(prompt_ids)
        while len(input_ids) < LONG_PROMPT_TOKENS:
            input_ids.extend(filler_ids)
        return input_ids[:LONG_PROMPT_TOKENS]

    def test_lora_radix_cache(self):
        model_case = CI_MULTI_LORA_MODELS[0]
        adaptor_names = [adaptor.name for adaptor in model_case.adaptors]
        self.assertGreaterEqual(len(adaptor_names), 2)
        first_adaptor, second_adaptor = adaptor_names[0], adaptor_names[1]

        references = {}

        # Reference arm: radix cache disabled, so every request is a full prefill.
        with self._start_engine(model_case, True, PORT_CACHE_OFF) as reference_engine:
            tokenizer = reference_engine.tokenizer_manager.tokenizer
            long_inputs = [
                self._build_long_input_ids(tokenizer, prompt) for prompt in PROMPTS
            ]
            for prompt_idx, prompt in enumerate(PROMPTS):
                for adaptor in adaptor_names:
                    short_response = self._generate(reference_engine, prompt, adaptor)
                    long_response = self._generate(
                        reference_engine, long_inputs[prompt_idx], adaptor
                    )
                    references[(prompt_idx, adaptor, "short")] = self._first_logprob(
                        short_response
                    )
                    references[(prompt_idx, adaptor, "long")] = self._first_logprob(
                        long_response
                    )

        # Cache-enabled arm.
        with self._start_engine(model_case, False, PORT_CACHE_ON) as engine:
            for prompt_idx, prompt in enumerate(PROMPTS):
                long_ids = long_inputs[prompt_idx]

                # ---- Short prompts: below one page, so a repeat is still a miss.
                for adaptor in adaptor_names:
                    short_label = (
                        f"prompt{prompt_idx + 1}/{os.path.basename(adaptor)}/short"
                    )
                    cold = self._generate(engine, prompt, adaptor)
                    self._assert_cached_tokens(cold, 0, f"{short_label}/first")
                    reference = references[(prompt_idx, adaptor, "short")]
                    warm = self._generate(
                        engine, prompt, adaptor, target_token_id=reference[1]
                    )
                    self._assert_cached_tokens(warm, 0, f"{short_label}/repeat")
                    self._assert_same_target_logprob(reference, warm, short_label)

                # ---- Long prompt: A cold / A hit, then B cold / B hit.
                first_label = (
                    f"prompt{prompt_idx + 1}/{os.path.basename(first_adaptor)}/long"
                )
                first_cold = self._generate(engine, long_ids, first_adaptor)
                self._assert_cached_tokens(first_cold, 0, f"{first_label}/first")
                first_hit = self._generate(
                    engine,
                    long_ids,
                    first_adaptor,
                    target_token_id=references[(prompt_idx, first_adaptor, "long")][1],
                )
                self._assert_cached_tokens(
                    first_hit, PAGE_TOKENS, f"{first_label}/repeat"
                )
                self._assert_same_target_logprob(
                    references[(prompt_idx, first_adaptor, "long")],
                    first_hit,
                    first_label,
                )

                second_label = (
                    f"prompt{prompt_idx + 1}/{os.path.basename(second_adaptor)}/long"
                )
                second_cold = self._generate(engine, long_ids, second_adaptor)
                # A different adapter must not reuse the first adapter's prefix.
                self._assert_cached_tokens(second_cold, 0, f"{second_label}/first")
                second_hit = self._generate(
                    engine,
                    long_ids,
                    second_adaptor,
                    target_token_id=references[(prompt_idx, second_adaptor, "long")][1],
                )
                self._assert_cached_tokens(
                    second_hit, PAGE_TOKENS, f"{second_label}/repeat"
                )
                self._assert_same_target_logprob(
                    references[(prompt_idx, second_adaptor, "long")],
                    second_hit,
                    second_label,
                )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
