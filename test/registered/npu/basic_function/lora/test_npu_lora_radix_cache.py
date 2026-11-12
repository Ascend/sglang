import multiprocessing as mp
import os
import unittest

import torch

from sglang.test.ascend.lora_utils import CI_MULTI_LORA_MODELS
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

os.environ["SGLANG_ENABLE_FAST_INPUT_LOGPROBS"] = "0"

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


os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"

PAGE_TOKENS = 128
LOGPROB_TOLERANCE = 0.3


class TestLoRARadixCache(CustomTestCase):
    @staticmethod
    def _generate(engine, prompt, adaptor, *, token_ids_logprob=None):
        return engine.generate(
            prompt,
            lora_path=adaptor.name,
            sampling_params={"max_new_tokens": 1, "temperature": 0},
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=5,
            token_ids_logprob=token_ids_logprob,
        )

    def test_lora_radix_cache(self):
        model_case = CI_MULTI_LORA_MODELS[0]
        adaptors = model_case.adaptors
        self.assertGreaterEqual(len(adaptors), 2)
        prompts = list(PROMPTS)
        # About 180 tokens: a repeated request reuses exactly one 128-token page.
        long_prefix = " ".join([PROMPTS[0]] * 20)
        prompts.append(long_prefix)
        torch_dtype = torch.bfloat16
        lora_paths = [adaptor.name for adaptor in adaptors]
        # Keep the CI BF16 setup and each model case's existing tolerance unchanged.
        runner_args = {
            "torch_dtype": torch_dtype,
            "model_type": "generation",
            "tp_size": model_case.tp_size,
            "lora_paths": lora_paths,
            "max_loras_per_batch": model_case.max_loras_per_batch,
            "max_loaded_loras": model_case.max_loaded_loras,
            "lora_backend": "csgmv",
            "attention_backend": "ascend",
            "disable_cuda_graph": False,
            "mem_fraction_static": 0.88,
        }

        # One generated token isolates the next-token distribution under a fixed
        # history. The old independent HF free-generation branch is not a
        # radix-cache oracle.
        with SRTRunner(
            model_case.base, disable_radix_cache=True, **runner_args
        ) as runner:
            off_reference = {}
            for prompt_index, prompt in enumerate(prompts):
                for adaptor_index, adaptor in enumerate(adaptors[:2]):
                    response = self._generate(runner.engine, prompt, adaptor)
                    ranked = response["meta_info"]["output_top_logprobs"][0]
                    self.assertEqual(len(ranked), 5)
                    off_reference[(prompt_index, adaptor_index)] = {
                        int(entry[1]): float(entry[0]) for entry in ranked
                    }

        # Revisit each prompt with both adapters to exercise cache hits and
        # isolation. flush_cache() makes the first request of every prompt a
        # guaranteed cold start, independent of the prompts that came before.
        with SRTRunner(
            model_case.base, disable_radix_cache=False, **runner_args
        ) as runner:
            for prompt_index, prompt in enumerate(prompts):
                runner.engine.flush_cache()
                long_case = prompt_index >= len(PROMPTS)
                seen = set()
                for adaptor_index, adaptor in (
                    (0, adaptors[0]),
                    (1, adaptors[1]),
                    (0, adaptors[0]),
                    (1, adaptors[1]),
                ):
                    key = (prompt_index, adaptor_index)
                    target_ids = list(off_reference[key])
                    cold_start = not (long_case and key in seen)
                    seen.add(key)
                    label = (
                        f"prompt{prompt_index + 1}/"
                        f"{os.path.basename(adaptor.name)}/"
                        f"{'cold' if cold_start else 'hit'}"
                    )
                    response = self._generate(
                        runner.engine, prompt, adaptor, token_ids_logprob=target_ids
                    )
                    meta = response["meta_info"]
                    cached = meta["cached_tokens"]
                    expected_cached = 0 if cold_start else PAGE_TOKENS
                    self.assertEqual(
                        cached,
                        expected_cached,
                        f"{label}: expected {expected_cached} cached tokens, "
                        f"got {cached}",
                    )

                    hit_logprobs = {
                        int(entry[1]): float(entry[0])
                        for entry in meta["output_token_ids_logprobs"][0]
                    }
                    self.assertEqual(len(hit_logprobs), len(target_ids))
                    diffs = []
                    for token_id in target_ids:
                        ref = off_reference[key][token_id]
                        got = hit_logprobs[token_id]
                        diff = abs(ref - got)
                        diffs.append(diff)
                        print(
                            f"{label}: token={token_id} ref={ref:.6f} "
                            f"hit={got:.6f} diff={diff:.6f} cached={cached}"
                        )
                    self.assertLess(
                        max(diffs),
                        LOGPROB_TOLERANCE,
                        f"{label}: max logprob diff {max(diffs)} over the top-5 "
                        f"ids exceeds tolerance {LOGPROB_TOLERANCE} "
                        f"(cached={cached})",
                    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
