import dataclasses
import random
from typing import List, Optional

import torch

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_4B_LORA_V2_WEIGHTS_PATH,
    QWEN3_4B_LORA_ZH_WEBNOVELTY_V0_0_WEIGHTS_PATH,
    QWEN3_4B_WEIGHTS_PATH,
)
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import calculate_rouge_l


@dataclasses.dataclass
class LoRAAdaptor:
    name: str
    prefill_tolerance: float = None
    decode_tolerance: float = None
    rouge_l_tolerance: float = None


@dataclasses.dataclass
class LoRAModelCase:
    base: str
    adaptors: List[LoRAAdaptor]
    tp_size: int = 1
    prefill_tolerance: float = 1e-1
    decode_tolerance: float = 1e-1
    rouge_l_tolerance: float = 1.0
    max_loras_per_batch: int = 1
    max_loaded_loras: Optional[int] = None
    skip_long_prompt: bool = False

    def __post_init__(self):
        if len(self.adaptors) > self.max_loras_per_batch:
            raise ValueError(
                f"For base '{self.base}', number of adaptors ({len(self.adaptors)}) "
                f"must be <= max_loras_per_batch ({self.max_loras_per_batch})"
            )


TORCH_DTYPES = [torch.float16]

LORA_MODELS_QWEN3 = [
    LoRAModelCase(
        base=QWEN3_4B_WEIGHTS_PATH,
        adaptors=[
            LoRAAdaptor(
                name=QWEN3_4B_LORA_V2_WEIGHTS_PATH,
                prefill_tolerance=3e-1,
            ),
            LoRAAdaptor(
                name=QWEN3_4B_LORA_ZH_WEBNOVELTY_V0_0_WEIGHTS_PATH,
                prefill_tolerance=3e-1,
            ),
        ],
        max_loras_per_batch=2,
        max_loaded_loras=64,
    ),
]

TEST_MULTIPLE_BATCH_PROMPTS = [
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
    ### Question 2:
    What do you know about llamas?
    ### Answer:
    """,
    """
    ### Instruction:
    Write a poem about the transformers Python library.
    Mention the word "large language models" in that poem.
    ### Response:
    The Transformers are large language models,
    They're used to make predictions on text.
    """,
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]


def create_multiple_batch_test_samples(
    prompts: List[str], lora_adapter_paths: List[str]
):
    random.seed(42)

    test_cases = [
        (
            [
                random.choice(prompts),
                random.choice(prompts),
                random.choice(prompts),
            ],
            [
                None,
                lora_adapter_paths[0],
                lora_adapter_paths[1],
            ],
        ),
        (
            [
                random.choice(prompts),
                random.choice(prompts),
                random.choice(prompts),
            ],
            [lora_adapter_paths[0], lora_adapter_paths[1], None],
        ),
    ]

    return test_cases


def ensure_reproducibility():
    seed = 42
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def run_lora_multiple_batch_on_model_cases(
    model_cases: List[LoRAModelCase],
    use_spec_decoding: bool = False,
    attention_backend: str = "torch_native",
    disable_cuda_graph: bool = True,
    enable_deterministic_inference: bool = False,
    disable_radix_cache: bool = True,
    enable_lora_overlap_loading: Optional[bool] = None,
):
    if not torch.npu.is_available():
        raise RuntimeError(
            "NPU device not available. Please ensure NPU environment is properly configured."
        )

    for model_case in model_cases:
        for torch_dtype in TORCH_DTYPES:
            max_new_tokens = 32
            base_path = model_case.base
            lora_adapter_paths = [a.name for a in model_case.adaptors]
            assert len(lora_adapter_paths) >= 2

            batches = create_multiple_batch_test_samples(
                TEST_MULTIPLE_BATCH_PROMPTS, lora_adapter_paths
            )

            ensure_reproducibility()
            spec_args = (
                {}
                if not use_spec_decoding
                else {
                    "speculative_algorithm": "NGRAM",
                    "speculative_num_draft_tokens": 5,
                }
            )
            srt_runner = SRTRunner(
                base_path,
                torch_dtype=torch_dtype,
                model_type="generation",
                lora_paths=[lora_adapter_paths[0], lora_adapter_paths[1]],
                enable_lora_overlap_loading=enable_lora_overlap_loading,
                max_loras_per_batch=len(lora_adapter_paths) + 1,
                max_loaded_loras=model_case.max_loaded_loras,
                sleep_on_idle=True,
                attention_backend=attention_backend,
                enable_deterministic_inference=enable_deterministic_inference,
                disable_cuda_graph=disable_cuda_graph,
                disable_radix_cache=disable_radix_cache,
                **spec_args,
            )

            ensure_reproducibility()
            hf_runner = HFRunner(
                base_path,
                torch_dtype=torch_dtype,
                model_type="generation",
                patch_model_do_sample_false=True,
            )

            with srt_runner, hf_runner:
                for i, (prompts, lora_paths) in enumerate(batches):
                    srt_outputs = srt_runner.batch_forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=lora_paths,
                    )
                    hf_outputs = hf_runner.forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=lora_paths,
                    )

                    for srt_out, hf_out in zip(
                        srt_outputs.output_strs, hf_outputs.output_strs
                    ):
                        srt_str = srt_out.strip()
                        hf_str = hf_out.strip()
                        if isinstance(model_case, str):
                            continue
                        rouge_tol = model_case.rouge_l_tolerance
                        rouge_score = calculate_rouge_l([srt_str], [hf_str])[0]
                        if rouge_score < rouge_tol:
                            raise AssertionError(
                                f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                                f"for base '{base_path}', adaptor '{lora_paths}', prompt: '{prompts}...'"
                            )
