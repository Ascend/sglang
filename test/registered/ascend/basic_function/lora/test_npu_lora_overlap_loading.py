import multiprocessing as mp
import unittest

import torch

from sglang.test.ascend.lora_utils import (
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)


class TestLoRAOverlapLoading(CustomTestCase):

    def test_ci_lora_models_batch_splitting(self):
        run_lora_batch_splitting_equivalence_test(
            CI_MULTI_LORA_MODELS,
            enable_lora_overlap_loading=True,
            torch_dtype=torch.bfloat16,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
