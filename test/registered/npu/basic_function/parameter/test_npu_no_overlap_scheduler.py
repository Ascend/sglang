import unittest

import sglang.test.test_utils as test_utils
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, run_mmlu_test

# run_mmlu_test resolves the model from sglang.test.test_utils at call time.
# Point it at the locally cached Ascend weights instead of the gated HF repo id
# (meta-llama/Llama-3.1-8B-Instruct), which returns 403 from hf-mirror.com.
test_utils.DEFAULT_MODEL_NAME_FOR_TEST = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)


class TestOverlapSchedule(CustomTestCase):
    """Testcase: Verify that the model can successfully process inference requests and achieve an accuracy of ≥ 0.65 when the overlap scheduler is disabled,
    covering all combination scenarios of radix cache (enabled/disabled) and chunked prefill (enabled/disabled).

    [Test Category] Parameter
    [Test Target] --disable-radix-cache;--disable-overlap
    """

    def test_no_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True,
            chunked_prefill_size=128,
            disable_overlap=True,
        )

    def test_no_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    def test_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False,
            chunked_prefill_size=128,
            disable_overlap=True,
        )

    def test_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
