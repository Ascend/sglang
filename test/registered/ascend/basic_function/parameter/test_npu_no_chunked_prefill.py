import unittest

from sglang.test import test_utils as _test_utils
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, run_bench_serving, run_mmlu_test

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)


class TestNoChunkedPrefill(CustomTestCase):
    """Testcase: Verify Llama-3.1-8B-Instruct accuracy ≥ 0.65 and serving normal with chunked prefill disabled.

    [Test Category] Parameter
    [Test Target] --chunked-prefill-size
    """

    def test_no_chunked_prefill(self):
        original_model = _test_utils.DEFAULT_MODEL_NAME_FOR_TEST
        _test_utils.DEFAULT_MODEL_NAME_FOR_TEST = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        try:
            run_mmlu_test(
                disable_radix_cache=False,
                enable_mixed_chunk=False,
                chunked_prefill_size=-1,
            )
        finally:
            _test_utils.DEFAULT_MODEL_NAME_FOR_TEST = original_model

    def test_no_chunked_prefill_without_radix_cache(self):
        res = run_bench_serving(
            model=LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache", "--chunked-prefill-size", "-1"],
        )

        assert res["completed"] == 10


if __name__ == "__main__":
    unittest.main()
