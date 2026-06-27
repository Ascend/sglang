"""End-to-end test for --detokenizer-worker-num on NPU.

Transplanted from: test/registered/tokenizer/test_multi_detokenizer.py (GPU)
- T1 test_multi_detokenizer_ttft: benchmark with 4 detokenizer + 4 tokenizer workers
  Adapted for NPU: Llama-3.2-1B model, ascend backend, register_npu_ci
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    auto_config_device,
    get_benchmark_args,
    popen_launch_server,
    run_benchmark,
)

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)


class TestNpuMultiDetokenizer(CustomTestCase):
    """Transplanted from GPU test_multi_detokenizer.py — verify multi-detokenizer worker
    in benchmark scenario.

    GPU origin: TestMultiDetokenizer.test_multi_detokenizer_ttft
    Differences:
    - Model: DEFAULT_MODEL_NAME_FOR_TEST → LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    - Registration: register_cuda_ci + register_amd_ci → register_npu_ci
    - Backend: added --attention-backend ascend, --mem-fraction-static 0.7
    - Worker count: detokenizer=4 (same), tokenizer=4 (reduced from 8 for 1B model)
    - Removed: MMLUMixin, write_github_step_summary (not applicable to NPU CI)

    [Test Category] Parameter / Boundary
    [Test Target] --detokenizer-worker-num=4
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                "4",
                "--detokenizer-worker-num",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multi_detokenizer_ttft_npu(self):
        """Run benchmark with multi-detokenizer workers, verify completion and latency.

        GPU origin: test_multi_detokenizer_ttft in test/registered/tokenizer/test_multi_detokenizer.py
        """
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            dataset_path="",
            tokenizer=None,
            num_prompts=100,
            random_input_len=4096,
            random_output_len=2048,
            sharegpt_context_len=None,
            request_rate=1,
            disable_stream=False,
            disable_ignore_eos=False,
            seed=0,
            device=auto_config_device(),
            lora_name=None,
        )
        res = run_benchmark(args)
        self.assertGreater(res["completed"], 0)
        self.assertLess(res["median_e2e_latency_ms"], 60000)
        self.assertLess(res["median_ttft_ms"], 2000)
        self.assertLess(res["median_itl_ms"], 50)


if __name__ == "__main__":
    unittest.main()
