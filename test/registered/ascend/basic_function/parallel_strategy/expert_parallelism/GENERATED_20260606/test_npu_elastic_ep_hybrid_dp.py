import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestElasticEPTP(CustomTestCase):
    """Test elastic EP on NPU with TP-only hybrid configuration.

    [Test Category] Expert Parallelism
    [Test Target] --moe-a2a-backend deepep; --ep-num-redundant-experts; --enable-eplb
    """

    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--attention-backend",
                "ascend",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "32",
                "--chunked-prefill-size",
                "512",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--max-running-requests",
                "32",
                "--mem-fraction-static",
                0.9,
                *cls.extra_args,
            ],
            env={
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "HCCL_BUFFSIZE": "2048",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "TASK_QUEUE_ENABLE": "0",
                "TRANSFORMERS_VERBOSITY": "error",
                "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
                "PYTHONWARNINGS": "ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.60)


@unittest.skipIf(is_in_ci(), "Skip in CI: NPU pure-DP EP test is resource intensive.")
class TestElasticEPPureDP(TestElasticEPTP):
    """Test elastic EP on NPU with pure DP attention hybrid configuration.

    [Test Category] Expert Parallelism
    [Test Target] --enable-dp-attention; --dp 16; --moe-a2a-backend deepep
    """

    extra_args = [
        "--enable-dp-attention",
        "--dp",
        "16",
    ]


@unittest.skipIf(is_in_ci(), "Skip in CI: Hybrid DP+TP EP test is resource intensive.")
class TestElasticEPHybridDPTP(TestElasticEPTP):
    """Test elastic EP on NPU with hybrid DP attention + TP configuration.

    [Test Category] Expert Parallelism
    [Test Target] --enable-dp-attention; --dp 8 (DP+TP hybrid); --moe-a2a-backend deepep
    """

    extra_args = [
        "--enable-dp-attention",
        "--dp",
        "8",
    ]


if __name__ == "__main__":
    unittest.main()
