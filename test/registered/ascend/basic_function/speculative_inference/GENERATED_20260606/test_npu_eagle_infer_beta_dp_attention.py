import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


NPU_ENV = {
    **os.environ,
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "HCCL_BUFFSIZE": "200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "HCCL_EXEC_TIMEOUT": "200",
    "STREAMS_PER_DEVICE": "32",
}


def _run_gsm8k(base_url: str, model: str):
    requests.get(base_url + "/flush_cache", timeout=30)

    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        api="completion",
        max_tokens=512,
        num_examples=200,
        num_threads=128,
    )
    metrics = run_eval(args)
    server_info = requests.get(base_url + "/server_info", timeout=10).json()
    avg_spec_accept_length = server_info["internal_states"][0]["avg_spec_accept_length"]
    return metrics, avg_spec_accept_length


class TestNpuEagleDPAttnServerSmall(CustomTestCase):
    """Test EAGLE3 with DP attention on NPU (TP=2, DP=2, 4 NPU).

    [Test Category] Speculative Decoding
    [Test Target] --speculative-algorithm=EAGLE3; --tp-size 2; --dp-size 2;
    --enable-dp-attention; --speculative-draft-model-path
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.7",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=NPU_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        metrics, avg_spec_accept_length = _run_gsm8k(self.base_url, self.model)
        # NPU speculative-eagle-topk is fixed to 1 (NPU ascend has fixed page_size),
        # so accuracy and avg_spec_accept_length are lower than GPU.
        # Observed: score=0.95, avg_spec_accept_length=1.98 on 8B.
        self.assertGreater(metrics["score"], 0.55)
        self.assertGreater(avg_spec_accept_length, 1.9)


if __name__ == "__main__":
    unittest.main()
