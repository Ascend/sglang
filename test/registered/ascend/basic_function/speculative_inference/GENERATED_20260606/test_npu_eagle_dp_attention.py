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
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
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


class TestNpuEAGLE3EngineDPAttention(CustomTestCase):
    """Test EAGLE3 speculative decoding with DP attention on NPU (TP=2, DP=2).

    [Test Category] Speculative Decoding
    [Test Target] --speculative-algorithm=EAGLE3; --speculative-draft-model-path;
    --speculative-num-steps; --speculative-eagle-topk; --speculative-num-draft-tokens;
    --tp-size 2; --dp-size 2; --enable-dp-attention; --enable-dp-lm-head
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.draft_model = QWEN3_8B_EAGLE3_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-num-steps",
            "6",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "32",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--tp-size",
            "2",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            "1",
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
        requests.get(self.base_url + "/flush_cache", timeout=30)

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

        server_info = requests.get(self.base_url + "/server_info", timeout=10).json()
        avg_spec_accept_length = None
        if "internal_states" in server_info and len(server_info["internal_states"]) > 0:
            internal_state = server_info["internal_states"][0]
            if "avg_spec_accept_length" in internal_state:
                avg_spec_accept_length = internal_state["avg_spec_accept_length"]
            elif "spec_accept_length" in internal_state:
                avg_spec_accept_length = internal_state["spec_accept_length"]

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (EAGLE3 DP Attention on NPU)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["score"], 0.69)
            if avg_spec_accept_length is not None:
                # NPU speculative-eagle-topk is fixed to 1 (NPU ascend has fixed page_size),
                # so avg_spec_accept_length is lower than GPU; only require > 1.0.
                self.assertGreater(avg_spec_accept_length, 1.0)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (EAGLE3 DP Attention on NPU)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            # NPU speculative-eagle-topk=1 → acc_length is lower than GPU.
            self.assertGreater(acc_length, 1.0)
            self.assertGreater(speed, 10)


if __name__ == "__main__":
    unittest.main()
