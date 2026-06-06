import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=406, suite="nightly-1-npu-a3", nightly=True)


class TestNPUStandaloneSpeculativeDecodingBase(CustomTestCase):
    """Test STANDALONE V1 (non-V2) speculative decoding with ascend backend.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-algorithm=STANDALONE; --speculative-draft-model-path;
    SGLANG_ENABLE_SPEC_V2=0; --speculative-eagle-topk=2;
    --speculative-num-draft-tokens=7
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def get_server_args(cls):
        return [
            "--trust-remote-code",
            "--speculative-algorithm",
            "STANDALONE",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "7",
            "--mem-fraction-static",
            "0.7",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(False)
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.get_server_args(),
            )
        except Exception:
            envs.SGLANG_ENABLE_SPEC_V2.clear()
            raise

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        envs.SGLANG_ENABLE_SPEC_V2.clear()

    def test_gsm8k(self):
        from types import SimpleNamespace

        import requests

        from sglang.test.run_eval import run_eval

        requests.get(self.base_url + "/flush_cache", timeout=30)

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
            gsm8k_data_path=None,
        )
        metrics = run_eval(args)

        self.assertGreaterEqual(metrics["score"], 0.69)

        server_info = requests.get(self.base_url + "/server_info", timeout=10).json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        self.assertGreater(avg_spec_accept_length, 3.6)


if __name__ == "__main__":
    unittest.main()
