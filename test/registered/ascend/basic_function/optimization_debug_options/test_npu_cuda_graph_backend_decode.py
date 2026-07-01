import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import _run_accuracy_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

_MMLU_THRESHOLD = 0.2


class TestDecodeBackendBreakable(CustomTestCase):
    """Testcase: verify --cuda-graph-backend-decode=breakable

    Verify graph capture succeeds and MMLU accuracy does not regress.

    [Test Category] Parameter
    [Test Target] --cuda-graph-backend-decode
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--cuda-graph-backend-decode",
                "breakable",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_decode_backend_breakable(self):
        # verify breakable backend was selected
        self.err_log_file.seek(0)
        err_log = self.err_log_file.read()
        self.assertIn(
            "cuda_graph_backend_decode='breakable'",
            err_log,
            "Expected stderr to contain \"cuda_graph_backend_decode='breakable'\", "
            "proving breakable backend was selected for decode graph",
        )

        # verify MMLU accuracy does not regress
        _run_accuracy_eval(
            self,
            eval_name="mmlu",
            score_threshold=_MMLU_THRESHOLD,
            num_examples=64,
            num_threads=32,
        )


class TestDecodeBackendTcPiecewise(CustomTestCase):
    """Testcase: verify --cuda-graph-backend-decode=tc_piecewise

    Verify graph capture succeeds and MMLU accuracy does not regress.

    [Test Category] Parameter
    [Test Target] --cuda-graph-backend-decode
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--cuda-graph-backend-decode",
                "tc_piecewise",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")

    def test_decode_backend_tc_piecewise(self):
        # verify tc_piecewise backend was selected
        self.err_log_file.seek(0)
        err_log = self.err_log_file.read()
        self.assertIn(
            "cuda_graph_backend_decode='tc_piecewise'",
            err_log,
            "Expected stderr to contain \"cuda_graph_backend_decode='tc_piecewise'\", "
            "proving tc_piecewise backend was selected for decode graph",
        )

        # verify MMLU accuracy does not regress
        _run_accuracy_eval(
            self,
            eval_name="mmlu",
            score_threshold=_MMLU_THRESHOLD,
            num_examples=64,
            num_threads=32,
        )


if __name__ == "__main__":
    unittest.main()
