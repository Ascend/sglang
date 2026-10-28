import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
    is_npu_a5,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
    disabled="the compile is conflict with graph",
)


class TestEnableTorchCompileDebugMode(CustomTestCase):
    """
    Testcase: When --enable-torch-compile-debug-mode is enabled, inference works
    correctly. With debug mode on, the piecewise backend records input tensor
    addresses at graph capture time and re-checks them on every replay; an
    address mismatch raises an assertion inside the scheduler and crashes the
    server. A passing GSM8K run with normal accuracy proves the debug checks
    hold under real traffic without breaking inference or output quality.

    [Test Category] Parameter
    [Test Target] --enable-torch-compile-debug-mode
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH if is_npu_a5() else QWEN3_8B_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8" if is_npu_a5() else "0.7",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--cuda-graph-backend-prefill=tc_piecewise",
    ]
    enable_args = [
        "--enable-torch-compile-debug-mode",
        "--cuda-graph-backend-prefill=tc_piecewise",
        "--cuda-graph-max-bs-prefill",
        "64",
    ]

    def setUp(self):
        self.base_url = DEFAULT_URL_FOR_TEST
        self.process = None

    def tearDown(self):
        if hasattr(self, "process") and self.process and self.process.pid:
            try:
                kill_process_tree(self.process.pid)
                self.process = None
            except Exception:
                pass

    def test_enable_torch_compile_debug_mode(self):
        # Launch the server with debug mode enabled. During piecewise graph
        # capture the backend records the input tensor addresses, and every
        # graph replay re-checks them. A false positive would raise an
        # assertion inside the scheduler, crashing the server or failing the
        # requests below, so a passing eval means the checks hold.
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args + self.enable_args,
        )

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)

        print("run_gsm8k_accuracy:", metrics["accuracy"])
        # Debug mode only adds host-side address-consistency checks, so it
        # must not break inference nor degrade output quality.
        self.assertGreaterEqual(
            metrics["accuracy"],
            0.6,
            f"Torch compile debug mode broke inference: "
            f"accuracy={metrics['accuracy']}",
        )


if __name__ == "__main__":
    unittest.main()
