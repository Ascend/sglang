import io
import os
import tempfile
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

SERVER_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM": "1",
    "ASCEND_USE_FIA": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
}

SERVER_ARGS = [
    "--tp-size",
    "2",
    "--nnodes",
    "1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    "-1",
    "--max-total-tokens",
    "600000",
    "--max-prefill-tokens",
    "65536",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests",
    "10",
    "--max-mamba-cache-size",
    "20",
    "--mem-fraction-static",
    "0.65",
    "--cuda-graph-bs-decode",
    "2",
    "4",
    "8",
    "12",
    "14",
    "16",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]

LOG_READ_TIMEOUT = 60
# Server output is dumped here so a failure stays triagable after the screen scrolls.
LOG_DIR = os.path.join(tempfile.gettempdir(), "qwen36_block_fp8_moe_startup_logs")


def wait_for_log_text(buffers, needle, timeout=LOG_READ_TIMEOUT):
    """Poll the captured output until the needle shows up or the deadline passes."""
    deadline = time.time() + timeout
    while True:
        if any(needle in buf.getvalue() for buf in buffers) or time.time() >= deadline:
            return "\n".join(buf.getvalue() for buf in buffers)
        time.sleep(1)


class _StartupErrorBase(CustomTestCase):
    """Shared check: start the server with a bad flag and expect a clear failure."""

    model = QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    other_args = []
    expected_error = ""

    def dump_log(self, text, name):
        """Persist the captured server output so a failure can be triaged later."""
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"{name}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def captured_output(self, stdout, stderr):
        return "\n".join(buf.getvalue() for buf in (stdout, stderr))

    def assert_startup_fails(self):
        name = type(self).__name__
        # The construct-time guards are mode independent, so pin the native mode.
        env = {**SERVER_ENVS, "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": "0"}
        # popen_launch_server raises a generic "Server process exited with code N"
        # without the real error text, so tee the server output into buffers and
        # assert the expected message on them.
        stdout = io.StringIO()
        stderr = io.StringIO()
        timed_out = False
        try:
            process = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=SERVER_ARGS + self.other_args,
                env=env,
                return_stdout_stderr=(stdout, stderr),
            )
        except Exception as exc:
            # TimeoutError means the process never exited -> silent hang suspected.
            timed_out = isinstance(exc, TimeoutError)
            print(f"[{name}] startup failed as expected: {exc}")
        else:
            kill_process_tree(process.pid)
            log_path = self.dump_log(self.captured_output(stdout, stderr), name)
            self.fail(
                f"{name}: server started successfully with invalid args; {log_path}"
            )

        output = wait_for_log_text((stdout, stderr), self.expected_error)
        log_path = self.dump_log(output, name)
        print(f"[{name}] log={log_path}")

        if timed_out:
            self.fail(
                f"{name}: the server never exited within "
                f"{DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH}s, silent hang suspected; "
                f"{log_path}"
            )
        self.assertIn(
            self.expected_error,
            output,
            f"{name}: the server exited but its log carries no expected error text; "
            f"{log_path}",
        )


# The A/B prefixes keep unittest's alphabetical collection in the S1/S2 order of
# the case table (runner-backend check runs first, dispatcher-dtype check second).
class TestE2E6ARunnerBackendRejected(_StartupErrorBase):
    """Testcase: Verify an invalid --moe-runner-backend (a legal CLI value, but only auto/ascend is accepted by this feature) makes startup fail with an explicit error instead of silently serving.

    [Test Category] Exception
    [Test Target] --moe-runner-backend triton
    [Test Config] TP2, single node, native Block FP8 mode
    """

    other_args = ["--moe-runner-backend", "triton"]
    expected_error = "requires --moe-runner-backend auto"

    def test_startup_fails_with_runner_backend_error(self):
        self.assert_startup_fails()


class TestE2E6BDispatcherDtypeRejected(_StartupErrorBase):
    """Testcase: Verify an invalid --deepep-dispatcher-output-dtype (a legal CLI value, but only auto/bf16 is accepted) makes startup fail with an explicit error instead of silently serving.

    [Test Category] Exception
    [Test Target] --deepep-dispatcher-output-dtype fp8
    [Test Config] TP2, single node, native Block FP8 mode
    """

    other_args = ["--deepep-dispatcher-output-dtype", "fp8"]
    expected_error = "must be auto or bf16"

    def test_startup_fails_with_dispatcher_dtype_error(self):
        self.assert_startup_fails()


if __name__ == "__main__":
    unittest.main()
