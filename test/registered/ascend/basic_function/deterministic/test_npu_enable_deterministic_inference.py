import os
import shlex
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_4B_WEIGHTS_PATH,
    logger,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNpuDeterministicNoTp(CustomTestCase):
    """Testcase: Using sglang.test.test_deterministic, verified that with parallelism strategies disabled, batch invariance
    is stably maintained across no-shared-prefix, shared-prefix, and radix cache scenarios, thereby achieving the goal of inference consistency

    [Test Category] Functional
    [Test Target] --enable-deterministic-inference, --attention-backend, --sampling-backend
    """

    MODEL = QWEN3_4B_WEIGHTS_PATH
    BASE_URL = DEFAULT_URL_FOR_TEST

    OTHER_ARGS = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--sampling-backend",
        "ascend",
        "--device",
        "npu",
        "--chunked-prefill-size",
        "2048",
        "--mem-fraction-static",
        "0.75",
        "--disable-cuda-graph",
        "--enable-deterministic-inference",
    ]

    @classmethod
    def setUpClass(cls):
        os.environ["ASCEND_USE_FIA"] = "1"
        parsed = urlparse(cls.BASE_URL)
        cls.host, cls.port = parsed.hostname, parsed.port
        cls.process = popen_launch_server(
            model=cls.MODEL,
            base_url=cls.BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.OTHER_ARGS,
            env=os.environ.copy(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def flush_cache(self):
        requests.post(f"{self.BASE_URL}/flush_cache")

    def run_deterministic(self, extra_args, checks):
        command = [
            "python3",
            "-m",
            "sglang.test.test_deterministic",
            "--host",
            self.host,
            "--port",
            str(self.port),
            *extra_args,
        ]
        logger.info(f"command={shlex.join(command)}")

        output = run_command(
            command,
            shell=False,
        )

        logger.info("===== run_command start =====")
        logger.info(output)
        logger.info("===== run_command end =======")

        for check in checks:
            self.assertIn(check, output)

        return output

    def test_single(self):
        for t in ("0.0", "0.7"):
            self.flush_cache()
            self.run_deterministic(
                [
                    "--n-trials",
                    "50",
                    "--test-mode",
                    "single",
                    "--temperature",
                    t,
                ],
                [
                    "Total samples: 50, Unique samples: 1",
                ],
            )

    def test_prefix(self):
        for t in ("0.0", "0.7"):
            self.flush_cache()

            output = self.run_deterministic(
                [
                    "--n-start",
                    "1",
                    "--n-trials",
                    "50",
                    "--test-mode",
                    "prefix",
                    "--return-logprob",
                    "--temperature",
                    t,
                ],
                [
                    "Unique samples: 1",
                    "Logprobs are identical across all batch sizes!",
                ],
            )

            self.assertEqual(output.count("Unique samples: 1"), 4)

    def test_radix_cache(self):
        for t in ("0.0", "0.7"):
            self.flush_cache()
            self.run_deterministic(
                [
                    "--test-mode",
                    "radix_cache",
                    "--temperature",
                    t,
                ],
                [
                    "TEST PASSED - Radix cache is consistent!",
                ],
            )


class TestNpuDeterministicTp(TestNpuDeterministicNoTp):
    """Testcase：Using sglang.test.test_deterministic, verified that with parallelism strategies enabled, batch invariance
    is stably maintained across no-shared-prefix, shared-prefix, and radix cache scenarios, thereby achieving the goal of inference consistency

    [Test Category] Functional
    [Test Target] --enable-deterministic-inference, --attention-backend, --sampling-backend
    """

    OTHER_ARGS = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--sampling-backend",
        "ascend",
        "--device",
        "npu",
        "--chunked-prefill-size",
        "2048",
        "--mem-fraction-static",
        "0.7",
        "--disable-cuda-graph",
        "--enable-deterministic-inference",
        "--tp-size",
        4,
    ]

    @classmethod
    def setUpClass(cls):
        os.environ["ASCEND_USE_FIA"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "strict"
        parsed = urlparse(cls.BASE_URL)
        cls.host, cls.port = parsed.hostname, parsed.port
        cls.process = popen_launch_server(
            model=cls.MODEL,
            base_url=cls.BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.OTHER_ARGS,
            env=os.environ.copy(),
        )


if __name__ == "__main__":
    unittest.main()