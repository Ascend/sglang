import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_WEIGHTS_PATH,
    logger,
    popen_with_error_check,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-16-npu-a3",
    nightly=True,
)

MODEL_PATH = QWEN3_32B_WEIGHTS_PATH

COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-radix-cache",
]

PROMPT_TEXT = "The capital of France is"
EXPECTED_ANSWER = "Paris"


def send_request(url):
    return requests.post(
        f"{url}/generate",
        json={
            "text": PROMPT_TEXT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
            },
        },
    )


class PDDisaggregationHAMixedWorkerFailoverTest(CustomTestCase):
    """Verify PD disaggregation high availability under prefill node failure

    [Test Category] Functional
    [Test Target] PD Disaggregation High Availability (HA)
    --health-failure-threshold; --health-success-threshold; --health-check-timeout-secs; --health-check-interval-secs
    """

    @classmethod
    def setUpClass(cls):
        cls._init_ports()
        cls.log_files = {}
        cls.processes = {}

        cls._start_prefill("prefill_1", base_gpu_id=0)
        cls._start_prefill("prefill_2", base_gpu_id=4)
        cls._start_decode()
        cls._start_lb()

    @classmethod
    def _init_ports(cls):
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname

        base_port = int(parsed.port)
        cls.lb_port = base_port

        cls.ports = {
            "prefill_1": base_port + 100,
            "prefill_2": base_port + 200,
            "decode_1": base_port + 300,
        }

        cls.bootstrap_ports = {
            "prefill_1": base_port + 400,
            "prefill_2": base_port + 500,
        }

        cls.ascend_mf_store_url = f"tcp://{cls.base_host}:{base_port + 600}"

        cls.urls = {
            name: f"http://{cls.base_host}:{port}" for name, port in cls.ports.items()
        }

        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url

    @classmethod
    def _open_logs(cls, name):
        out = open(f"./{name}_out_log.txt", "w+", encoding="utf-8")
        err = open(f"./{name}_err_log.txt", "w+", encoding="utf-8")
        cls.log_files[name] = (out, err)
        return out, err

    @classmethod
    def _mf_env(cls):
        return {
            **os.environ,
            "ASCEND_MF_STORE_URL": cls.ascend_mf_store_url,
        }

    @classmethod
    def _start_prefill(cls, name, base_gpu_id):
        out, err = cls._open_logs(name)

        args = COMMON_ARGS + [
            "--tp-size",
            "4",
            "--base-gpu-id",
            str(base_gpu_id),
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            str(cls.bootstrap_ports[name]),
        ]

        cls.processes[name] = popen_launch_server(
            MODEL_PATH,
            cls.urls[name],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=cls._mf_env(),
            return_stdout_stderr=(out, err),
        )

    @classmethod
    def _start_decode(cls):
        out, err = cls._open_logs("decode_1")

        args = COMMON_ARGS + [
            "--tp-size",
            "4",
            "--base-gpu-id",
            "8",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--load-balance-method",
            "round_robin",
        ]

        cls.processes["decode_1"] = popen_launch_server(
            MODEL_PATH,
            cls.urls["decode_1"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=cls._mf_env(),
            return_stdout_stderr=(out, err),
        )

    @classmethod
    def _start_lb(cls):
        out, err = cls._open_logs("lb")

        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--decode",
            cls.urls["decode_1"],
            "--prefill",
            cls.urls["prefill_1"],
            str(cls.bootstrap_ports["prefill_1"]),
            "--prefill",
            cls.urls["prefill_2"],
            str(cls.bootstrap_ports["prefill_2"]),
            "--host",
            cls.base_host,
            "--port",
            str(cls.lb_port),
            "--policy",
            "round_robin",
            "--health-failure-threshold",
            "2",
            "--health-success-threshold",
            "2",
            "--health-check-timeout-secs",
            "30",
            "--health-check-interval-secs",
            "15",
        ]

        cls.processes["lb"] = popen_with_error_check(
            cmd, return_stdout_stderr=(out, err)
        )

        cls._wait_ready(f"{cls.lb_url}/health")
        logger.info("Waiting 60 seconds for the server to fully initialize...")
        time.sleep(60)

    @classmethod
    def _wait_ready(cls, url, timeout=120):
        start = time.perf_counter()
        while True:
            try:
                if requests.get(url).status_code == 200:
                    logger.info(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")

            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        for name, proc in cls.processes.items():
            if proc:
                try:
                    kill_process_tree(proc.pid)
                    logger.info(f"Killed {name} pid={proc.pid}")
                except Exception as e:
                    logger.warning(f"Failed to kill {name}: {e}")

        time.sleep(5)

        for out, err in cls.log_files.values():
            for f in (out, err):
                try:
                    f.close()
                except Exception:
                    pass

        for name in list(cls.log_files.keys()):
            for suffix in ("out_log.txt", "err_log.txt"):
                path = f"./{name}_{suffix}"
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")

    def count_requests(self, name):
        out, _ = self.log_files[name]
        out.seek(0)
        return out.read().count("POST /generate HTTP/1.1")

    def test_1_normal_round_robin(self):
        """Verify normal load balancing across prefill nodes."""
        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = [ex.submit(send_request, self.base_url) for _ in range(12)]
            for f in futures:
                resp = f.result()
                self.assertEqual(resp.status_code, 200)
                self.assertIn("Paris", resp.text)
                logger.info(resp.json())

        self.assertGreaterEqual(self.count_requests("prefill_1"), 6)
        self.assertGreaterEqual(self.count_requests("prefill_2"), 6)

    def test_2_failover_when_prefill_down(self):
        """Verify prefill node failover under health check failures."""
        if self.processes.get("prefill_2"):
            kill_process_tree(self.processes["prefill_2"].pid)

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(send_request, self.base_url) for _ in range(10)]
            for f in futures:
                resp = f.result()
                self.assertEqual(resp.status_code, 200)
                self.assertIn("Paris", resp.text)
                logger.info(resp.json())

        self.assertGreaterEqual(self.count_requests("prefill_1"), 16)

        lb_log, _ = self.log_files["lb"]
        lb_log.seek(0)
        self.assertIn(
            f"HTTP health check failed for {self.urls['prefill_2']}/health",
            lb_log.read(),
        )


if __name__ == "__main__":
    unittest.main()
