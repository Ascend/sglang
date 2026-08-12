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
    suite="full-8-npu-a3",
    nightly=True,
)

MODEL_PATH = QWEN3_32B_WEIGHTS_PATH
TP_SIZE = 4
MEM_FRACTION = "0.8"
COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    MEM_FRACTION,
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-radix-cache",
]

PROMPT_TEXT = "The capital of France is"
EXPECTED_ANSWER = "Paris"


def send_request(base_url):
    return requests.post(
        f"{base_url}/generate",
        json={
            "text": PROMPT_TEXT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
            },
        },
        timeout=60,
    )


class RouterHAMixedWorkerFailoverTest(CustomTestCase):
    """Verify router high availability and failover under mixed-worker deployment.

    [Test Category] Functional
    [Test Target] Router High Availability (HA)
    --health-failure-threshold; --health-success-threshold; --health-check-timeout-secs; --health-check-interval-secs
    """

    @classmethod
    def setUpClass(cls):
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname

        base_port = int(parsed.port)
        cls.lb_port = base_port
        cls.server1_port = base_port + 100
        cls.server2_port = base_port + 200

        cls.server1_url = f"http://{cls.base_host}:{cls.server1_port}"
        cls.server2_url = f"http://{cls.base_host}:{cls.server2_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"

        cls.log_files = {}
        cls.processes = {}

        cls._start_server(1, base_gpu_id=0)
        cls._start_server(2, base_gpu_id=4)
        cls._launch_lb()

    @classmethod
    def _open_logs(cls, prefix):
        out = open(f"./{prefix}_out_log.txt", "w+", encoding="utf-8")
        err = open(f"./{prefix}_err_log.txt", "w+", encoding="utf-8")
        cls.log_files[prefix] = (out, err)
        return out, err

    @classmethod
    def _start_server(cls, idx, base_gpu_id):
        port = getattr(cls, f"server{idx}_port")
        url = getattr(cls, f"server{idx}_url")

        out, err = cls._open_logs(f"server{idx}")
        args = COMMON_ARGS + [
            "--tp-size",
            str(TP_SIZE),
            "--base-gpu-id",
            str(base_gpu_id),
        ]

        cls.processes[f"server{idx}"] = popen_launch_server(
            MODEL_PATH,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            return_stdout_stderr=(out, err),
        )

    @classmethod
    def _launch_lb(cls):
        out, err = cls._open_logs("lb")

        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--worker-urls",
            cls.server1_url,
            cls.server2_url,
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
        cls._wait_ready(cls.lb_url + "/health")
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

    def assert_generate_success(self, future):
        resp = future.result()
        self.assertEqual(resp.status_code, 200)
        self.assertIn(EXPECTED_ANSWER, resp.text)

    def count_requests(self, log_name):
        out, _ = self.log_files[log_name]
        out.seek(0)
        return out.read().count("POST /generate HTTP/1.1")

    def test_1_normal_round_robin(self):
        """Verify normal load balancing across mixed workers."""
        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = [ex.submit(send_request, self.lb_url) for _ in range(12)]
            for f in futures:
                self.assert_generate_success(f)

        self.assertGreaterEqual(self.count_requests("server1"), 6)
        self.assertGreaterEqual(self.count_requests("server2"), 6)

    def test_2_failover_when_worker_down(self):
        """Verify worker failover when a backend node becomes unhealthy."""
        proc = self.processes.get("server2")
        if proc:
            kill_process_tree(proc.pid)

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(send_request, self.lb_url) for _ in range(10)]
            for f in futures:
                self.assert_generate_success(f)

        self.assertGreaterEqual(self.count_requests("server1"), 16)

        lb_log, _ = self.log_files["lb"]
        lb_log.seek(0)
        content = lb_log.read()
        self.assertIn(
            f"HTTP health check failed for {self.server2_url}/health", content
        )
        self.assertIn("Failed to send typed request worker_url=", content)


if __name__ == "__main__":
    unittest.main()
