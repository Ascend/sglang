import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import logger, popen_with_error_check, QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,

)

register_npu_ci(
    est_time=400,
    suite="nightly-8-npu-a3",
    nightly=True,
)


def send_request(base_url):
    return requests.post(
        f"{base_url}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
            },
        },
    )


def _get_log_content(log_file):
    log_file.seek(0)
    log_file.flush()
    os.fsync(log_file.fileno())
    return log_file.read()


class DisaggregationHiCacheBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_1_port = f"{int(base_port) + 100}"
        cls.prefill_2_port = f"{int(base_port) + 200}"
        cls.decode_1_port = f"{int(base_port) + 300}"
        cls.prefill_1_bootstrap_port = f"{int(base_port) + 400}"
        cls.prefill_2_bootstrap_port = f"{int(base_port) + 500}"
        cls.ascend_mf_store_url = f"tcp://{cls.base_host}:{int(base_port) + 600}"
        cls.prefill_1_url = f"http://{cls.base_host}:{cls.prefill_1_port}"
        cls.prefill_2_url = f"http://{cls.base_host}:{cls.prefill_2_port}"
        cls.decode_1_url = f"http://{cls.base_host}:{cls.decode_1_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.process_lb, cls.process_prefill_1, cls.process_prefill_2, cls.process_decode_1 = None, None, None, None
        cls.base_url = cls.lb_url
        # cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.model = "/home/weights/Qwen3-32B"
        cls.start_prefill_1()
        cls.start_prefill_2()
        cls.start_decode_1()
        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        cls.lb_out_log.close()
        cls.lb_err_log.close()
        cls.prefill_1_out_log.close()
        cls.prefill_1_err_log.close()
        cls.prefill_2_out_log.close()
        cls.prefill_2_err_log.close()
        cls.decode_1_out_log.close()
        cls.decode_1_err_log.close()
        # os.remove("./prefill_1_out_log.txt")
        # os.remove("./prefill_1_err_log.txt")
        # os.remove("./prefill_2_out_log.txt")
        # os.remove("./prefill_2_err_log.txt")
        # os.remove("./lb_out_log.txt")
        # os.remove("./lb_err_log.txt")

        for process in [cls.process_lb, cls.process_prefill_1, cls.process_prefill_2, cls.process_decode_1]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    logger.info(f"Error killing process {process.pid}: {e}")

        # wait for 5 seconds
        time.sleep(5)

    @classmethod
    def wait_server_ready(cls, url, timeout=120):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    logger.info(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    @classmethod
    def start_prefill_1(cls):
        cls.prefill_1_out_log = open("./prefill_1_out_log.txt", "w+", encoding="utf-8")
        cls.prefill_1_err_log = open("./prefill_1_err_log.txt", "w+", encoding="utf-8")
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--tp-size",
            "4",
            "--base-gpu-id",
            0,
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.prefill_1_bootstrap_port,

        ]
        env = {
            **os.environ,
            "ASCEND_MF_STORE_URL": cls.ascend_mf_store_url,
        }
        cls.process_prefill_1 = popen_launch_server(
            cls.model,
            cls.prefill_1_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(cls.prefill_1_out_log, cls.prefill_1_err_log)
        )

    @classmethod
    def start_prefill_2(cls):
        cls.prefill_2_out_log = open("./prefill_2_out_log.txt", "w+", encoding="utf-8")
        cls.prefill_2_err_log = open("./prefill_2_err_log.txt", "w+", encoding="utf-8")
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--tp-size",
            "4",
            "--base-gpu-id",
            4,
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.prefill_2_bootstrap_port,
        ]
        env = {
            **os.environ,
            "ASCEND_MF_STORE_URL": cls.ascend_mf_store_url,
        }
        cls.process_prefill_2 = popen_launch_server(
            cls.model,
            cls.prefill_2_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(cls.prefill_2_out_log, cls.prefill_2_err_log),
        )

    @classmethod
    def start_decode_1(cls):
        cls.decode_1_out_log = open("./decode_1_out_log.txt", "w+", encoding="utf-8")
        cls.decode_1_err_log = open("./decode_1_err_log.txt", "w+", encoding="utf-8")
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--tp-size",
            "4",
            "--base-gpu-id",
            8,
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--load-balance-method",
            "round_robin",
        ]
        env = {
            **os.environ,
            "ASCEND_MF_STORE_URL": cls.ascend_mf_store_url,
        }
        cls.process_decode_1 = popen_launch_server(
            cls.model,
            cls.decode_1_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(cls.decode_1_out_log, cls.decode_1_err_log),
        )

    @classmethod
    def launch_lb(cls):
        cls.lb_out_log = open("./lb_out_log.txt", "w+", encoding="utf-8")
        cls.lb_err_log = open("./lb_err_log.txt", "w+", encoding="utf-8")
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--decode",
            cls.decode_1_url,
            "--prefill",
            cls.prefill_1_url,
            cls.prefill_1_bootstrap_port,
            "--prefill",
            cls.prefill_2_url,
            cls.prefill_2_bootstrap_port,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
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
        cls.process_lb = popen_with_error_check(lb_command, return_stdout_stderr=(cls.lb_out_log, cls.lb_err_log))
        cls.wait_server_ready(cls.lb_url + "/health")
        logger.info(
            f"Waiting 60 seconds for the server to fully initialize..."
        )
        time.sleep(60)

    def test_1(self):
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(send_request, self.base_url)
                for _ in range(12)
            ]

            for future in futures:
                response = future.result()
                self.assertEqual(response.status_code, 200)
                self.assertIn("Paris", response.text)
                logger.info(response.json())
        self.prefill_1_out_log.seek(0)
        self.assertGreaterEqual(self.prefill_1_out_log.read().count("POST /generate HTTP/1.1"), 6)
        self.prefill_2_out_log.seek(0)
        self.assertGreaterEqual(self.prefill_2_out_log.read().count("POST /generate HTTP/1.1"), 6)

    def test_2(self):
        if self.process_prefill_2:
            try:
                kill_process_tree(self.process_prefill_2.pid)
            except Exception as e:
                logger.error(f"Error killing process {self.process_prefill_2.pid}: {e}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(send_request, self.base_url)
                for _ in range(10)
            ]

            for future in futures:
                response = future.result()
                self.assertEqual(response.status_code, 200)
                self.assertIn("Paris", response.text)
                logger.info(response.json())
        self.prefill_1_out_log.seek(0)
        self.assertGreaterEqual(self.prefill_1_out_log.read().count("POST /generate HTTP/1.1"), 16)
        self.lb_out_log.seek(0)
        self.assertIn(f"HTTP health check failed for {self.prefill_2_url}/health", self.lb_out_log.read())


if __name__ == "__main__":
    unittest.main()
