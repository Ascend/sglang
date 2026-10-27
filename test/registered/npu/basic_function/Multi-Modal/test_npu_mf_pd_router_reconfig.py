"""MemFabric PD disaggregation router reconfiguration tests.

Automation of MemFabric_PDSeparate_Func_01/02/03. A pool of 2 prefill +
2 decode workers (Qwen3-4B, TP=1, 4 NPUs) is started once; each test
restarts the router with a different topology:
  - test_01_scale_up_1p1d_to_2p2d: 1P1D -> kill router -> 2P2D router
  - test_02_kill_single_prefill:   2P1D -> kill router + one prefill -> 1P1D router
  - test_03_kill_single_decode:    1P2D -> kill router + one decode  -> 1P1D router

Tests are order-dependent (test_02 kills prefill_2, test_03 kills decode_2)
and run in filename order.
"""

import os
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_4B_WEIGHTS_PATH,
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
    est_time=1500,
    suite="full-4-npu-a3",
    nightly=True,
)

MODEL_PATH = QWEN3_4B_WEIGHTS_PATH
MEM_FRACTION = "0.8"

COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    MEM_FRACTION,
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
]

# 2 prefills + 2 decodes, one NPU each (TP=1)
WORKER_LAYOUT = {
    "prefill_1": {"mode": "prefill", "base_gpu_id": 0},
    "prefill_2": {"mode": "prefill", "base_gpu_id": 1},
    "decode_1": {"mode": "decode", "base_gpu_id": 2},
    "decode_2": {"mode": "decode", "base_gpu_id": 3},
}

PROMPT_TEXT = "The capital of France is"
EXPECTED_ANSWER = "Paris"


def send_request(url):
    """Send one /generate request."""
    return requests.post(
        f"{url}/generate",
        json={
            "text": PROMPT_TEXT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
            },
        },
        timeout=60,
    )


class MFPDRouterReconfigTest(CustomTestCase):
    """Verify PD disaggregation service availability across router reconfig
    and worker failure scenarios.

    [Test Category] Functional
    [Test Target] MemFabric PD Disaggregation Router Reconfig (HA)
    ASCEND_MF_STORE_URL; ASCEND_MF_TRANSFER_PROTOCOL; --pd-disaggregation;
    --health-failure-threshold; --health-success-threshold
    """

    def _callTestMethod(self, method):
        # Stateful tests: no retry on failure.
        method()

    @classmethod
    def setUpClass(cls):
        cls._init_ports()
        cls.log_files = {}
        cls.processes = {}

        # Start all workers up front; popen_launch_server blocks until ready.
        for name in WORKER_LAYOUT:
            cls._start_worker(name)

    @classmethod
    def _init_ports(cls):
        """Port layout (base port bp from DEFAULT_URL_FOR_TEST):
        lb=bp, prefill_1=bp+100, prefill_2=bp+200, decode_1=bp+300, decode_2=bp+350,
        bootstrap: prefill_1=bp+400, prefill_2=bp+500, MF store=bp+600
        """
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        bp = int(parsed.port)

        cls.lb_port = bp
        cls.ports = {
            "prefill_1": bp + 100,
            "prefill_2": bp + 200,
            "decode_1": bp + 300,
            "decode_2": bp + 350,
        }
        cls.bootstrap_ports = {
            "prefill_1": bp + 400,
            "prefill_2": bp + 500,
        }
        cls.ascend_mf_store_url = f"tcp://{cls.base_host}:{bp + 600}"
        cls.urls = {
            name: f"http://{cls.base_host}:{port}" for name, port in cls.ports.items()
        }
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"

    @classmethod
    def _open_logs(cls, name):
        out = open(f"./{name}_out_log.txt", "w+", encoding="utf-8")
        err = open(f"./{name}_err_log.txt", "w+", encoding="utf-8")
        cls.log_files[name] = (out, err)
        return out, err

    @classmethod
    def _mf_env(cls):
        """Env for MemFabric PD disaggregation (sdma on single-node A3)."""
        return {
            **os.environ,
            "ASCEND_MF_STORE_URL": cls.ascend_mf_store_url,
            "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
            "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "TASK_QUEUE_ENABLE": "2",
            "STREAMS_PER_DEVICE": "32",
            "DP_ROUND_ROBIN": "1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "48",
        }

    @classmethod
    def _start_worker(cls, name):
        """Start one prefill/decode worker per WORKER_LAYOUT."""
        spec = WORKER_LAYOUT[name]
        out, err = cls._open_logs(name)

        args = COMMON_ARGS + [
            "--tp-size",
            "1",
            "--base-gpu-id",
            str(spec["base_gpu_id"]),
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            spec["mode"],
        ]

        if spec["mode"] == "prefill":
            args += [
                "--disaggregation-bootstrap-port",
                str(cls.bootstrap_ports[name]),
                "--enable-hierarchical-cache",
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "372",
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
    def _kill_router(cls):
        """Kill the current router (idempotent); workers are left running."""
        proc = cls.processes.get("lb")
        if proc:
            try:
                kill_process_tree(proc.pid)
                logger.info(f"Killed old router pid={proc.pid}")
            except Exception as e:
                logger.warning(f"Failed to kill old router: {e}")
            cls.processes["lb"] = None
            time.sleep(5)

        logs = cls.log_files.pop("lb", None)
        if logs:
            for f in logs:
                try:
                    f.close()
                except Exception:
                    pass

    @classmethod
    def _launch_router(cls, prefill_names, decode_names):
        """Restart the router with the given topology and wait until ready."""
        cls._kill_router()
        out, err = cls._open_logs("lb")

        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--policy",
            "random",
        ]
        for name in prefill_names:
            cmd += ["--prefill", cls.urls[name], str(cls.bootstrap_ports[name])]
        for name in decode_names:
            cmd += ["--decode", cls.urls[name]]

        cmd += [
            "--host",
            cls.base_host,
            "--port",
            str(cls.lb_port),
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
        cls._wait_pd_ready(prefill_names, decode_names)

    @classmethod
    def _wait_ready(cls, url, timeout=120):
        start = time.perf_counter()
        while True:
            try:
                if requests.get(url, timeout=10).status_code == 200:
                    logger.info(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")

            time.sleep(1)

    @classmethod
    def _wait_pd_ready(cls, prefill_names, decode_names, timeout=60):
        """Wait for the PD channel: probe the LB until a request returns 200."""
        start = time.perf_counter()
        while True:
            try:
                resp = send_request(cls.lb_url)
                if resp.status_code == 200:
                    logger.info(
                        f"PD channel ready in {time.perf_counter() - start:.0f}s "
                        f"(prefills={prefill_names}, decodes={decode_names})"
                    )
                    return
                logger.info(
                    f"PD channel not ready yet (status={resp.status_code}), retrying..."
                )
            except Exception as e:
                logger.info(f"PD probe failed ({e}), retrying...")

            if time.perf_counter() - start > timeout:
                raise RuntimeError(
                    f"PD channel failed to become ready in {timeout}s "
                    f"(prefills={prefill_names}, decodes={decode_names})"
                )

            time.sleep(5)

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

    # ---------------- Helpers ----------------

    def _send_and_assert(self):
        """Send one request and assert 200 with the expected answer."""
        resp = send_request(self.lb_url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(EXPECTED_ANSWER, resp.text)
        logger.info(resp.json())

    def _assert_workers_healthy(self, names):
        """Check worker /health endpoints directly (bypassing the router)."""
        for name in names:
            resp = requests.get(f"{self.urls[name]}/health", timeout=10)
            self.assertEqual(resp.status_code, 200, f"Worker {name} is not healthy")
        logger.info(f"Workers {names} are healthy")

    def _kill_worker(self, name):
        """Kill a worker and confirm it terminated (idempotent)."""
        proc = self.processes.get(name)
        if proc is None:
            logger.warning(f"Worker {name} already killed, skip")
            return
        kill_process_tree(proc.pid)
        time.sleep(5)
        self.assertIsNotNone(proc.poll(), f"Worker {name} not terminated yet")
        self.processes[name] = None
        logger.info(f"Worker {name} killed")

    # ---------------- Tests ----------------

    def test_01_scale_up_1p1d_to_2p2d(self):
        """Func_01: 1P1D service works; after router restart with an added
        prefill and decode (2P2D), existing workers stay alive and requests
        still succeed.
        """
        self._launch_router(["prefill_1"], ["decode_1"])

        self._assert_workers_healthy(["prefill_1", "decode_1"])
        self._send_and_assert()

        # Restart router as 2P2D
        self._launch_router(["prefill_1", "prefill_2"], ["decode_1", "decode_2"])

        # Existing workers unaffected
        self._assert_workers_healthy(["prefill_1", "decode_1"])
        self._send_and_assert()

    def test_02_kill_single_prefill(self):
        """Func_02: 2P1D service works; after killing the router and one
        prefill, remaining workers stay alive and a 1P1D router restores
        request serving.
        """
        self._launch_router(["prefill_1", "prefill_2"], ["decode_1"])
        self._send_and_assert()

        self._kill_router()
        self._kill_worker("prefill_2")

        # Remaining workers unaffected
        self._assert_workers_healthy(["prefill_1", "decode_1", "decode_2"])

        # Restart router as 1P1D
        self._launch_router(["prefill_1"], ["decode_1"])

        self._send_and_assert()

    def test_03_kill_single_decode(self):
        """Func_03: 1P2D service works; after killing the router and one
        decode, remaining workers stay alive and a 1P1D router restores
        request serving.
        """
        self._launch_router(["prefill_1"], ["decode_1", "decode_2"])
        self._send_and_assert()

        self._kill_router()
        self._kill_worker("decode_2")

        # Remaining workers unaffected (prefill_2 was killed in test_02)
        self._assert_workers_healthy(["prefill_1", "decode_1"])

        # Restart router as 1P1D
        self._launch_router(["prefill_1"], ["decode_1"])

        self._send_and_assert()


if __name__ == "__main__":
    unittest.main()
