import json
import os
import time
import unittest
from pathlib import Path

import requests
import zmq

from sglang.test.ascend.test_ascend_utils import logger
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)


class TestNPUMetricsMFUEnabled(TestNPULoggingBase):
    """
    NPU integration test for forward pass metrics (FPM).

    Verifies:
    - Forward-pass metrics are emitted over ZMQ IPC
    - worker_id is overridden by the runtime
    - Per-step identifier exists
    - Prometheus metrics remain unaffected
    """

    # -----------------------------
    # Constants
    # -----------------------------
    ipc_path = f"/tmp/sglang-test-fwd-metrics-{os.getpid()}"
    ipc_endpoint = f"ipc://{ipc_path}"

    zmq_rcv_timeout_ms = 5000
    metric_recv_timeout_sec = 20

    metrics_args = [
        "--enable-forward-pass-metrics",
        "--forward-pass-metrics-worker-id",
        "should-be-overridden",
        "--forward-pass-metrics-ipc-name",
        ipc_endpoint,
    ]

    # -----------------------------
    # Class-level state
    # -----------------------------
    _zmq_ctx: zmq.Context | None = None
    _zmq_sub: zmq.Socket | None = None

    # -----------------------------
    # Setup / Teardown
    # -----------------------------
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.other_args.extend(cls.metrics_args)
        cls.launch_server()

        cls._zmq_ctx = zmq.Context()
        cls._zmq_sub = cls._zmq_ctx.socket(zmq.SUB)
        cls._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        cls._zmq_sub.setsockopt(zmq.RCVTIMEO, cls.zmq_rcv_timeout_ms)
        cls._zmq_sub.connect(f"{cls.ipc_endpoint}.0")

        logger.info(
            "ZMQ SUB connected to %s.0 (timeout=%dms)",
            cls.ipc_endpoint,
            cls.zmq_rcv_timeout_ms,
        )

    @classmethod
    def tearDownClass(cls):
        if cls._zmq_sub is not None:
            cls._zmq_sub.close(linger=0)
        if cls._zmq_ctx is not None:
            cls._zmq_ctx.term()
        super().tearDownClass()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _recv_fpm_metric(self, timeout: float = metric_recv_timeout_sec) -> dict:
        """Receive a valid JSON forward-pass metric via ZMQ."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                frames = self._zmq_sub.recv_multipart(flags=zmq.NOBLOCK)
                logger.info("FPM ZMQ frames: %r", frames)
            except zmq.Again:
                time.sleep(0.05)
                continue

            logger.debug("ZMQ frames: %s", [f[:128] for f in frames])

            for frame in frames:
                if not frame:
                    # Skip b'' (topic / heartbeat / delimiter)
                    continue
                try:
                    obj = json.loads(frame)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    logger.debug("Ignore non-JSON ZMQ frame: %r", frame[:128])

        self.fail(
            f"No valid JSON forward-pass metric received on "
            f"{self.ipc_endpoint}.0 within {timeout}s"
        )

    # -----------------------------
    # Test case
    # -----------------------------
    def test_forward_pass_metrics_all_args_configured(self):
        # Trigger forward pass
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": ["The capital of France is"] * 2,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
                "stream": False,
                "ignore_eos": True,
            },
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200, resp.text)

        # Receive FPM metric
        metric = self._recv_fpm_metric()

        # -----------------------------
        # Core assertions
        # -----------------------------
        self.assertTrue(
            Path(self.ipc_path).is_socket(),
            "Forward-pass metrics IPC socket should exist",
        )

        worker_id = metric.get("worker_id")
        self.assertIsInstance(worker_id, str)
        self.assertTrue(worker_id, "worker_id must not be empty")
        self.assertNotEqual(
            worker_id,
            "should-be-overridden",
            "Runtime must override forward_pass_metrics_worker_id",
        )

        self.assertTrue(
            "step_id" in metric or "iter" in metric,
            "Per-step identifier missing from forward-pass metric",
        )

        # -----------------------------
        # Prometheus orthogonality check
        # -----------------------------
        prom_resp = requests.get(f"{self.base_url}/metrics", timeout=5)
        self.assertTrue(prom_resp.ok)
        self.assertTrue(
            prom_resp.text.lstrip().startswith("#"),
            "Prometheus metrics must use text exposition format",
        )


if __name__ == "__main__":
    unittest.main()
