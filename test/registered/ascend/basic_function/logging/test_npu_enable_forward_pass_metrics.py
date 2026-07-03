import json
import os
import unittest
from pathlib import Path

import requests
import zmq

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)


class TestNPUMetricsMFUEnabled(TestNPULoggingBase):
    """
    NPU integration test for forward pass metrics with full argument coverage.
    """

    # -----------------------------
    # Constants
    # -----------------------------
    ipc_name = f"/tmp/sglang-test-fwd-metrics-{os.getpid()}"
    ipc_url = f"ipc://{ipc_name}"

    zmq_rcv_timeout_ms = 5000

    metrics_args = [
        "--enable-forward-pass-metrics",
        "--forward-pass-metrics-worker-id",
        "should-be-overridden",
        "--forward-pass-metrics-ipc-name",
        ipc_name,
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
        # Inject forward-pass metrics arguments before server startup
        cls.other_args.extend(cls.metrics_args)

        # launch_server() already performs startup + health check
        cls.launch_server()

        # ZMQ subscriber setup
        cls._zmq_ctx = zmq.Context()
        cls._zmq_sub = cls._zmq_ctx.socket(zmq.SUB)
        cls._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        cls._zmq_sub.setsockopt(zmq.RCVTIMEO, cls.zmq_rcv_timeout_ms)
        cls._zmq_sub.connect(cls.ipc_url)

    @classmethod
    def tearDownClass(cls):
        if cls._zmq_sub is not None:
            cls._zmq_sub.close(linger=0)
        if cls._zmq_ctx is not None:
            cls._zmq_ctx.term()

        super().tearDownClass()

    # -----------------------------
    # Test case
    # -----------------------------
    def test_forward_pass_metrics_all_args_configured(self):
        ipc_path = Path(self.ipc_name)

        # --- Trigger forward pass ---
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

        # --- Receive ZMQ metric ---
        try:
            metric = json.loads(self._zmq_sub.recv_string())
        except zmq.Again:
            self.fail(
                f"No forward-pass metrics received on {self.ipc_url}. "
                f"IPC exists: {ipc_path.exists()}, server: {self.base_url}"
            )

        # --- Core assertions ---
        self.assertTrue(ipc_path.is_socket(), "IPC socket should exist")

        worker_id = metric.get("worker_id")
        self.assertIsInstance(worker_id, str)
        self.assertTrue(worker_id, "worker_id must not be empty")
        self.assertNotEqual(
            worker_id,
            "should-be-overridden",
            "Framework must override forward_pass_metrics_worker_id",
        )

        self.assertTrue(
            "step_id" in metric or "iter" in metric,
            "Per-step identifier missing from forward-pass metric",
        )

        # --- Prometheus orthogonality check ---
        prom_resp = requests.get(f"{self.base_url}/metrics", timeout=5)
        self.assertTrue(prom_resp.ok)
        self.assertTrue(
            prom_resp.text.lstrip().startswith("#"),
            "Prometheus metrics must use text exposition format",
        )


if __name__ == "__main__":
    unittest.main()
