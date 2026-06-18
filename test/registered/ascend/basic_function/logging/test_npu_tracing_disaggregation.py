"""Test tracing in PD disaggregation mode on NPU.

This module validates that tracing works correctly in PD disaggregation mode
on NPU by starting prefill and decode servers and verifying that spans are
exported to an in-memory OTLP collector.
"""

import os

# Configure OTLP exporter for faster test execution
# Must be set before importing sglang trace module
os.environ.setdefault("SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "50")
os.environ.setdefault("SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "4")

import logging
import time
import unittest

import requests

from sglang.srt.observability.req_time_stats import RequestStage
from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.otel_collector import LightweightOtlpCollector
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

logger = logging.getLogger(__name__)

# CI registration - PD disaggregation requires 2 NPUs
register_npu_ci(est_time=120, suite="full-2-npu-a3", nightly=True)


class TestNPUTracingDisaggregation(TestDisaggregationBase):
    """Test tracing in PD disaggregation mode on NPU.

    [Description]
        Validates that --enable-trace exports spans correctly in PD
        disaggregation mode, including disaggregation-specific transfer spans.

    [Test Category] Functionality
    [Test Target] --enable-trace; --otlp-traces-endpoint; PD disaggregation
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = QWEN3_0_6B_WEIGHTS_PATH

        # Initialize collector first
        cls.collector = LightweightOtlpCollector()
        cls.collector.start()
        time.sleep(0.2)

        # Start prefill and decode servers, then launch LB
        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")
        cls.launch_lb()

        # Wait for warmup spans to be exported and clear them
        time.sleep(1)
        cls.collector.clear()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "1",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env={
                "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": "50",
                "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": "4",
            },
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "1",
            "--base-gpu-id",
            "1",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices

        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env={
                "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": "50",
                "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": "4",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.collector:
            cls.collector.stop()
            cls.collector = None
        super().tearDownClass()

    def setUp(self):
        """Wait for spans to be drained before each test."""
        max_wait_seconds = 10
        check_interval = 0.2
        elapsed = 0
        consecutive_zero_count = 0
        required_consecutive_zeros = 3

        # Poll the collector until no new spans arrive for several
        # consecutive checks, ensuring leftover spans from previous
        # tests are fully drained before the current test starts.
        while elapsed < max_wait_seconds:
            span_count = self.collector.count_spans()
            if span_count == 0:
                consecutive_zero_count += 1
                if consecutive_zero_count >= required_consecutive_zeros:
                    break
            else:
                consecutive_zero_count = 0
                self.collector.clear()
            time.sleep(check_interval)
            elapsed += check_interval
        else:
            raise RuntimeError(
                f"Timeout waiting for spans to drain after {max_wait_seconds}s. "
                f"Remaining spans: {self.collector.count_spans()}"
            )

    def test_disaggregation_transfer_spans(self):
        """Test that disaggregation produces transfer-related spans."""
        # Set trace level on both prefill and decode servers
        response = requests.get(
            f"{self.prefill_url}/set_trace_level?level=1"
        )
        self.assertEqual(response.status_code, 200)
        response = requests.get(
            f"{self.decode_url}/set_trace_level?level=1"
        )
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        # Send a request through load balancer
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                },
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        # Wait for async export
        time.sleep(1)

        # Verify spans were collected
        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from disaggregation request",
        )

        # Verify disaggregation-specific spans exist
        span_names = self.collector.get_span_names()

        # Check for transfer-related spans
        self.assertTrue(
            self.collector.has_any_span(
                [
                    RequestStage.PREFILL_TRANSFER_KV_CACHE.stage_name,
                    RequestStage.DECODE_TRANSFERRED.stage_name,
                ]
            ),
            f"Expected disaggregation transfer spans, got {sorted(span_names)}",
        )


if __name__ == "__main__":
    unittest.main()
