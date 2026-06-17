"""Integration tests for tracing on NPU with a lightweight in-process OTLP collector.

This module validates that the --enable-trace flag works correctly on NPU
by starting a real sglang server and verifying that spans are exported to
an in-memory OTLP collector.
"""

import os
import time
import unittest

import requests

from sglang.srt.observability.req_time_stats import RequestStage
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.otel_collector import LightweightOtlpCollector

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)

# Pre-computed expected span names for each trace level
EXPECTED_SPANS_LEVEL_1 = [
    RequestStage.PREFILL_FORWARD.stage_name,
    RequestStage.DECODE_FORWARD.stage_name,
]

EXPECTED_SPANS_LEVEL_2 = EXPECTED_SPANS_LEVEL_1 + [
    RequestStage.REQUEST_PROCESS.stage_name,
]

EXPECTED_SPANS_LEVEL_3 = EXPECTED_SPANS_LEVEL_2 + [
    RequestStage.DECODE_LOOP.stage_name,
]


class TestNPUTracing(TestNPULoggingBase):
    """Test tracing functionality on single NPU.

    [Description]
        Validates that --enable-trace exports spans to the configured
        OTLP endpoint and that different trace levels control span
        granularity correctly.

    [Test Category] Functionality
    [Test Target] --enable-trace; --otlp-traces-endpoint
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(
            [
                "--enable-trace",
                "--otlp-traces-endpoint",
                "127.0.0.1:4317",
            ]
        )
        # Speed up OTLP export for faster test execution
        cls.env = {
            "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": "50",
            "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": "4",
        }
        cls.launch_server()

    def setUp(self):
        """Start a fresh collector before each test."""
        super().setUp()
        self.collector = LightweightOtlpCollector(port=4317)
        self.collector.start()
        time.sleep(0.2)

    def tearDown(self):
        """Stop the collector after each test."""
        if self.collector:
            self.collector.stop()
            self.collector = None
        super().tearDown()

    def _send_request_and_wait(self, text, max_new_tokens=32, trace_level=None):
        """Send a generate request and wait for spans to be collected."""
        if trace_level is not None:
            response = requests.get(
                f"{self.base_url}/set_trace_level?level={trace_level}"
            )
            self.assertEqual(response.status_code, 200)
            self.collector.clear()

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "stream": True,
            },
            stream=True,
        )
        for _ in response.iter_lines(decode_unicode=False):
            pass

        time.sleep(1)

    def _wait_for_spans_to_drain(self):
        """Wait until no new spans arrive for several consecutive checks."""
        max_wait_seconds = 10
        check_interval = 0.2
        elapsed = 0
        consecutive_zero_count = 0
        required_consecutive_zeros = 3

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

    def test_trace_level_0(self):
        """Trace level 0 should not export any spans."""
        self._send_request_and_wait("Hello world", max_new_tokens=5, trace_level=0)
        self.assertEqual(
            self.collector.count_spans(),
            0,
            f"Spans collected but expected none: {sorted(self.collector.get_span_names())}",
        )

    def test_trace_level_1(self):
        """Trace level 1 should export basic prefill and decode spans."""
        self._send_request_and_wait("The capital of France is", trace_level=1)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected but expected some",
        )

        span_names = self.collector.get_span_names()
        matched = [name for name in EXPECTED_SPANS_LEVEL_1 if name in span_names]
        self.assertGreater(
            len(matched),
            0,
            f"No expected spans found. Expected any of {EXPECTED_SPANS_LEVEL_1}, "
            f"got {sorted(span_names)}",
        )

    def test_trace_level_2(self):
        """Trace level 2 should export more detailed spans including request_process."""
        self._send_request_and_wait("What is AI?", trace_level=2)

        span_names = self.collector.get_span_names()
        matched = [name for name in EXPECTED_SPANS_LEVEL_2 if name in span_names]
        self.assertGreater(
            len(matched),
            0,
            f"No expected spans found. Expected any of {EXPECTED_SPANS_LEVEL_2}, "
            f"got {sorted(span_names)}",
        )

    def test_trace_level_3(self):
        """Trace level 3 should export the most detailed spans including decode_loop."""
        self._send_request_and_wait("Explain quantum computing", trace_level=3)

        span_names = self.collector.get_span_names()
        matched = [name for name in EXPECTED_SPANS_LEVEL_3 if name in span_names]
        self.assertGreater(
            len(matched),
            0,
            f"No expected spans found. Expected any of {EXPECTED_SPANS_LEVEL_3}, "
            f"got {sorted(span_names)}",
        )

    def test_batch_request(self):
        """Batch requests with distinct prompts should produce one prefill span per prompt."""
        response = requests.get(f"{self.base_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        batch_size = 4
        # Use distinct prompts to avoid radix-cache merging that would
        # collapse multiple requests into a single prefill span.
        prompts = [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
            "The capital of Spain is",
        ]
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                },
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(0.5)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from batch request",
        )

        all_spans = self.collector.get_spans()
        request_spans = [
            s for s in all_spans if s.name == RequestStage.PREFILL_FORWARD.stage_name
        ]
        self.assertEqual(
            len(request_spans),
            batch_size,
            f"Expected {batch_size} prefill_forward spans, got {len(request_spans)}",
        )

    def test_parallel_sample(self):
        """Parallel sampling should produce at least one prefill span."""
        response = requests.get(f"{self.base_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        parallel_num = 4
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.5,
                    "max_new_tokens": 10,
                    "n": parallel_num,
                },
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(0.5)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from parallel sample request",
        )

        all_spans = self.collector.get_spans()
        request_spans = [
            s for s in all_spans if s.name == RequestStage.PREFILL_FORWARD.stage_name
        ]
        self.assertGreaterEqual(
            len(request_spans),
            1,
            f"Expected at least 1 prefill_forward span, got {len(request_spans)}",
        )


if __name__ == "__main__":
    unittest.main()
