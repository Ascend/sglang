"""Test tracing in PD disaggregation mode."""

import os

os.environ.setdefault("SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "50")
os.environ.setdefault("SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "4")

import logging
import time
import unittest

import requests

from sglang.srt.observability.req_time_stats import RequestStage
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.otel_collector import LightweightOtlpCollector
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestNPUTraceDisaggregation(PDDisaggregationServerBase):
    """Test tracing in PD disaggregation mode on NPU.

    [Test Category] Observability
    [Test Target] Tracing in disaggregation transfer
    """

    @classmethod
    def setUpClass(cls):
        cls.collector = LightweightOtlpCollector()
        cls.collector.start()
        time.sleep(0.2)

        super().setUpClass()
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.extra_prefill_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
        ]
        cls.extra_decode_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
        ]
        cls.launch_all()

        time.sleep(1)
        cls.collector.clear()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls.collector:
            cls.collector.stop()

    def setUp(self):
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

    def test_disaggregation_transfer_spans(self):
        response = requests.get(f"{self.prefill_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        response = requests.get(f"{self.decode_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

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

        time.sleep(1)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from disaggregation request",
        )

        span_names = self.collector.get_span_names()

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
