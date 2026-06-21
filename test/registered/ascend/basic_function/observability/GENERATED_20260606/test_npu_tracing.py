"""Integration tests for tracing with a lightweight in-process OTLP collector.

This module implements a minimal OTLP collector that receives traces via gRPC
and stores them in memory for test assertions, eliminating the need for
Docker-based opentelemetry-collector and file I/O.
"""

import os

os.environ.setdefault("SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "50")
os.environ.setdefault("SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "4")

import logging
import multiprocessing as mp
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional, Union

import requests
import zmq

from sglang import Engine
from sglang.srt.observability.req_time_stats import RequestStage
from sglang.srt.observability.trace import (
    TraceReqContext,
    TraceSliceContext,
    get_cur_time_ns,
    process_tracing_init,
    set_global_trace_level,
    trace_set_thread_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_zmq_socket
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.otel_collector import LightweightOtlpCollector
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logger = logging.getLogger(__name__)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


def _get_span_names_by_level(level: int) -> List[str]:
    span_names = []
    for attr_name in dir(RequestStage):
        if attr_name.startswith("_"):
            continue
        attr = getattr(RequestStage, attr_name)
        if hasattr(attr, "stage_name") and hasattr(attr, "level"):
            if attr.level <= level and attr.stage_name:
                span_names.append(attr.stage_name)
    return span_names


SPAN_NAMES_LEVEL_1 = _get_span_names_by_level(1)
SPAN_NAMES_LEVEL_2 = _get_span_names_by_level(2)
SPAN_NAMES_LEVEL_3 = _get_span_names_by_level(3)

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


@dataclass
class Req:
    rid: int
    req_context: Optional[Union[TraceReqContext]] = None


def _subprocess_worker():
    process_tracing_init("127.0.0.1:4317", "test")
    trace_set_thread_info("Sub Process")

    context = zmq.Context(2)
    recv_from_main = get_zmq_socket(context, zmq.PULL, "ipc:///tmp/zmq_test.ipc", True)

    try:
        req = recv_from_main.recv_pyobj()
        req.req_context.rebuild_thread_context()
        req.req_context.trace_slice_start("work", level=1)
        time.sleep(0.2)
        req.req_context.trace_slice_end("work", level=1, thread_finish_flag=True)
    finally:
        recv_from_main.close()
        context.term()


class TestNPUTracePackage(CustomTestCase):
    """Unit tests for tracing package API without server/engine.

    [Test Category] Observability
    [Test Target] Trace API, OTLP exporter
    """

    def setUp(self):
        self.collector = None

    def tearDown(self):
        if self.collector:
            self.collector.stop()
            self.collector = None

    def _start_collector(self):
        self.collector = LightweightOtlpCollector()
        self.collector.start()
        time.sleep(0.2)

    def test_slice_simple(self):
        self._start_collector()

        try:
            process_tracing_init("127.0.0.1:4317", "test")
            trace_set_thread_info("Test")
            set_global_trace_level(3)
            req_context = TraceReqContext(0)
            req_context.trace_req_start()
            req_context.trace_slice_start("test slice", level=1)
            time.sleep(0.1)
            req_context.trace_slice_end("test slice", level=1)
            req_context.trace_req_finish()

            time.sleep(0.3)

            self.assertTrue(
                self.collector.has_span("test slice"),
                f"Expected span 'test slice', got {self.collector.get_span_names()}",
            )
        finally:
            pass

    def test_slice_complex(self):
        self._start_collector()

        try:
            process_tracing_init("127.0.0.1:4317", "test")
            trace_set_thread_info("Test")
            set_global_trace_level(3)
            req_context = TraceReqContext(0)
            req_context.trace_req_start()

            t1 = get_cur_time_ns()
            time.sleep(0.1)
            req_context.trace_event("event test", 1)
            t2 = get_cur_time_ns()
            time.sleep(0.1)
            t3 = get_cur_time_ns()

            slice1 = TraceSliceContext("slice A", t1, t2)
            slice2 = TraceSliceContext("slice B", t2, t3)
            req_context.trace_slice(slice1)
            req_context.trace_slice(slice2, thread_finish_flag=True)
            req_context.trace_req_finish()

            time.sleep(0.3)

            self.assertTrue(
                self.collector.has_all_spans(["slice A", "slice B"]),
                f"Expected spans 'slice A' and 'slice B', got {self.collector.get_span_names()}",
            )
        finally:
            pass

    def test_context_propagate(self):
        self._start_collector()

        ctx = mp.get_context("spawn")

        context = zmq.Context(2)
        send_to_subproc = get_zmq_socket(
            context, zmq.PUSH, "ipc:///tmp/zmq_test.ipc", False
        )

        try:
            process_tracing_init("127.0.0.1:4317", "test")
            trace_set_thread_info("Main Process")

            subproc = ctx.Process(target=_subprocess_worker)
            subproc.start()

            time.sleep(0.3)

            req = Req(rid=0)
            req.req_context = TraceReqContext(0)
            req.req_context.trace_req_start()
            req.req_context.trace_slice_start("dispatch", level=1)
            time.sleep(0.2)
            send_to_subproc.send_pyobj(req)
            req.req_context.trace_slice_end("dispatch", level=1)

            subproc.join()
            req.req_context.trace_req_finish()

            time.sleep(0.5)

            self.assertTrue(
                self.collector.has_all_spans(["dispatch", "work"]),
                f"Expected spans 'dispatch' and 'work', got {self.collector.get_span_names()}",
            )
        finally:
            send_to_subproc.close()
            context.term()


class TestNPUTraceServer(CustomTestCase):
    """Integration tests for tracing with server - starts server once for all tests.

    [Test Category] Observability
    [Test Target] Trace integration, server startup
    """

    @classmethod
    def setUpClass(cls):
        cls.collector = LightweightOtlpCollector()
        cls.collector.start()
        time.sleep(0.2)

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-trace",
                "--otlp-traces-endpoint",
                "127.0.0.1:4317",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.3",
            ],
        )

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        assert response.status_code == 200

        cls.collector.clear()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
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

    def _send_request_and_wait(
        self, text, max_new_tokens=32, stream=True, trace_level=None
    ):
        if trace_level is not None:
            response = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/set_trace_level?level={trace_level}"
            )
            self.assertEqual(response.status_code, 200)
            self.collector.clear()

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "stream": stream,
            },
            stream=stream,
        )
        if stream:
            for _ in response.iter_lines(decode_unicode=False):
                pass
        else:
            self.assertEqual(response.status_code, 200)

        time.sleep(1)

    def test_trace_level_0(self):
        self._send_request_and_wait("Hello world", max_new_tokens=5, trace_level=0)
        self.assertEqual(
            self.collector.count_spans(),
            0,
            f"Spans collected but expected none: {sorted(self.collector.get_span_names())}",
        )

    def test_trace_level_1(self):
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
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        batch_size = 4
        prompts = ["The capital of France is"] * batch_size
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        parallel_num = 4
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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


class TestNPUTraceEngine(CustomTestCase):
    """Integration tests for tracing with Engine API - each test creates its own engine.

    [Test Category] Observability
    [Test Target] Engine trace API
    """

    def setUp(self):
        self.collector = None

    def tearDown(self):
        if self.collector:
            self.collector.stop()
            self.collector = None

    def _start_collector(self):
        self.collector = LightweightOtlpCollector()
        self.collector.start()
        time.sleep(0.2)

    def test_trace_engine_enable(self):
        self._start_collector()

        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
        )

        try:
            engine.generate(prompt, sampling_params)
            time.sleep(0.5)

            self.assertGreater(
                self.collector.count_spans(),
                0,
                "No spans collected from Engine.generate",
            )
            self.assertTrue(
                self.collector.has_any_span([RequestStage.PREFILL_FORWARD.stage_name]),
                f"Expected prefill_forward span, got {self.collector.get_span_names()}",
            )
        finally:
            engine.shutdown()

    def test_trace_engine_encode(self):
        self._start_collector()

        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
            is_embedding=True,
        )

        try:
            engine.encode(prompt)
            time.sleep(0.5)

            self.assertGreater(
                self.collector.count_spans(),
                0,
                "No spans collected from Engine.encode",
            )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
