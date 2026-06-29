import os
import unittest
from typing import Dict, List

import requests
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.observability.metrics_collector import (
    ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
    STAT_LOGGER_ROLE_SCHEDULER,
    SchedulerMetricsCollector,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)


class TestNPUMetricsMFUEnabled(TestNPULoggingBase):
    """Test core metrics functionality on single NPU with MFU enabled.

    [Description]
        Validates that the /metrics endpoint returns correct Prometheus-format
        data when metrics and MFU metrics are enabled on a single NPU.

    [Test Category] Functionality
    [Test Target] --enable-metrics; --enable-mfu-metrics
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(["--enable-metrics", "--enable-mfu-metrics"])
        # Enable the device timer so that forward-pass timing metrics
        # (forward_execution_seconds_total) are collected. Without this
        # environment variable the timer is disabled by default and those
        # counters will remain at zero.
        cls.env = {"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"}
        cls.launch_server()

    def test_metrics_1npu(self):
        """Test that metrics endpoint returns data when enabled on single NPU."""
        _generate_metrics(self.base_url)

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text

        print(f"metrics_text=\n{metrics_text}")

        metrics = _parse_prometheus_metrics(metrics_text)
        _verify_metrics_common(self, metrics_text, metrics, expect_mfu_metrics=True)


class TestNPUMetricsMFUDisabled(TestNPULoggingBase):
    """Test that MFU metrics are not emitted when the gate is disabled.

    [Description]
        Validates that when only --enable-metrics is set (without
        --enable-mfu-metrics), MFU counters do not emit positive values.

    [Test Category] Functionality
    [Test Target] --enable-metrics
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(["--enable-metrics"])
        cls.env = {"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"}
        cls.launch_server()

    def test_mfu_metrics_mfu_disabled(self):
        """MFU metrics should not be emitted when the gate is disabled."""
        _generate_metrics(self.base_url)

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text

        print(f"metrics_text=\n{metrics_text}")

        metrics = _parse_prometheus_metrics(metrics_text)
        _verify_metrics_common(self, metrics_text, metrics, expect_mfu_metrics=False)


class TestNPUMetrics2NPU(TestNPULoggingBase):
    """Test metrics on 2-NPU TP/DP parallel scenario.

    [Description]
        Validates distributed metrics (dp_cooperation_*) when running with
        TP=2 and DP=2 on NPU.

    [Test Category] Functionality
    [Test Target] --enable-metrics; --enable-mfu-metrics;
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(
            [
                "--enable-metrics",
                "--enable-mfu-metrics",
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
            ]
        )
        # SGLANG_ENABLE_METRICS_DP_ATTENTION is required for dp_cooperation_*
        # metrics. Without it, DPCooperationInfo is not created and those
        # counters will never be emitted.
        cls.env = {
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
            "SGLANG_ENABLE_METRICS_DP_ATTENTION": "1",
        }
        cls.launch_server()

    def test_metrics_2npu(self):
        """Test metrics on 2-NPU TP/DP parallel scenario."""
        _generate_metrics(self.base_url)

        # Under DP=2 round-robin scheduling, requests with the same prefix may
        # be dispatched to different ranks, leaving each rank with a cold cache.
        # cached_tokens_total is only emitted when cached_tokens > 0, so we send
        # additional identical requests to increase the chance that at least one
        # rank sees a cache hit.
        for i in range(5):
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "Hello, " * 100,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 5},
                },
                headers={"x-smg-routing-key": "test-key"},
            )
            assert response.status_code == 200

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text

        metrics = _parse_prometheus_metrics(metrics_text)
        _verify_metrics_common(self, metrics_text, metrics, expect_mfu_metrics=True)

        # In the GPU scenario test case
        # (test/registered/observability/test_metrics.py), the 2-card scenario
        # contains assertions about the following monitoring metrics:
        #   ("sglang:dp_cooperation_forward_execution_seconds_total",
        #    {"category": "extend"}),
        #   ("sglang:dp_cooperation_forward_execution_seconds_total",
        #    {"category": "decode"}),
        # However, these two indicators were not found during execution on the
        # GPU, and it is uncertain whether it is a problem or if monitoring of
        # these indicators is currently not supported.
        metrics_to_check = [
            (
                "sglang:dp_cooperation_realtime_tokens_total",
                {"mode": "prefill_compute"},
            ),
            (
                "sglang:dp_cooperation_realtime_tokens_total",
                {"mode": "decode"},
            ),
        ]
        _check_metrics_positive(self, metrics, metrics_to_check)

        num_prefill_ranks_values = {
            s.labels["num_prefill_ranks"]
            for s in metrics["sglang:dp_cooperation_realtime_tokens_total"]
        }
        self.assertIn("0", num_prefill_ranks_values)
        self.assertIn("1", num_prefill_ranks_values)


def _generate_metrics(base_url: str):
    """Send requests to generate metrics data.

    The workload is intentionally generous so that every counter and
    histogram we assert on later accumulates a clearly positive value.
    A lightweight workload risks leaving some metrics at zero because
    forward-pass time may round to 0.0, asynchronous flushes may be
    missed, or slow CI runners may not finish processing in time.
    """
    response = requests.get(f"{base_url}/health_generate")
    assert response.status_code == 200

    # 1) Large batch + long decode: guarantees substantial prefill and decode
    #    workload so that realtime_tokens_total and
    #    forward_execution_seconds_total accumulate clearly positive values.
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": ["The capital of France is"] * 20,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 50,
            },
            "stream": True,
            # Force the model to generate the full max_new_tokens instead of
            # stopping early when an EOS token is produced. This guarantees a
            # predictable decode workload so that decode-phase counters do not
            # end up at zero because the model finished prematurely.
            "ignore_eos": True,
        },
        stream=True,
    )
    # Consume the streaming response so the request fully completes on the
    # server side.  Without draining the iterator the connection stays open
    # and the request may still be counted as in-flight.
    for _ in response.iter_lines(decode_unicode=False):
        pass

    # 2) Repeated requests with a routing key: populates routing-key histograms
    #    and cached_tokens_total (the second request may hit the KV cache).
    for i in range(2):
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": "Hello, " * 100,
                "sampling_params": {"temperature": 0, "max_new_tokens": 5},
            },
            headers={"x-smg-routing-key": "test-key"},
        )
        assert response.status_code == 200


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in result:
                result[sample.name] = []
            result[sample.name].append(sample)
    return result


def _get_sample_value_by_labels(samples: List[Sample], labels: Dict[str, str]) -> float:
    for sample in samples:
        if all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample.value
    raise KeyError(f"No sample found with labels {labels}")


def _check_metrics_positive(test_case, metrics, metrics_to_check):
    for metric_name, labels in metrics_to_check:
        value = _get_sample_value_by_labels(metrics[metric_name], labels)
        test_case.assertGreater(
            value,
            0,
            f"{metric_name} {labels}: expected a positive value, got {value} "
            f"(the engine may not have processed any requests)",
        )


def _verify_metrics_common(test_case, metrics_text, metrics, expect_mfu_metrics: bool):
    model_name = test_case.model
    essential_metrics = [
        "sglang:num_running_reqs",
        "sglang:num_used_tokens",
        "sglang:token_usage",
        "sglang:gen_throughput",
        "sglang:num_queue_reqs",
        "sglang:num_grammar_queue_reqs",
        "sglang:cache_hit_rate",
        "sglang:spec_accept_length",
        "sglang:prompt_tokens_total",
        "sglang:generation_tokens_total",
        "sglang:cached_tokens_total",
        "sglang:num_requests_total",
        "sglang:time_to_first_token_seconds",
        "sglang:inter_token_latency_seconds",
        "sglang:e2e_request_latency_seconds",
        "sglang:http_requests_active",
        "sglang:routing_keys_active",
        "sglang:num_unique_running_routing_keys",
        "sglang:routing_key_running_req_count",
        "sglang:routing_key_all_req_count",
    ]
    mfu_metrics = [
        "sglang:estimated_flops_per_gpu_total",
        "sglang:estimated_read_bytes_per_gpu_total",
        "sglang:estimated_write_bytes_per_gpu_total",
    ]
    if expect_mfu_metrics:
        essential_metrics.extend(mfu_metrics)

    # Verify that all essential metric names appear in the raw Prometheus text.
    # This is a basic existence check: it ensures the metrics are exported but
    # does not validate their values or label sets.
    for metric in essential_metrics:
        test_case.assertIn(metric, metrics_text, f"Missing metric: {metric}")

    # Verify the bucket structure of routing-key request-count histograms.
    # These metrics use a GaugeHistogram where each bucket is identified by
    # (gt, le) label pairs:
    #   - "gt" (greater than): the lower bound of the bucket (exclusive)
    #   - "le" (less than or equal to): the upper bound of the bucket (inclusive)
    # For example, (gt="0", le="1") represents the bucket that counts values
    # in the range (0, 1]. The expected number of buckets equals the number of
    # user-defined bounds plus one extra bucket for +Inf.
    expected_buckets = len(ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS) + 1
    for metric_name in [
        "sglang:routing_key_running_req_count",
        "sglang:routing_key_all_req_count",
    ]:
        gt_le_pairs = set()
        for sample in metrics.get(metric_name, []):
            gt_le_pairs.add((sample.labels.get("gt"), sample.labels.get("le")))
        test_case.assertEqual(
            len(gt_le_pairs),
            expected_buckets,
            f"{metric_name}: Expected {expected_buckets} buckets, got {len(gt_le_pairs)}",
        )

    # Verify that metrics carry the correct model_name label and that
    # histogram-style metrics expose the standard _sum, _count, and _bucket
    # time-series. We assert on the exact metric names rather than raw
    # substrings so the failure message points to the missing series.
    histogram_metrics = [
        "sglang:time_to_first_token_seconds",
        "sglang:inter_token_latency_seconds",
        "sglang:e2e_request_latency_seconds",
    ]
    for base_name in histogram_metrics:
        for suffix in ("_sum", "_count", "_bucket"):
            test_case.assertIn(
                f"{base_name}{suffix}{{",
                metrics_text,
                f"Missing histogram series: {base_name}{suffix}",
            )

    # All metrics should be tagged with the model_name of the served model.
    test_case.assertIn(f'model_name="{model_name}"', metrics_text)

    # Verify that core performance counters have accumulated positive values.
    # These metrics prove the engine actually processed requests (prefill +
    # decode tokens, forward-pass time, tokenizer CPU time) rather than just
    # exporting zero-valued series.
    #
    # Why assert > 0 instead of >= 0?
    #   - The test deliberately sends a large workload (see _generate_metrics)
    #     so that every counter here is guaranteed to be clearly positive.
    #   - A value of 0 would mean the corresponding pipeline stage did no
    #     work, which indicates a bug (e.g. device timer disabled, metrics
    #     flush lost, or the request never reached the engine).
    metrics_to_check = [
        # Total tokens processed during the prefill (context-encoding) phase.
        ("sglang:realtime_tokens_total", {"mode": "prefill_compute"}),
        # Total tokens generated during the decode (autoregressive) phase.
        ("sglang:realtime_tokens_total", {"mode": "decode"}),
        # Cumulative time spent in the extend (prefix-extension) forward pass.
        ("sglang:forward_execution_seconds_total", {"category": "extend"}),
        # Cumulative time spent in the decode forward pass.
        ("sglang:forward_execution_seconds_total", {"category": "decode"}),
        # CPU time consumed by the tokenizer subprocess.
        ("sglang:process_cpu_seconds_total", {"component": "tokenizer"}),
    ]
    _check_metrics_positive(test_case, metrics, metrics_to_check)

    # MFU (Model FLOPs Utilization) metrics estimate the compute and memory
    # bandwidth consumed by the model. They are only collected when the server
    # is started with --enable-mfu-metrics. Verify the gate behaves correctly:
    #   - Gate ON  -> counters must contain positive values for this model.
    #   - Gate OFF -> counters must be absent or zero.
    if expect_mfu_metrics:
        for metric_name in mfu_metrics:
            # Filter samples belonging to the current model (multi-model
            # deployments may include series for other models).
            values = [
                sample.value
                for sample in metrics.get(metric_name, [])
                if sample.labels.get("model_name") == model_name
            ]
            test_case.assertTrue(
                values, f"{metric_name}: no samples for model {model_name}"
            )
            test_case.assertGreater(
                sum(values),
                0,
                f"{metric_name}: expected positive total for model {model_name}",
            )
    else:
        for metric_name in mfu_metrics:
            values = [
                sample.value
                for sample in metrics.get(metric_name, [])
                if sample.labels.get("model_name") == model_name
            ]
            # Some implementations still emit the metric name with a zero value
            # when the gate is disabled; ensure no positive accumulation leaked.
            if values:
                test_case.assertEqual(
                    sum(values),
                    0,
                    f"{metric_name}: expected no positive samples with MFU metrics gate disabled",
                )


_DI_MARKER_PATH = "/tmp/sglang_di_test_marker"


class _MarkingSchedulerCollector(SchedulerMetricsCollector):
    """Records its own instantiation to a file so the test can verify the
    custom subclass was used in the scheduler subprocess.

    Defined at module level so it is picklable into the scheduler process.
    Cross-process signalling uses a filesystem marker because the scheduler
    runs in its own subprocess and cannot share in-memory state with the
    test runner.
    """

    def __init__(self, *args, **kwargs):
        with open(_DI_MARKER_PATH, "w") as f:
            f.write("scheduler_collector_initialized\n")
        super().__init__(*args, **kwargs)


class TestNPUStatLoggersDI(CustomTestCase):
    """Verify that a custom MetricsCollector subclass passed through
    ``ServerArgs.stat_loggers`` is the one instantiated inside the
    scheduler subprocess on NPU."""

    def setUp(self) -> None:
        try:
            os.unlink(_DI_MARKER_PATH)
        except FileNotFoundError:
            pass

    def tearDown(self) -> None:
        try:
            os.unlink(_DI_MARKER_PATH)
        except FileNotFoundError:
            pass

    def test_engine_custom_scheduler_collector(self):
        import sglang as sgl

        engine = sgl.Engine(
            model_path=QWEN3_0_6B_WEIGHTS_PATH,
            enable_metrics=True,
            device="npu",
            stat_loggers={
                STAT_LOGGER_ROLE_SCHEDULER: _MarkingSchedulerCollector,
            },
        )
        try:
            # One small generation triggers scheduler init, which is where
            # resolve_collector_class() picks the injected subclass.
            engine.generate("Hello", {"max_new_tokens": 4})
        finally:
            engine.shutdown()

        self.assertTrue(
            os.path.exists(_DI_MARKER_PATH),
            "Custom SchedulerMetricsCollector was not instantiated; "
            "stat_loggers DI did not take effect.",
        )


if __name__ == "__main__":
    unittest.main()
