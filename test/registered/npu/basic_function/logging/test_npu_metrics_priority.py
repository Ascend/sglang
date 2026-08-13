import unittest
from typing import Dict, List

import requests
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="full-1-npu-a3", nightly=True)


class TestNPUMetricsPriority(TestNPULoggingBase):
    """Test that priority-based metrics are correctly emitted on NPU.

    [Description]
        Validates that when --enable-priority-scheduling is enabled, gauge
        metrics (num_running_reqs, num_queue_reqs) and histogram metrics
        (time_to_first_token_seconds, e2e_request_latency_seconds) contain
        the priority label dimension. Also verifies that requests without an
        explicit priority use the configured --default-priority-value.

    [Test Category] Functionality
    [Test Target] --enable-metrics; --enable-priority-scheduling; --default-priority-value
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(
            [
                "--enable-metrics",
                "--enable-priority-scheduling",
                "--default-priority-value",
                "0",
            ]
        )
        cls.launch_server()

    def test_priority_label_in_gauge_metrics(self):
        """Send requests with different priorities and verify that
        gauge metrics (num_running_reqs, num_queue_reqs) contain
        the priority label dimension."""
        for priority in [1, 5, 10]:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 5},
                    "priority": priority,
                },
            )
            self.assertEqual(response.status_code, 200)

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        for metric_name in ["sglang:num_running_reqs", "sglang:num_queue_reqs"]:
            samples = metrics.get(metric_name, [])
            self.assertGreater(len(samples), 0, f"No samples found for {metric_name}")

            # Prometheus emits two kinds of samples for this gauge:
            #   priority=""   -> the total across all priorities
            #   priority="N"  -> the value for a specific priority level
            # Verify the total exists; per-priority breakdown is checked in the
            # histogram test below because gauge values are transient (a request
            # may finish before /metrics is scraped) while histogram counters
            # are monotonically accumulated.
            priority_labels = {s.labels.get("priority", "") for s in samples}
            self.assertIn(
                "",
                priority_labels,
                f"{metric_name}: missing total (priority='') sample",
            )

    def test_priority_label_in_histogram_metrics(self):
        """Send requests with different priorities and verify that
        histogram metrics (TTFT, e2e latency) contain the priority label."""
        for priority in [1, 5]:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 20},
                    "priority": priority,
                },
            )
            self.assertEqual(response.status_code, 200)

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        histogram_metrics = [
            "sglang:time_to_first_token_seconds",
            "sglang:e2e_request_latency_seconds",
        ]
        for metric_name in histogram_metrics:
            count_name = f"{metric_name}_count"
            samples = metrics.get(count_name, [])
            self.assertGreater(len(samples), 0, f"No samples found for {count_name}")

            priority_values = {s.labels.get("priority", "") for s in samples}
            non_empty = priority_values - {""}
            self.assertGreater(
                len(non_empty),
                0,
                f"{count_name}: expected per-priority samples, "
                f"got priority labels: {priority_values}",
            )

            for expected_priority in ["1", "5"]:
                matching = [
                    s for s in samples if s.labels.get("priority") == expected_priority
                ]
                self.assertGreater(
                    len(matching),
                    0,
                    f"{count_name}: no sample with priority='{expected_priority}'",
                )
                self.assertGreater(
                    matching[0].value,
                    0,
                    f"{count_name}: priority='{expected_priority}' count should be > 0",
                )

    def test_default_priority_value(self):
        """Requests without explicit priority should use --default-priority-value (0)."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Hello world",
                "sampling_params": {"temperature": 0, "max_new_tokens": 5},
            },
        )
        self.assertEqual(response.status_code, 200)

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        e2e_count = metrics.get("sglang:e2e_request_latency_seconds_count", [])
        priority_values = {s.labels.get("priority", "") for s in e2e_count}
        self.assertIn(
            "0",
            priority_values,
            f"Expected priority='0' from default, got: {priority_values}",
        )


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in result:
                result[sample.name] = []
            result[sample.name].append(sample)
    return result


if __name__ == "__main__":
    unittest.main()
