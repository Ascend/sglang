import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.prefill_delayer import (
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
    _NegotiateOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillDelayerHighWatermark(CustomTestCase):
    def test_default_window_expires_peak_after_16_attempts(self):
        tracker = RecentPrefillBatchSizeTracker()

        self.assertEqual(tracker.observe_attempt(100), 100)
        for _ in range(15):
            self.assertEqual(tracker.observe_attempt(2), 100)

        self.assertEqual(tracker.observe_attempt(2), 2)

    def test_peak_expires_after_recent_attempt_window(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        self.assertEqual(tracker.observe_attempt(100), 100)
        for attempted_prefill_bs in [2, 1, 2]:
            self.assertEqual(tracker.observe_attempt(attempted_prefill_bs), 100)

        self.assertEqual(tracker.observe_attempt(2), 2)

    def test_recurring_small_peak_remains_effective(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        for attempted_prefill_bs in [2, 1, 1, 2, 1, 1, 2]:
            self.assertEqual(tracker.observe_attempt(attempted_prefill_bs), 2)

    def test_rejected_non_empty_attempt_advances_window(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)
        self.assertEqual(tracker.observe_attempt(10), 10)

        delayer = MagicMock()
        delayer.enable_dp_attention = True
        delayer.dp_size = 1
        delayer._metrics_collector = None
        delayer._debug_log_enabled = False
        delayer._negotiate_should_allow_prefill.return_value = _NegotiateOutput(
            next_state=None,
            input_estimation="all",
            output_allow=False,
            output_reason="delay",
            num_prefillable=1,
            num_token_watermark_force_allow=0,
        )

        for _ in range(4):
            executor = PrefillDelayerSinglePassExecutor(delayer, token_usage=0.9)
            self.assertFalse(
                executor.negotiate_should_allow_prefill(
                    local_prefillable=True,
                    running_batch=15,
                    max_prefill_bs=tracker.max_prefill_bs,
                    max_running_requests=20,
                    waiting_queue_len=2,
                )
            )
            attempted_prefill_bs = executor.finalize(actual_prefill_bs=0)
            self.assertEqual(attempted_prefill_bs, 2)
            tracker.observe_attempt(attempted_prefill_bs)

        self.assertEqual(tracker.max_prefill_bs, 2)

    def test_rejects_empty_attempts(self):
        with self.assertRaisesRegex(ValueError, "window_size must be positive"):
            RecentPrefillBatchSizeTracker(window_size=0)

        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        with self.assertRaisesRegex(ValueError, "non-empty attempt"):
            tracker.observe_attempt(0)

        self.assertEqual(tracker.max_prefill_bs, 0)


if __name__ == "__main__":
    unittest.main()
