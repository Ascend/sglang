import unittest

from sglang.srt.managers.prefill_delayer import (
    update_max_prefill_bs_high_watermark,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillDelayerHighWatermark(CustomTestCase):
    def test_high_watermark_does_not_decay_and_tracks_larger_admission(self):
        high_watermark = 64.0

        for _ in range(2000):
            high_watermark = update_max_prefill_bs_high_watermark(
                high_watermark, admitted_prefill_bs=0
            )

        self.assertEqual(high_watermark, 64.0)

        for _ in range(2000):
            high_watermark = update_max_prefill_bs_high_watermark(
                high_watermark, admitted_prefill_bs=16
            )

        self.assertEqual(high_watermark, 64.0)
        self.assertEqual(
            update_max_prefill_bs_high_watermark(
                4.0, admitted_prefill_bs=8
            ),
            8.0,
        )


if __name__ == "__main__":
    unittest.main()
