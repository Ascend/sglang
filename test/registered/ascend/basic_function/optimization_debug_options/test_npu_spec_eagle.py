"""EAGLE3 spec-decoding core: overlap (spec v2) x no-overlap (spec v1) matrix,
same standard config (topk=1, page_size=1), only ``disable_overlap`` differs.
flashinfer is pinned (the 5090 default) so a default-selection change can't
silently alter what this exercises.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import (
    EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class _Core(Eagle3Base):
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)
    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128


class TestEagle3Overlap(_Core, *_KITS):
    """Spec v2 (overlap scheduler on)."""

    disable_overlap = False


class TestEagle3NoOverlap(_Core, *_KITS):
    """Spec v1 (overlap scheduler off)."""

    disable_overlap = True


if __name__ == "__main__":
    unittest.main()
