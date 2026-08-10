import argparse
from pathlib import Path


PREFILL_DELAYER_HELPER_ANCHOR = """\
_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)
"""
PREFILL_DELAYER_IMPORT_OLD = """\
import dataclasses
import logging
import time
"""
PREFILL_DELAYER_IMPORT_NEW = """\
import dataclasses
import logging
import time
from collections import deque
"""
PREFILL_DELAYER_HELPER = """\
_DEBUG_LOG = get_bool_env_var("SGLANG_PREFILL_DELAYER_DEBUG_LOG")

logger = logging.getLogger(__name__)


class RecentPrefillBatchSizeTracker:
    \"\"\"Track the largest of the latest real prefill admissions.

    The default window keeps 64 non-empty admissions. Decode-only and idle
    scheduler passes do not age the high-watermark.
    \"\"\"

    def __init__(self, window_size: int = 64):
        if window_size <= 0:
            raise ValueError(f\"window_size must be positive, got {window_size}\")
        self._recent_admission_sizes = deque(maxlen=window_size)

    @property
    def max_prefill_bs(self) -> int:
        return max(self._recent_admission_sizes, default=0)

    def observe_admission(self, admitted_prefill_bs: int) -> int:
        if admitted_prefill_bs <= 0:
            raise ValueError(
                \"admitted_prefill_bs must be positive for a real admission, \"
                f\"got {admitted_prefill_bs}\"
            )
        self._recent_admission_sizes.append(admitted_prefill_bs)
        return self.max_prefill_bs
"""

SCHEDULER_IMPORT_OLD = """\
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
"""
SCHEDULER_IMPORT_NEW = """\
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
)
"""

SCHEDULER_HIGH_WATERMARK_INIT_OLD = """\
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.max_prefill_bs: float = 0.0
"""
SCHEDULER_HIGH_WATERMARK_INIT_NEW = """\
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.prefill_bs_tracker = RecentPrefillBatchSizeTracker()
        self.max_prefill_bs: int = 0
"""

SCHEDULER_PASS_DECAY = """\
            # Decay the max-prefill-bs high-watermark once per pass so one
            # unusually large admission burst does not permanently raise the
            # slot_condition bar in the delayer (0.998/pass ~= half-life of
            # ~350 forward passes).
            self.max_prefill_bs *= 0.998
"""
SCHEDULER_ADMISSION_OLD = (
    "        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))"
)
SCHEDULER_ADMISSION_NEW = """\
        self.max_prefill_bs = self.prefill_bs_tracker.observe_admission(
            len(can_run_list)
        )
""".rstrip()


def _replace_once(source: str, old: str, new: str, path: Path) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old!r}")
    return source.replace(old, new, 1)


def apply_fix(runtime_python_root: Path) -> None:
    prefill_delayer_path = (
        runtime_python_root / "sglang/srt/managers/prefill_delayer.py"
    )
    scheduler_path = runtime_python_root / "sglang/srt/managers/scheduler.py"

    prefill_delayer_source = prefill_delayer_path.read_text()
    scheduler_source = scheduler_path.read_text()

    already_applied = (
        "class RecentPrefillBatchSizeTracker:" in prefill_delayer_source
        and SCHEDULER_IMPORT_NEW in scheduler_source
        and SCHEDULER_HIGH_WATERMARK_INIT_NEW in scheduler_source
        and SCHEDULER_ADMISSION_NEW in scheduler_source
        and SCHEDULER_PASS_DECAY not in scheduler_source
    )
    if already_applied:
        print("Prefill high-watermark fix is already present", flush=True)
        return

    prefill_delayer_source = _replace_once(
        prefill_delayer_source,
        PREFILL_DELAYER_IMPORT_OLD,
        PREFILL_DELAYER_IMPORT_NEW,
        prefill_delayer_path,
    )
    prefill_delayer_source = _replace_once(
        prefill_delayer_source,
        PREFILL_DELAYER_HELPER_ANCHOR,
        PREFILL_DELAYER_HELPER,
        prefill_delayer_path,
    )
    scheduler_source = _replace_once(
        scheduler_source,
        SCHEDULER_IMPORT_OLD,
        SCHEDULER_IMPORT_NEW,
        scheduler_path,
    )
    scheduler_source = _replace_once(
        scheduler_source,
        SCHEDULER_PASS_DECAY,
        "",
        scheduler_path,
    )
    scheduler_source = _replace_once(
        scheduler_source,
        SCHEDULER_HIGH_WATERMARK_INIT_OLD,
        SCHEDULER_HIGH_WATERMARK_INIT_NEW,
        scheduler_path,
    )
    scheduler_source = _replace_once(
        scheduler_source,
        SCHEDULER_ADMISSION_OLD,
        SCHEDULER_ADMISSION_NEW,
        scheduler_path,
    )

    prefill_delayer_path.write_text(prefill_delayer_source)
    scheduler_path.write_text(scheduler_source)
    print(f"Applied prefill high-watermark fix to {runtime_python_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_python_root", type=Path)
    args = parser.parse_args()
    apply_fix(args.runtime_python_root)


if __name__ == "__main__":
    main()
