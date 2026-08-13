import argparse
from pathlib import Path


ENVIRON_WINDOW_SIZE_ANCHOR = """\
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
"""
ENVIRON_WINDOW_SIZE_FIELD = """\
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE = EnvInt(16)
"""

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
    \"\"\"Track the largest of the latest non-empty prefill attempts.

    The default window keeps 16 attempts. Successful admissions use their
    actual batch size; rejected attempts use a conservative local estimate.
    Decode-only and idle scheduler passes do not age the high-watermark.
    \"\"\"

    def __init__(self, window_size: int = 16):
        if window_size <= 0:
            raise ValueError(f\"window_size must be positive, got {window_size}\")
        self._recent_attempt_sizes = deque(maxlen=window_size)

    @property
    def max_prefill_bs(self) -> int:
        return max(self._recent_attempt_sizes, default=0)

    def observe_attempt(self, attempted_prefill_bs: int) -> int:
        if attempted_prefill_bs <= 0:
            raise ValueError(
                \"attempted_prefill_bs must be positive for a non-empty attempt, \"
                f\"got {attempted_prefill_bs}\"
            )
        max_prefill_bs_before = self.max_prefill_bs
        self._recent_attempt_sizes.append(attempted_prefill_bs)
        max_prefill_bs_after = self.max_prefill_bs
        if _DEBUG_LOG:
            logger.info(
                "PrefillDelayer tracker update "
                "(observed_prefill_bs=%d, max_prefill_bs_before=%d, "
                "max_prefill_bs_after=%d, recent_attempt_sizes=%s)",
                attempted_prefill_bs,
                max_prefill_bs_before,
                max_prefill_bs_after,
                list(self._recent_attempt_sizes),
            )
        return max_prefill_bs_after
"""

PREFILL_DELAYER_EXECUTOR_INIT_OLD = """\
        self._result: Optional[_NegotiateOutput] = None
"""
PREFILL_DELAYER_EXECUTOR_INIT_NEW = """\
        self._result: Optional[_NegotiateOutput] = None
        self._attempted_prefill_bs = 0
"""

PREFILL_DELAYER_FINALIZE_OLD = """\
    def finalize(self, *, actual_prefill: bool):
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)

        _record_single_pass_result(
            actual_execution=actual_prefill,
            output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
            debug_log_enabled=self._prefill_delayer._debug_log_enabled,
        )
"""
PREFILL_DELAYER_FINALIZE_NEW = """\
    def finalize(self, *, actual_prefill_bs: int) -> int:
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)

        _record_single_pass_result(
            actual_execution=actual_prefill_bs > 0,
            output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
            debug_log_enabled=self._prefill_delayer._debug_log_enabled,
        )
        if _DEBUG_LOG and (actual_prefill_bs > 0 or self._attempted_prefill_bs > 0):
            logger.info(
                "PrefillDelayer non-empty pass result "
                "(estimated_attempted_prefill_bs=%d, actual_prefill_bs=%d, "
                "input_estimation=%s, output_allow=%s, output_reason=%s, "
                "delayed_count_after=%d)",
                self._attempted_prefill_bs,
                actual_prefill_bs,
                self._result.input_estimation,
                self._result.output_allow,
                self._result.output_reason,
                (
                    self._result.next_state.delayed_count
                    if self._result.next_state is not None
                    else 0
                ),
            )
        return actual_prefill_bs or self._attempted_prefill_bs

    def _estimate_attempted_prefill_bs(
        self,
        *,
        running_batch: int,
        max_running_requests: int,
        waiting_queue_len: int,
    ) -> int:
        local_max_running_requests = max_running_requests
        if not self._prefill_delayer.enable_dp_attention:
            local_max_running_requests = (
                max_running_requests + self._prefill_delayer.dp_size - 1
            ) // self._prefill_delayer.dp_size

        # The delayer negotiates before PrefillAdder materializes can_run_list,
        # so a rejected pass has no exact batch size. This upper bound is exact
        # when the waiting queue is the limiter (for example, two queued
        # requests after a cached BS=10 spike), and it never exceeds the local
        # request slots available to the candidate batch.
        free_slots = max(local_max_running_requests - running_batch, 1)
        non_empty_queue_len = max(waiting_queue_len, 1)
        return min(non_empty_queue_len, free_slots)
"""

PREFILL_DELAYER_NEGOTIATE_OLD = """\
    def negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> bool:
        if not self._called:
"""
PREFILL_DELAYER_NEGOTIATE_NEW = """\
    def negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> bool:
        if local_prefillable:
            self._attempted_prefill_bs = max(
                self._attempted_prefill_bs,
                self._estimate_attempted_prefill_bs(
                    running_batch=running_batch,
                    max_running_requests=max_running_requests,
                    waiting_queue_len=waiting_queue_len,
                ),
            )
        if not self._called:
"""

PREFILL_DELAYER_ALL_PATH_CONDITIONS_OLD = """\
            queue_condition = False
            if self._queue_trigger_enabled and global_running_batch_max > 0:
"""
PREFILL_DELAYER_ALL_PATH_CONDITIONS_NEW = """\
            queue_condition = False
            queue_min_effective = 0
            if self._queue_trigger_enabled and global_running_batch_max > 0:
"""
PREFILL_DELAYER_SLOT_CONDITION_OLD = """\
            slot_condition = (
                max_running_requests - global_running_batch_max
                < global_max_prefill_bs_max
            )

            if slot_condition or queue_condition:
"""
PREFILL_DELAYER_SLOT_CONDITION_NEW = """\
            slot_condition = (
                max_running_requests - global_running_batch_max
                < global_max_prefill_bs_max
            )

            if _DEBUG_LOG:
                logger.info(
                    "PrefillDelayer all-path conditions "
                    "(running_batch=%d, waiting_queue_len=%d, "
                    "max_running_requests=%d, free_slots=%d, "
                    "max_prefill_bs=%d, queue_min_effective=%d, "
                    "slot_condition=%s, queue_condition=%s, "
                    "delayed_count_before=%d)",
                    global_running_batch_max,
                    global_waiting_queue_max,
                    max_running_requests,
                    max_running_requests - global_running_batch_max,
                    global_max_prefill_bs_max,
                    queue_min_effective,
                    slot_condition,
                    queue_condition,
                    prev_state.delayed_count if prev_state else 0,
                )

            if slot_condition or queue_condition:
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
        self.prefill_bs_tracker = RecentPrefillBatchSizeTracker(
            window_size=envs.SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE.get()
        )
        self.max_prefill_bs: int = 0
"""

SCHEDULER_PASS_DECAY = """\
            # Decay the max-prefill-bs high-watermark once per pass so one
            # unusually large admission burst does not permanently raise the
            # slot_condition bar in the delayer (0.998/pass ~= half-life of
            # ~350 forward passes).
            self.max_prefill_bs *= 0.998
"""
SCHEDULER_FINALIZE_OLD = """\
        if self.prefill_delayer:
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)
"""
SCHEDULER_FINALIZE_NEW = """\
        if self.prefill_delayer:
            observed_prefill_bs = prefill_delayer_single_pass.finalize(
                actual_prefill_bs=ret.batch_size() if ret is not None else 0
            )
            if observed_prefill_bs > 0:
                self.max_prefill_bs = self.prefill_bs_tracker.observe_attempt(
                    observed_prefill_bs
                )
"""
SCHEDULER_ADMISSION_OLD = (
    "        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))"
)
SCHEDULER_ADMISSION_NEW = ""


def _replace_once(source: str, old: str, new: str, path: Path) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old!r}")
    return source.replace(old, new, 1)


def apply_fix(runtime_python_root: Path) -> None:
    environ_path = runtime_python_root / "sglang/srt/environ.py"
    prefill_delayer_path = (
        runtime_python_root / "sglang/srt/managers/prefill_delayer.py"
    )
    scheduler_path = runtime_python_root / "sglang/srt/managers/scheduler.py"

    environ_source = environ_path.read_text()
    prefill_delayer_source = prefill_delayer_path.read_text()
    scheduler_source = scheduler_path.read_text()

    already_applied = (
        ENVIRON_WINDOW_SIZE_FIELD in environ_source
        and "class RecentPrefillBatchSizeTracker:" in prefill_delayer_source
        and "def observe_attempt(" in prefill_delayer_source
        and "def finalize(self, *, actual_prefill_bs: int) -> int:"
        in prefill_delayer_source
        and SCHEDULER_IMPORT_NEW in scheduler_source
        and SCHEDULER_HIGH_WATERMARK_INIT_NEW in scheduler_source
        and SCHEDULER_FINALIZE_NEW in scheduler_source
        and SCHEDULER_ADMISSION_OLD not in scheduler_source
        and SCHEDULER_PASS_DECAY not in scheduler_source
    )
    if already_applied:
        print("Prefill high-watermark fix is already present", flush=True)
        return

    environ_source = _replace_once(
        environ_source,
        ENVIRON_WINDOW_SIZE_ANCHOR,
        ENVIRON_WINDOW_SIZE_FIELD,
        environ_path,
    )
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
    prefill_delayer_source = _replace_once(
        prefill_delayer_source,
        PREFILL_DELAYER_EXECUTOR_INIT_OLD,
        PREFILL_DELAYER_EXECUTOR_INIT_NEW,
        prefill_delayer_path,
    )
    prefill_delayer_source = _replace_once(
        prefill_delayer_source,
        PREFILL_DELAYER_FINALIZE_OLD,
        PREFILL_DELAYER_FINALIZE_NEW,
        prefill_delayer_path,
    )
    prefill_delayer_source = _replace_once(
        prefill_delayer_source,
        PREFILL_DELAYER_NEGOTIATE_OLD,
        PREFILL_DELAYER_NEGOTIATE_NEW,
        prefill_delayer_path,
    )
    prefill_delayer_source = _replace_once(
        prefill_delayer_source,
        PREFILL_DELAYER_ALL_PATH_CONDITIONS_OLD,
        PREFILL_DELAYER_ALL_PATH_CONDITIONS_NEW,
        prefill_delayer_path,
    )
    prefill_delayer_source = _replace_once(
        prefill_delayer_source,
        PREFILL_DELAYER_SLOT_CONDITION_OLD,
        PREFILL_DELAYER_SLOT_CONDITION_NEW,
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
        SCHEDULER_FINALIZE_OLD,
        SCHEDULER_FINALIZE_NEW,
        scheduler_path,
    )
    scheduler_source = _replace_once(
        scheduler_source,
        SCHEDULER_ADMISSION_OLD,
        SCHEDULER_ADMISSION_NEW,
        scheduler_path,
    )

    environ_path.write_text(environ_source)
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
