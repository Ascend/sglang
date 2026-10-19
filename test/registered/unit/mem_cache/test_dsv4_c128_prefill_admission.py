"""Fix-verification test for the DSV4 C128 prefill-admission bug.

The asserted invariant IS the bug: a prefill that the allocator's own C128 gate
will refuse must never be admitted by the prefill budget. Pre-fix, the budget is
C128-blind, so it admits the batch and ``alloc_extend`` then returns None ->
RuntimeError("Prefill out of memory") -> the DP group SIGQUITs.

The same file runs on both trees, so it is a real regression gate:

* pre-fix (0623be4ccb) -> assertion failure naming the admitted-but-refused batch
* fixed                -> green

No ``DSV4PrefillBudget`` import: the budget comes from the allocator factory
(``create_prefill_budget``), falling back to the stock SWA factory when the DSV4
override does not exist yet.
"""

import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
    DSV4NPUTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FULL_PAGE_SIZE = 128
C128_PAGE_SIZE = 16          # compressed tokens per C128 page
FULL_TOKENS_PER_C128_PAGE = 128 * C128_PAGE_SIZE   # 2048


def _bind(stub, name):
    """Bind a real allocator method onto a plain stub (house CPU-test idiom)."""
    return MethodType(getattr(DSV4NPUTokenToKVPoolAllocator, name), stub)


def _allocator(*, full_free=7296, c128_free_pages=0, swa_free=5120):
    stub = SimpleNamespace(
        page_size=FULL_PAGE_SIZE,
        full_attn_allocator=SimpleNamespace(available_size=lambda: full_free),
        c128_attn_allocator=SimpleNamespace(
            page_size=C128_PAGE_SIZE,
            num_pages=32,
            available_size=lambda: c128_free_pages * C128_PAGE_SIZE,
        ),
        swa_available_size=lambda: swa_free,
    )
    stub.c128_num_pages_needed = _bind(stub, "c128_num_pages_needed")
    stub._has_c128_sidecar_capacity = _bind(stub, "_has_c128_sidecar_capacity")
    # Factory: the DSV4 override when it exists, else the inherited SWA one.
    factory = getattr(
        DSV4NPUTokenToKVPoolAllocator,
        "create_prefill_budget",
        SWATokenToKVPoolAllocator.create_prefill_budget,
    )
    stub.create_prefill_budget = MethodType(factory, stub)
    return stub


def _tree_cache(*, full_evictable=0):
    return SimpleNamespace(
        full_evictable_size=lambda: full_evictable,
        swa_evictable_size=lambda: 0,
        sliding_window_size=None,
    )


def _admit(allocator, tree_cache, tokens):
    """Ask the budget the way PrefillAdder does (schedule_policy.py:804)."""
    budget = allocator.create_prefill_budget(tree_cache)
    admitted, _ = budget.check_prefill(
        extend_input_len=tokens,
        total_tokens=tokens,
        max_new_tokens=16,
        input_tokens=tokens,
        swa_host_hit_length=0,
        chunk_limit=None,
    )
    return budget, admitted


def _allocator_accepts(allocator, chains):
    """The allocator's own gate for a fresh batch (prefix_lens all zero)."""
    need = allocator.c128_num_pages_needed(
        torch.zeros(len(chains), dtype=torch.int64),
        torch.tensor(chains, dtype=torch.int64),
    )
    return allocator._has_c128_sidecar_capacity(
        torch.zeros(len(chains), dtype=torch.int64),
        torch.tensor(chains, dtype=torch.int64),
    ), need


class TestC128AdmissionInvariant(CustomTestCase):
    def _assert_never_admits_a_refused_batch(self, *, chains, tokens, **kw):
        allocator = _allocator(**kw)
        accepts, need = _allocator_accepts(allocator, chains)
        _, admitted = _admit(allocator, _tree_cache(
            full_evictable=kw.get("full_evictable", 0)), tokens)
        if admitted and not accepts:
            self.fail(
                "admission accepted a batch the allocator refuses: "
                f"need {need} C128 page(s), free "
                f"{kw.get('c128_free_pages', 0)} -> alloc_extend would return "
                "None and the scheduler would raise 'Prefill out of memory'"
            )

    def test_log_scenario_full_has_room_but_c128_is_dry(self):
        # logs/1.log.txt DP12: FULL free 7296, SWA free 5120, request 1224 tokens,
        # C128 free pages 0, nothing evictable.
        self._assert_never_admits_a_refused_batch(
            chains=[1224], tokens=1224, full_free=7296, c128_free_pages=0,
        )

    def test_sub_page_evictable_leaves_are_not_counted_as_c128_headroom(self):
        # 100 sub-page FULL leaves (L=100 < one C128 page) are evictable: an
        # implementation that credits full_evictable_size() as C128 headroom
        # admits this batch, but evicting them frees no C128 page.
        self._assert_never_admits_a_refused_batch(
            chains=[1224], tokens=1224, full_free=7296, c128_free_pages=0,
            full_evictable=10_000,
        )

    def test_per_chain_tail_pages_are_not_over_committed(self):
        # 2 free pages; each 200-token chain opens its own page, so at most two
        # chains may be admitted in one pass.
        allocator = _allocator(full_free=1 << 20, c128_free_pages=2)
        budget = allocator.create_prefill_budget(_tree_cache())
        admitted_chains = 0
        for _ in range(3):
            admitted, _ = budget.check_prefill(
                extend_input_len=200, total_tokens=200, max_new_tokens=16,
                input_tokens=200, swa_host_hit_length=0, chunk_limit=None,
            )
            if not admitted:
                break
            admitted_chains += 1
            budget.reserve(200, 16)
        self.assertLessEqual(
            admitted_chains, 2,
            "more chains admitted than free C128 pages; the third chain would "
            "fail in alloc_extend",
        )

    def test_still_admits_when_c128_has_room(self):
        # Guard against "fix by always refusing".
        _, admitted = _admit(
            _allocator(full_free=4096, c128_free_pages=8), _tree_cache(), 1224
        )
        self.assertTrue(admitted)


if __name__ == "__main__":
    unittest.main()
