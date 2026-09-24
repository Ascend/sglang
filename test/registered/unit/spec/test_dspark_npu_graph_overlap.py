"""CPU contracts for token-keyed NPU graphs and overlapped confidence relay."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    schedule_verify_lens_topk,
)
from sglang.srt.hardware_backend.npu.attention.dspark_compact import (
    NpuCompactGraphMetadata,
    schedule_verify_lens_npu,
)
from sglang.srt.managers.overlap_utils import (
    CONFIDENCE_RELAY_RING_DEPTH,
    CONFIDENCE_RELAY_RING_LAG,
    ConfidenceRelay,
)
from sglang.srt.speculative.dspark_components import dspark_planner
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def method(path, cls, name, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = copy.deepcopy(
        next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    fn.decorator_list = []
    for n in ast.walk(fn):
        if isinstance(n, ast.arg):
            n.annotation = None
        if isinstance(n, ast.FunctionDef):
            n.returns = None
    scope = dict(torch=torch, **namespace)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize("bs,gamma", [(1, 1), (3, 7), (9, 16), (32, 8)])
def test_device_ranker_matches_reference(bs, gamma):
    generator = torch.Generator().manual_seed(19)
    cfg = DSparkScheduleConfig(gamma=gamma)
    for confidence in (
        torch.rand(bs, gamma, generator=generator),
        torch.ones(bs, gamma),
        torch.zeros(bs, gamma),
    ):
        for budget in (0, 1, bs, bs * gamma // 2, bs * gamma, bs * gamma + 1):
            torch.testing.assert_close(
                schedule_verify_lens_npu(confidence, budget=budget, cfg=cfg),
                schedule_verify_lens_topk(
                    confidence=confidence, budget=budget, cfg=cfg
                ),
            )


def test_graph_buffers_reused_without_changing_live_causal_positions():
    state = NpuCompactGraphMetadata(
        num_slots=3, num_tokens=16, table_width=4, device="cpu"
    )
    ptrs = [t.data_ptr() for t in (state.query_ends, state.kv_lens, state.block_tables)]
    table = torch.arange(16, 16 + 4 * 16).view(4, 16)
    for lens, prefixes, reqs in (
        ([2, 4, 1], [3, 7, 8], [2, 0, 3]),
        ([1, 2], [5, 9], [3, 2]),
        ([4, 4, 4], [4, 8, 12], [1, 3, 0]),
    ):
        state.update(
            prefix_lens=torch.tensor(prefixes),
            req_pool_indices=torch.tensor(reqs),
            req_to_token=table,
            verify_lens=torch.tensor(lens),
            page_size=4,
        )
        assert ptrs == [
            t.data_ptr() for t in (state.query_ends, state.kv_lens, state.block_tables)
        ]
        assert state.query_ends[-1] == 16
        assert (
            state.query_ends[: len(lens)].tolist()
            == torch.tensor(lens).cumsum(0).tolist()
        )
        assert state.kv_lens[: len(lens)].tolist() == [
            p + v for p, v in zip(prefixes, lens)
        ]
        assert not state.block_tables[-1].any()
        assert not state.kv_lens[len(lens) : -1].any()
        # In right-aligned causal attention, q row j sees prefix+j+1 keys.
        starts = [0] + state.query_ends[: len(lens) - 1].tolist()
        for i, (p, v) in enumerate(zip(prefixes, lens)):
            q_len = int(state.query_ends[i]) - starts[i]
            assert int(state.kv_lens[i]) - q_len == p
            assert q_len == v


def test_metadata_capture_tiers_with_same_slot_count_do_not_alias():
    init = method(
        "hardware_backend/npu/attention/ascend_backend.py",
        "AscendAttnBackend",
        "_init_compact_graph_metadata",
        NpuCompactGraphMetadata=NpuCompactGraphMetadata,
        ForwardMetadata=SimpleNamespace,
    )
    backend = SimpleNamespace(
        graph_metadata={},
        req_to_token=torch.arange(256).view(4, 64),
        page_size=8,
        device="cpu",
    )
    pointers = []
    for total in (8, 16, 8):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([1, 3]), graph_num_tokens=total
        )
        fb = SimpleNamespace(
            batch_size=2,
            seq_lens=torch.tensor([7, 15]),
            req_pool_indices=torch.tensor([2, 1]),
        )
        init(backend, fb, layout)
        pointers.append(backend.forward_metadata.actual_seq_lengths_q.data_ptr())
        assert backend.forward_metadata.actual_seq_lengths_q.tolist() == [1, 4, total]
    assert pointers[0] == pointers[2] != pointers[1]


def test_npu_runner_uses_token_key_and_stages_preplanned_layout():
    class Output:
        pass

    class Proxy:
        def __init__(self, tensors):
            self.tensors = tensors

    execute = method(
        "hardware_backend/npu/graph_runner/npu_graph_runner.py",
        "NPUGraphRunner",
        "execute",
        LogitsProcessorOutput=Output,
        PPProxyTensors=Proxy,
    )
    runner = SimpleNamespace(
        ragged_verify_mode=True,
        bs=2,
        raw_num_token=8,
        _replay_graph_key=(8, "variant"),
        backend=SimpleNamespace(replay=Mock(return_value=Proxy({}))),
    )
    runner.load_batch = Mock()
    fb = SimpleNamespace(needs_forward_metadata_init=lambda: False)
    assert execute(runner, fb).tensors == {}
    runner.load_batch.assert_called_once_with(fb, None)
    runner.backend.replay.assert_called_once_with((8, "variant"), fb)


def test_tp_broadcast_budget_precedes_tier_choice():
    planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner._budget_planner = Mock()
    planner._budget_from_resolved = Mock(return_value=1)
    planner._maybe_gather_dp_verify_tier = Mock()
    planner.verify_num_draft_tokens = 8
    planner._schedule_cfg = DSparkScheduleConfig(gamma=7)
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(),
        batch_size=lambda: 2,
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_extend_in_batch=False,
        req_pool_indices_cpu=torch.tensor([2, 5]),
    )
    group = SimpleNamespace(broadcast_object=Mock(return_value=6))
    with (
        patch.object(dspark_planner, "_is_npu", True),
        patch.object(dspark_planner, "get_tp_group", return_value=group),
    ):
        planner.prepare_verify_budget(batch, Mock())
    assert batch.spec_info.verify_token_budget == 6
    assert batch.spec_verify_tier_num_tokens == 8


def test_gpu_forced_budget_behavior_unchanged():
    planner = dspark_planner.DSparkVerifyPlanner.__new__(
        dspark_planner.DSparkVerifyPlanner
    )
    planner._budget_planner = Mock()
    planner._is_verify_all = True
    with patch.object(dspark_planner, "_is_npu", False):
        planner.set_forced_budget_frac(0.25)
    assert planner._is_verify_all is True
    assert planner._budget_planner.forced_budget_frac == 0.25


def test_confidence_relay_not_ready_generation_and_source_fence():
    events = [
        Mock(query=Mock(return_value=True)) for _ in range(CONFIDENCE_RELAY_RING_DEPTH)
    ]
    relay = ConfidenceRelay(
        device=torch.device("cpu"),
        req_pool_size=3,
        pool=SimpleNamespace(req_generation=torch.tensor([2, 3, 4])),
    )
    relay.initialized = True
    relay.confidence_buf = torch.zeros(3, 2)
    relay.conf_ring = torch.zeros(CONFIDENCE_RELAY_RING_DEPTH, 3, 2)
    relay.gen_ring = torch.zeros(CONFIDENCE_RELAY_RING_DEPTH, 3, dtype=torch.int64)
    relay.copy_done = events
    from contextlib import nullcontext

    stream, ready = Mock(), Mock()
    module = SimpleNamespace(
        stream=lambda s: nullcontext(), current_stream=Mock(return_value=stream)
    )
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(future_indices=torch.tensor([1])),
        req_pool_indices_cpu=torch.tensor([1]),
    )
    with patch("torch.get_device_module", return_value=module):
        for i in range(CONFIDENCE_RELAY_RING_LAG):
            relay.scatter(torch.tensor([1]), torch.tensor([[0.1 + i * 0.1, 0.5]]))
            relay.issue_ring_copy(stream=stream, publish_ready=ready)
        events[0].query.return_value = False
        assert relay.resolve(batch, stream=stream, publish_ready=ready) is None
        events[0].query.return_value = True
        resolved = relay.resolve(batch, stream=stream, publish_ready=ready)
        assert resolved.generation.tolist() == [3]
        torch.testing.assert_close(resolved.confidence, torch.tensor([[0.1, 0.5]]))
        with patch("sglang.srt.managers.overlap_utils._is_npu", True):
            relay.scatter(torch.tensor([1]), torch.tensor([[0.8, 0.9]]))
        stream.wait_event.assert_called_with(
            events[(relay.ring_pos - 1) % CONFIDENCE_RELAY_RING_DEPTH]
        )


def test_device_metadata_update_has_no_host_reads():
    state = NpuCompactGraphMetadata(
        num_slots=3, num_tokens=16, table_width=8, device="cpu"
    )
    args = dict(
        prefix_lens=torch.tensor([7, 17]),
        req_pool_indices=torch.tensor([1, 2]),
        req_to_token=torch.arange(256).view(4, 64),
        verify_lens=torch.tensor([1, 4]),
        page_size=8,
    )
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar sync")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")),
    ):
        state.update(**args)
        schedule_verify_lens_npu(
            torch.ones(3, 7), budget=5, cfg=DSparkScheduleConfig(gamma=7)
        )
    assert state.query_ends.tolist() == [1, 5, 5, 16]


def test_npu_executor_keeps_host_prefix_and_device_only_layout():
    from sglang.srt.speculative.dspark_components import dspark_verify

    executor = dspark_verify.TargetVerifyExecutor.__new__(
        dspark_verify.TargetVerifyExecutor
    )
    executor.verify_num_draft_tokens = 8
    executor.model_runner = SimpleNamespace(
        attn_backend=SimpleNamespace(supports_ragged_verify_graph=True)
    )
    executor._forward_prepared_verify = Mock(return_value="launched")
    prefix = torch.tensor([7, 15])
    batch = SimpleNamespace(seq_lens_cpu=prefix, seq_lens_sum=22)
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([2, 4]), graph_num_tokens=8
    )
    window = SimpleNamespace(
        verify_ids=torch.arange(8),
        positions=torch.arange(8),
        verify_cache_loc=torch.arange(8),
    )
    with (
        patch.object(dspark_verify, "_is_npu", True),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
    ):
        result = executor._run_ragged(
            batch=batch, layout=layout, ragged_window=window, sampling_info=None
        )
    assert result == "launched"
    assert batch.seq_lens_cpu is prefix
    assert batch.seq_lens_sum == 22


def test_non_npu_packing_retains_existing_cache_location_helper():
    from sglang.kernels.ops.speculative.dspark import dspark_verify_window as mod

    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=[2, 1], device="cpu", grid=[4]
    )
    table = torch.arange(32).view(2, 16)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([3, 5]), req_pool_indices=torch.tensor([1, 0])
    )
    with (
        patch.object(mod, "_is_npu", False),
        patch.object(
            mod, "assign_extend_cache_locs_func", return_value=torch.tensor([19, 20, 5])
        ) as helper,
    ):
        result = mod.build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=torch.tensor([[8, 9], [6, 7]]),
            draft_tokens=torch.tensor([[9], [7]]),
            bs=2,
            device="cpu",
            verify_num_draft_tokens=2,
            model_runner=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(req_to_token=table)
            ),
        )
    helper.assert_called_once()
    assert result.verify_cache_loc.tolist() == [19, 20, 5, 0]


def test_future_map_accepts_scheduler_string_device():
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    pool = SimpleNamespace(req_to_token=torch.zeros(4, 32, dtype=torch.int64))
    future = FutureMap(
        "cpu", SpeculativeAlgorithm.DSPARK, pool, needs_confidence_relay=True
    )
    assert future.npu_confidence_stream is None
