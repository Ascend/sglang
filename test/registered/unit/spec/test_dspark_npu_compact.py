"""CPU contracts for the NPU compact port; NPU operator execution is separate."""

import ast
import copy
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_greedy,
    finalize_accept_lens,
)
from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
    schedule_verify_lens_topk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    build_ragged_verify_window,
    scatter_compact_to_strided,
)
from sglang.srt.hardware_backend.npu.attention import dspark_compact
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    build_uninitialized_sps_table,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def layout_for(lens):
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=lens, device=torch.device("cpu"), grid=[sum(lens)]
    )


def _backend_method(name="init_forward_metadata", **extra):
    # Execute the production method with CPU tensors while avoiding imports of
    # torch_npu / sgl_kernel_npu on CPU CI. Only the device boundary is replaced;
    # no copy of the metadata implementation is maintained in this test.
    path = Path(dspark_compact.__file__).with_name("ascend_backend.py")
    tree = ast.parse(path.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttnBackend"
    )
    method = copy.deepcopy(
        next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    method.decorator_list = []
    for arg in method.args.args:
        arg.annotation = None
    for arg in method.args.kwonlyargs:
        arg.annotation = None
    method.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(
        torch=torch,
        np=np,
        ForwardMetadata=lambda: SimpleNamespace(),
        build_compact_verify_metadata=dspark_compact.build_compact_verify_metadata,
        get_parallel=lambda: SimpleNamespace(dcp_enabled=False),
    )
    namespace.update(extra)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class TestNpuCompactMetadata(CustomTestCase):
    def test_mixed_windows_and_page_boundary(self):
        prefix = torch.tensor([127, 255, 513])
        layout = layout_for([1, 3, 8])
        result = dspark_compact.build_compact_verify_metadata(
            prefix_lens=prefix, layout=layout, max_verify_len=8, num_tokens=12
        )
        self.assertEqual(result.query_ends.tolist(), [1, 4, 12])
        self.assertEqual(result.query_ends_cpu, [1, 4, 12])
        self.assertEqual(result.kv_lens.tolist(), [128, 258, 521])
        self.assertEqual(result.kv_lens_cpu.tolist(), [128, 258, 521])
        self.assertEqual(result.max_kv_len, 521)
        self.assertEqual(prefix.tolist(), [127, 255, 513])

    def test_device_only_layout(self):
        layout = layout_for([2, 1])
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=layout.verify_lens, graph_num_tokens=3
        )
        result = dspark_compact.build_compact_verify_metadata(
            prefix_lens=torch.tensor([8, 16]),
            layout=layout,
            max_verify_len=4,
            num_tokens=3,
        )
        self.assertEqual(result.query_ends_cpu, [2, 3])

    def test_reject_invalid_layout(self):
        for lens, prefixes, token_count, max_len in (
            ([2], [10, 20], 2, 4),
            ([5], [10], 5, 4),
            ([1, 3], [10, 20], 8, 4),
        ):
            with self.subTest(lens=lens, token_count=token_count):
                with self.assertRaises(ValueError):
                    dspark_compact.build_compact_verify_metadata(
                        prefix_lens=torch.tensor(prefixes),
                        layout=layout_for(lens),
                        max_verify_len=max_len,
                        num_tokens=token_count,
                    )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 3], device=torch.device("cpu"), grid=[8]
        )
        with self.assertRaisesRegex(ValueError, "padded"):
            dspark_compact.build_compact_verify_metadata(
                prefix_lens=torch.tensor([10, 20]),
                layout=layout,
                max_verify_len=4,
                num_tokens=8,
            )

    def test_real_backend_metadata_does_not_double_extend_cpu_lengths(self):
        method = _backend_method()
        table = torch.arange(4 * 1024).view(4, 1024)
        backend = SimpleNamespace(
            graph_mode=False,
            supports_ragged_verify_graph=False,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            page_size=128,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        for lens in ([1, 3, 8], [8, 8, 8], [1, 1, 1]):
            with self.subTest(lens=lens):
                prefix = torch.tensor([127, 255, 513])
                expanded_cpu = prefix + torch.tensor(lens)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    spec_info=SimpleNamespace(
                        ragged_verify_layout=layout_for(lens), draft_token_num=8
                    ),
                    spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
                    seq_lens=prefix,
                    seq_lens_cpu=expanded_cpu,
                    input_ids=torch.zeros(sum(lens), dtype=torch.int64),
                    global_num_token_non_padded_cpu=sum(lens),
                    req_pool_indices=torch.tensor([2, 0, 3]),
                    extend_seq_lens=None,
                )
                method(backend, batch)
                metadata = backend.forward_metadata
                self.assertEqual(metadata.seq_lens.tolist(), expanded_cpu.tolist())
                self.assertEqual(
                    metadata.seq_lens_cpu_int.tolist(), expanded_cpu.tolist()
                )
                self.assertEqual(
                    metadata.actual_seq_lengths_q.tolist(),
                    torch.tensor(lens).cumsum(0).tolist(),
                )
                self.assertEqual(metadata.block_tables.shape, (3, 5))
                self.assertEqual(batch.seq_lens.tolist(), [127, 255, 513])
                self.assertEqual(batch.seq_lens_cpu.tolist(), expanded_cpu.tolist())

    def test_noncompact_metadata_preserves_target_branch_host_lengths(self):
        # Without a ragged layout, preserve this branch's existing DSpark vs
        # other-speculative host-length contract and page-boundary behavior.
        method = _backend_method()
        backend = SimpleNamespace(
            supports_ragged_verify_graph=False,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(4 * 1024).view(4, 1024)
            ),
            page_size=128,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        for is_dspark in (True, False):
            with self.subTest(is_dspark=is_dspark):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    spec_info=SimpleNamespace(draft_token_num=8),
                    spec_algorithm=SimpleNamespace(is_dspark=lambda: is_dspark),
                    seq_lens=torch.tensor([120, 248]),
                    seq_lens_cpu=torch.tensor([128, 256]),
                    req_pool_indices=torch.tensor([2, 0]),
                    extend_seq_lens=None,
                )
                method(backend, batch)
                metadata = backend.forward_metadata
                expected = [128, 256] if is_dspark else [136, 264]
                self.assertEqual(metadata.seq_lens_cpu_int.tolist(), expected)
                self.assertEqual(metadata.block_tables.shape, (2, 3))
                self.assertEqual(metadata.actual_seq_lengths_q.tolist(), [8, 16])
                self.assertEqual(batch.seq_lens_cpu.tolist(), [128, 256])


class TestNpuCompactExecution(CustomTestCase):
    def test_two_round_executor_handoff(self):
        # Run the production compact executor and accept/commit routing. Replace
        # ForwardBatch construction and the target model at their boundaries;
        # the Ascend metadata method, packing and acceptance execute on CPU.
        metadata_method = _backend_method()
        table = torch.arange(3 * 128, dtype=torch.int32).view(3, 128)
        backend = SimpleNamespace(
            graph_mode=False,
            supports_ragged_verify_graph=False,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            page_size=16,
            is_hybrid_swa=False,
            use_mla=True,
            use_sliding_window_kv_pool=False,
            device="cpu",
        )
        runner = SimpleNamespace(
            req_to_token_pool=backend.req_to_token_pool, attn_backend=backend
        )
        seen_lengths = []

        def prepare(verify_input, batch, target_worker):
            return SimpleNamespace(
                input_ids=verify_input.draft_token,
                positions=verify_input.positions,
                spec_info=verify_input,
                forward_mode=ForwardMode.TARGET_VERIFY,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu,
                req_pool_indices=batch.req_pool_indices,
                extend_seq_lens=None,
                global_num_token_non_padded_cpu=verify_input.draft_token.numel(),
                spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            ), False

        def target_forward(*, batch, forward_batch, is_verify, skip_attn_backend_init):
            self.assertTrue(is_verify)
            self.assertIsNone(skip_attn_backend_init)
            metadata_method(backend, forward_batch)
            lens = forward_batch.spec_info.ragged_verify_layout.verify_lens
            self.assertEqual(
                backend.forward_metadata.seq_lens_cpu_int.tolist(),
                (forward_batch.seq_lens + lens).tolist(),
            )
            seen_lengths.append(backend.forward_metadata.actual_seq_lengths_q.tolist())
            ids = forward_batch.input_ids
            logits = torch.full((ids.numel(), 32), -10.0)
            logits.scatter_(1, (ids + 1).unsqueeze(1), 10.0)
            return SimpleNamespace(
                can_run_cuda_graph=False,
                logits_output=SimpleNamespace(
                    next_token_logits=logits,
                    hidden_states=torch.stack(
                        [ids, forward_batch.positions], dim=1
                    ).float(),
                ),
            )

        injector = Mock()
        executor = dspark_verify.TargetVerifyExecutor(
            target_worker=SimpleNamespace(forward_batch_generation=target_forward),
            gamma=3,
            verify_num_draft_tokens=4,
            model_runner=runner,
            kv_injector=injector,
            tp_sync=Mock(),
        )
        prefix, anchors = torch.tensor([15, 31]), torch.tensor([1, 5])
        for lengths in ([1, 3], [4, 1]):
            layout = layout_for(lengths)
            candidates = anchors[:, None] + torch.arange(4)[None, :]
            batch = SimpleNamespace(
                seq_lens=prefix,
                seq_lens_cpu=prefix.clone(),
                seq_lens_sum=int(prefix.sum()),
                req_pool_indices=torch.tensor([2, 0]),
            )
            original_cpu = batch.seq_lens_cpu
            with (
                patch.object(dspark_verify, "_is_npu", True),
                patch.object(
                    dspark_verify.DFlashVerifyInput, "prepare_for_verify", prepare
                ),
                patch(
                    "sglang.kernels.ops.speculative.dspark.dspark_verify_window._is_npu",
                    True,
                ),
            ):
                result, hidden = executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=candidates,
                    draft_tokens=candidates[:, 1:],
                    bs=2,
                    device="cpu",
                    sampling_info=None,
                )
            self.assertIs(batch.seq_lens_cpu, original_cpu)
            self.assertEqual(batch.seq_lens_sum, int(prefix.sum()))
            self.assertEqual(batch.out_cache_loc.numel(), sum(lengths))
            accepted = executor.accept_and_finalize(
                folded_accept=False,
                bs=2,
                verify_ids_2d=candidates,
                target_logits=result.logits_output.next_token_logits,
                draft_block=SimpleNamespace(
                    greedy_mask=torch.ones(2, dtype=torch.bool)
                ),
                sampling_info=None,
                draft_input=None,
                layout=layout,
                prefix_lens=prefix,
                draft_tokens=candidates[:, 1:],
            )
            self.assertEqual(accepted.commit_lens.tolist(), lengths)
            self.assertEqual(
                accepted.new_seq_lens.tolist(),
                (prefix + torch.tensor(lengths)).tolist(),
            )
            executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden,
                verify_window=None,
                logits_output=result.logits_output,
                commit_lens=accepted.commit_lens,
                bs=2,
                run_compact=True,
            )
            committed = injector.inject_ragged.call_args.kwargs
            self.assertIs(committed["layout"], layout)
            self.assertIs(committed["commit_lens"], accepted.commit_lens)
            for row, length in enumerate(lengths):
                self.assertEqual(
                    hidden.view(2, 4, 2)[row, :length, 0].tolist(),
                    candidates[row, :length].float().tolist(),
                )
                self.assertEqual(
                    accepted.out_tokens[row, :length].tolist(),
                    list(range(int(anchors[row]) + 1, int(anchors[row]) + length + 1)),
                )
            prefix, anchors = accepted.new_seq_lens, accepted.bonus
        self.assertEqual(seen_lengths, [[1, 4], [4, 5]])
        self.assertEqual(prefix.tolist(), [20, 35])
        self.assertEqual(anchors.tolist(), [6, 9])
        self.assertEqual(injector.inject_ragged.call_count, 2)

    @unittest.skipUnless(is_npu(), "requires an Ascend NPU")
    def test_npu_tensor_smoke(self):
        # Run explicitly on an NPU in addition to CPU CI. This exercises host
        # scheduling transfer, searchsorted/gather/scatter and metadata copies.
        cfg = DSparkScheduleConfig(gamma=3)
        lens = ScheduleVerifyLensTopk.execute(
            confidence=torch.tensor([[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]], device="npu"),
            budget=2,
            cfg=cfg,
        )
        self.assertEqual(lens.cpu().tolist(), [3, 1])
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[3, 1], device=torch.device("npu"), grid=[4]
        )
        prefix = torch.tensor([15, 31], device="npu")
        table = torch.arange(3 * 64, dtype=torch.int32, device="npu").view(3, 64)
        candidates = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="npu")
        packed = build_ragged_verify_window(
            batch=SimpleNamespace(
                seq_lens=prefix, req_pool_indices=torch.tensor([2, 0], device="npu")
            ),
            layout=layout,
            draft_block_ids=candidates,
            draft_tokens=candidates[:, 1:],
            bs=2,
            device="npu",
            verify_num_draft_tokens=4,
            model_runner=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(req_to_token=table)
            ),
        )
        self.assertEqual(packed.verify_ids.cpu().tolist(), [1, 2, 3, 5])
        self.assertEqual(packed.verify_cache_loc.cpu().tolist(), [143, 144, 145, 31])
        result = dspark_compact.build_compact_verify_metadata(
            prefix_lens=prefix, layout=layout, max_verify_len=4, num_tokens=4
        )
        self.assertEqual(result.kv_lens_cpu.tolist(), [18, 32])
        rows = scatter_compact_to_strided(
            compact=packed.verify_ids[:, None],
            layout=layout,
            fill_value=-1,
            verify_num_draft_tokens=4,
        )
        self.assertEqual(rows.cpu().flatten().tolist(), [1, 2, 3, -1, 5, -1, -1, -1])

    def test_sparse_operator_receives_compact_boundaries(self):
        op = Mock(return_value=(torch.zeros(4, 1, 4), None, None))
        method = _backend_method(
            "forward_sparse",
            torch_npu=SimpleNamespace(npu_sparse_flash_attention=op),
            _expand_dsa_sparse_indices=lambda x: x.unsqueeze(-2),
        )
        metadata = SimpleNamespace(
            actual_seq_lengths_q=torch.tensor([1, 4], dtype=torch.int32),
            actual_seq_lengths_kv=torch.tensor([16, 34], dtype=torch.int32),
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=SimpleNamespace(
                get_kv_buffer=lambda _: (torch.zeros(4, 4), torch.zeros(4, 2))
            ),
            _pad_topk_indices=lambda x, _: x,
        )
        method(
            backend,
            torch.zeros(4, 1, 4),
            None,
            None,
            SimpleNamespace(layer_id=0, scaling=0.5),
            SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
            save_kv_cache=False,
            q_rope=torch.zeros(4, 1, 2),
            topk_indices=torch.zeros(4, 2, dtype=torch.int32),
        )
        args = op.call_args.kwargs
        self.assertEqual(args["actual_seq_lengths_query"].tolist(), [1, 4])
        self.assertEqual(args["actual_seq_lengths_kv"].tolist(), [16, 34])
        self.assertEqual(args["sparse_mode"], 3)
        self.assertEqual(args["sparse_indices"].shape, (4, 1, 2))

    def test_fia_operator_receives_compact_boundaries(self):
        method = _backend_method(
            "forward_mtp", AttentionType=SimpleNamespace(ENCODER_ONLY="encoder")
        )
        metadata = SimpleNamespace(
            seq_lens_cpu_int=torch.tensor([16, 34], dtype=torch.int32),
            compact_seq_lengths_q=[1, 4],
            block_tables=torch.tensor([[0], [1]]),
        )
        backend = SimpleNamespace(
            use_mla=False,
            page_size=16,
            graph_mode=False,
            supports_ragged_verify_graph=False,
            is_hybrid_swa=False,
            forward_metadata=metadata,
            mtp_mask=torch.ones(16, 16),
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _: torch.zeros(32, 1, 4),
                get_value_buffer=lambda _: torch.zeros(32, 1, 4),
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=4,
            v_head_dim=4,
            scaling=0.5,
            sliding_window_size=-1,
            attn_type="decoder",
        )
        # Physical MLP padding is independent of logical query boundaries.
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            global_num_token_non_padded_cpu=4,
        )
        with patch.object(
            torch.ops.npu,
            "npu_fused_infer_attention_score",
            create=True,
            return_value=(torch.zeros(4, 1, 4), None),
        ) as op:
            output = method(
                backend,
                torch.zeros(8, 1, 4),
                None,
                None,
                layer,
                batch,
                save_kv_cache=False,
            )
        self.assertEqual(output.shape, (8, 4))
        self.assertEqual(op.call_args.kwargs["actual_seq_lengths"], [1, 4])
        self.assertEqual(op.call_args.kwargs["actual_seq_lengths_kv"], [16, 34])
        self.assertEqual(op.call_args.kwargs["sparse_mode"], 3)

    def test_pack_scatter_accept_commit_exhaustive(self):
        # All 64 layouts for B=3, W=4. Reordered request slots and random KV
        # addresses catch accidental use of compact offsets as cache addresses.
        torch.manual_seed(37)
        width, bs, vocab = 4, 3, 32
        table = torch.randperm(5 * 64).view(5, 64).to(torch.int32)
        prefix = torch.tensor([15, 31, 47])
        reqs = torch.tensor([4, 0, 2])
        candidates = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        predictions = torch.tensor([[2, 3, 4, 20], [6, 21, 8, 22], [23, 11, 12, 24]])
        dense_logits = torch.full((bs, width, vocab), -10.0)
        dense_logits.scatter_(2, predictions.unsqueeze(-1), 10.0)
        for lens in itertools.product(range(1, width + 1), repeat=bs):
            with self.subTest(lens=lens):
                layout = layout_for(list(lens))
                batch = SimpleNamespace(seq_lens=prefix, req_pool_indices=reqs)
                runner = SimpleNamespace(
                    req_to_token_pool=SimpleNamespace(req_to_token=table)
                )
                with patch(
                    "sglang.kernels.ops.speculative.dspark.dspark_verify_window._is_npu",
                    True,
                ):
                    packed = build_ragged_verify_window(
                        batch=batch,
                        layout=layout,
                        draft_block_ids=candidates,
                        draft_tokens=candidates[:, 1:],
                        bs=bs,
                        device="cpu",
                        verify_num_draft_tokens=width,
                        model_runner=runner,
                    )
                expected_ids, expected_positions, expected_locs = [], [], []
                for row, length in enumerate(lens):
                    expected_ids.extend(candidates[row, :length].tolist())
                    expected_positions.extend(
                        range(int(prefix[row]), int(prefix[row]) + length)
                    )
                    expected_locs.extend(
                        table[reqs[row], prefix[row] : prefix[row] + length].tolist()
                    )
                self.assertEqual(packed.verify_ids.tolist(), expected_ids)
                self.assertEqual(packed.positions.tolist(), expected_positions)
                self.assertEqual(packed.verify_cache_loc.tolist(), expected_locs)
                self.assertEqual(packed.verify_cache_loc.dtype, torch.int32)
                compact_logits = torch.cat(
                    [dense_logits[r, :n] for r, n in enumerate(lens)]
                )
                strided = scatter_compact_to_strided(
                    compact=compact_logits,
                    layout=layout,
                    fill_value=0.0,
                    verify_num_draft_tokens=width,
                )
                correct, bonus, trim = accept_greedy(
                    candidates=candidates,
                    target_logits=strided,
                    verify_num_draft_tokens=width,
                    cutoff_verify_lens=layout.verify_lens,
                )
                reference_correct, reference_bonus = [], []
                for row, length in enumerate(lens):
                    accepted = 0
                    while (
                        accepted < length - 1
                        and candidates[row, accepted + 1] == predictions[row, accepted]
                    ):
                        accepted += 1
                    reference_correct.append(accepted)
                    reference_bonus.append(int(predictions[row, accepted]))
                self.assertEqual(correct.tolist(), reference_correct)
                self.assertEqual(bonus.tolist(), reference_bonus)
                finalized = finalize_accept_lens(
                    correct_len=correct, cap_trim_lens=trim, prefix_lens=prefix
                )
                self.assertEqual(
                    finalized.commit_lens.tolist(), [x + 1 for x in reference_correct]
                )
                self.assertTrue(
                    bool(torch.all(finalized.commit_lens <= layout.verify_lens))
                )
                self.assertEqual(
                    finalized.new_seq_lens.tolist(),
                    (prefix + torch.tensor(reference_correct) + 1).tolist(),
                )

    def test_budget_allocates_prefixes_and_ties_deterministically(self):
        cfg = DSparkScheduleConfig(gamma=3)
        confidence = torch.tensor([[0.9, 0.8, 0.7], [0.2, 0.1, 0.1]])
        self.assertEqual(
            ScheduleVerifyLensTopk.execute(
                confidence=confidence, budget=2, cfg=cfg
            ).tolist(),
            [3, 1],
        )
        for budget in range(7):
            lens = schedule_verify_lens_topk(
                confidence=torch.ones(2, 3), budget=budget, cfg=cfg
            )
            self.assertEqual(int((lens - 1).sum()), budget)
            self.assertLessEqual(int((lens[0] - lens[1]).abs()), 1)

    @patch("sglang.srt.speculative.dspark_components.dspark_planner._is_npu", True)
    def test_forced_budget_overrides_verify_all_for_profiling(self):
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._budget_planner = SimpleNamespace(
            sps_table=build_uninitialized_sps_table(max_batch_tokens=32),
            forced_budget_frac=None,
        )
        planner._is_verify_all = True
        planner.set_forced_budget_frac(0.25)
        self.assertFalse(planner.is_verify_all)
        self.assertEqual(planner._budget_planner.forced_budget_frac, 0.25)
        planner.set_forced_budget_frac(None)
        self.assertTrue(planner.is_verify_all)


class TestNpuCompactConfiguration(CustomTestCase):
    def config(self, **changes):
        decode_backend = changes.pop("decode_backend", "disabled")
        prefill_backend = changes.pop("prefill_backend", "disabled")
        values = dict(
            device="npu",
            disable_cuda_graph=True,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=decode_backend),
                prefill=SimpleNamespace(backend=prefill_backend),
            ),
            disable_overlap_schedule=True,
            enable_dp_attention=False,
            dp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            pp_size=1,
            disaggregation_mode="null",
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_supported_configuration(self):
        dspark_compact.validate_npu_dspark_compact(self.config(), "compact")
        dspark_compact.validate_npu_dspark_compact(
            self.config(decode_backend="full", disable_overlap_schedule=False),
            "compact",
        )
        # Per-phase flags can disable graphs without setting the legacy flag.
        dspark_compact.validate_npu_dspark_compact(
            self.config(disable_cuda_graph=False), "compact"
        )

    def test_unsupported_combinations_fail(self):
        for changes in (
            dict(decode_backend="breakable"),
            dict(enable_dp_attention=True),
            dict(dp_size=2),
            dict(attn_cp_size=2),
            dict(dcp_size=2),
            dict(pp_size=2),
            dict(disaggregation_mode="decode"),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dspark_compact.validate_npu_dspark_compact(
                    self.config(**changes), "compact"
                )

    def test_other_modes_and_devices_unchanged(self):
        for mode in ("static", "cap-accept"):
            dspark_compact.validate_npu_dspark_compact(
                self.config(disable_cuda_graph=False), mode
            )
        dspark_compact.validate_npu_dspark_compact(
            self.config(device="cuda", disable_cuda_graph=False), "compact"
        )


if __name__ == "__main__":
    unittest.main()
