"""CPU regressions for A5 MXFP8 DeepEP low-latency dispatch wiring."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DeepEPMode, DispatcherOutputDtype
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNPUMXFP8DeepEPDispatch(CustomTestCase):
    def test_mode_specific_dtype_selection(self):
        quant_config = {
            "normal_dispatcher_output_dtype": "bf16",
            "low_latency_dispatcher_output_dtype": "mxfp8",
        }
        common = dict(
            quant_config=quant_config,
        )
        with (
            get_context().override_server_args(deepep_dispatcher_output_dtype="auto"),
            moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH.override(False),
        ):
            normal = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(**common, dispatch_mode=DeepEPMode.NORMAL)
            )
            low_latency = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(**common, dispatch_mode=DeepEPMode.LOW_LATENCY)
            )

        self.assertEqual(normal, DispatcherOutputDtype.BF16)
        self.assertEqual(low_latency, DispatcherOutputDtype.MXFP8)

    def test_mxfp8_config_selects_explicit_deepep_quant_mode(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        dispatcher.quant_config = {"low_latency_dispatcher_output_dtype": "mxfp8"}
        with patch.object(
            deepep, "get_deepep_output_dtype", return_value=DispatcherOutputDtype.MXFP8
        ):
            dispatcher.set_deepep_dispatcher_dtype()

        self.assertTrue(dispatcher.use_fp8)
        self.assertFalse(dispatcher.use_nvfp4)
        self.assertEqual(dispatcher.quant_mode, "mx_fp8_e4m3")

    def test_low_latency_forwards_explicit_quant_mode(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        dispatcher.quant_config = {}
        dispatcher.num_max_dispatch_tokens_per_rank = 64
        dispatcher.num_experts = 512
        dispatcher.use_fp8 = True
        dispatcher.use_nvfp4 = False
        dispatcher.quant_mode = "mx_fp8_e4m3"
        dispatcher.return_recv_hook = False

        event, hook = Mock(), Mock()
        buffer = Mock()
        buffer.low_latency_dispatch.return_value = (
            (Mock(), Mock()),
            Mock(),
            Mock(),
            event,
            hook,
        )
        dispatcher._get_buffer = Mock(return_value=buffer)

        hidden_states = torch.empty((8, 128), dtype=torch.bfloat16)
        topk_ids = torch.empty((8, 2), dtype=torch.int64)
        topk_weights = torch.empty((8, 2), dtype=torch.float32)
        with patch.object(deepep, "_deepep_precompile_tp_barrier"):
            dispatcher._dispatch_core(hidden_states, topk_ids, topk_weights)

        kwargs = buffer.low_latency_dispatch.call_args.kwargs
        self.assertEqual(kwargs["quant_mode"], "mx_fp8_e4m3")

    def test_switching_from_mxfp8_clears_quant_mode(self):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        dispatcher.quant_config = {}
        for dtype, expected_fp8, expected_mode in (
            (DispatcherOutputDtype.MXFP8, True, "mx_fp8_e4m3"),
            (DispatcherOutputDtype.BF16, False, None),
            (DispatcherOutputDtype.MXFP8, True, "mx_fp8_e4m3"),
            (DispatcherOutputDtype.INT8, True, None),
        ):
            with self.subTest(dtype=dtype), patch.object(deepep, "_is_npu", True):
                with patch.object(
                    deepep, "get_deepep_output_dtype", return_value=dtype
                ):
                    dispatcher.set_deepep_dispatcher_dtype()
                self.assertEqual(dispatcher.deepep_output_dtype, dtype)
                self.assertEqual(dispatcher.use_fp8, expected_fp8)
                self.assertFalse(dispatcher.use_nvfp4)
                self.assertEqual(dispatcher.quant_mode, expected_mode)

    def test_legacy_dispatch_omits_mxfp8_keyword_for_both_completion_modes(self):
        hidden = torch.zeros(2, 128, dtype=torch.bfloat16)
        ids = torch.zeros(2, 2, dtype=torch.int64)
        weights = torch.ones(2, 2)
        for recv_hook in (False, True):
            with self.subTest(recv_hook=recv_hook):
                dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
                dispatcher.quant_config = {}
                dispatcher.num_max_dispatch_tokens_per_rank = 64
                dispatcher.num_experts = 512
                dispatcher.use_fp8 = dispatcher.use_nvfp4 = False
                dispatcher.quant_mode = None
                dispatcher.return_recv_hook = recv_hook
                packed, counts, handle, event, hook = (object() for _ in range(5))
                buffer = Mock()
                buffer.low_latency_dispatch.return_value = (
                    packed,
                    counts,
                    handle,
                    event,
                    hook,
                )
                dispatcher._get_buffer = Mock(return_value=buffer)
                with patch.object(deepep, "_deepep_precompile_tp_barrier"):
                    output = dispatcher._dispatch_core(hidden, ids, weights)
                kwargs = buffer.low_latency_dispatch.call_args.kwargs
                self.assertNotIn("quant_mode", kwargs)
                self.assertNotIn("use_nvfp4", kwargs)
                self.assertFalse(kwargs["use_fp8"])
                self.assertEqual(kwargs["return_recv_hook"], recv_hook)
                self.assertEqual(kwargs["async_finish"], not recv_hook)
                self.assertEqual(output, (packed, counts, event, hook))
                self.assertIs(dispatcher.handle, handle)

    def test_missing_mode_override_falls_back_to_generic_dtype(self):
        with (
            get_context().override_server_args(deepep_dispatcher_output_dtype="auto"),
            moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH.override(False),
        ):
            for mode, expected in (
                (DeepEPMode.NORMAL, DispatcherOutputDtype.BF16),
                (DeepEPMode.LOW_LATENCY, DispatcherOutputDtype.MXFP8),
            ):
                with self.subTest(mode=mode):
                    dispatcher = SimpleNamespace(
                        dispatch_mode=mode,
                        quant_config={
                            "dispatcher_output_dtype": "bf16",
                            "low_latency_dispatcher_output_dtype": "mxfp8",
                        },
                    )
                    self.assertEqual(
                        moe_utils.get_deepep_output_dtype(dispatcher), expected
                    )


if __name__ == "__main__":
    unittest.main()
