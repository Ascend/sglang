"""Verify graph metadata selection and update-before-replay ordering."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkGraphUpdate(CustomTestCase):
    def test_capture_mode_selects_handler_even_for_idle_replay(self):
        runner = object.__new__(NPUGraphRunner)
        runner.attr_name = {
            "TARGET_VERIFY": "actual_seq_kvlen",
            AttentionArch.MLA: "actual_seq_lengths_kv",
        }
        runner.attr_type = {"TARGET_VERIFY": [], AttentionArch.MLA: np.ndarray}
        for existing_v2, dspark_v2, mode, expected_v2 in (
            (False, False, ForwardMode.TARGET_VERIFY, False),
            (False, True, ForwardMode.TARGET_VERIFY, True),
            (False, True, ForwardMode.DECODE, False),
            (True, False, ForwardMode.DECODE, True),
        ):
            with self.subTest(existing_v2=existing_v2, dspark_v2=dspark_v2, mode=mode):
                runner.if_use_v2 = existing_v2
                runner.use_fias_v2_bsnd = dspark_v2
                runner.capture_forward_mode = mode
                key = "TARGET_VERIFY" if expected_v2 else AttentionArch.MLA
                self.assertEqual(runner._get_update_attr_name(), runner.attr_name[key])
                self.assertIs(runner._get_update_attr_type(), runner.attr_type[key])

    def test_update_finishes_before_replay(self):
        backend = object.__new__(NPUCudaGraphBackend)
        state = {"kv_len": 0}
        seen = []

        def update(*, cpu_update_input):
            time.sleep(0.01)
            state["kv_len"] = cpu_update_input[0]["actual_seq_kvlen"][0]

        graph = SimpleNamespace(
            update=update, replay=lambda: seen.append(state["kv_len"])
        )
        backend._graphs = {"shape": graph}
        backend._outputs = {"shape": object()}
        backend._device_module = Mock()
        backend._device_id = 0
        for seq_len in (127, 128, 129, 0, 255):
            actual = backend.replay_with_input_update(
                "shape", None, cpu_update_input=[{"actual_seq_kvlen": [seq_len]}]
            )
            self.assertIs(actual, backend._outputs["shape"])
        self.assertEqual(seen, [127, 128, 129, 0, 255])

    def test_update_failure_does_not_replay(self):
        backend = object.__new__(NPUCudaGraphBackend)
        graph = Mock()
        graph.update.side_effect = RuntimeError("update failed")
        backend._graphs = {"shape": graph}
        backend._outputs = {"shape": None}
        backend._device_module = Mock()
        backend._device_id = 0
        with self.assertRaisesRegex(RuntimeError, "update failed"):
            backend.replay_with_input_update("shape", None, cpu_update_input=[])
        graph.replay.assert_not_called()

    def test_legacy_list_and_tensor_metadata_survive_page_boundaries(self):
        backend = object.__new__(NPUCudaGraphBackend)
        graph = Mock()
        backend._graphs = {"shape": graph}
        backend._outputs = {"shape": object()}
        backend._device_module = Mock()
        backend._device_id = 2
        lengths = [0, 127, 128, 129, 256]
        for attr_name, attr_type in (
            ("actual_seq_lengths_kv", []),
            ("context_lens", torch.empty(0)),
        ):
            with self.subTest(attr_name=attr_name):
                graph.reset_mock()
                output = backend.replay_with_input_update(
                    "shape", lengths, attr_name=attr_name, attr_type=attr_type
                )
                metadata = graph.update.call_args.kwargs["cpu_update_input"]
                self.assertEqual(len(metadata), 1)
                self.assertEqual(set(metadata[0]), {attr_name})
                actual = metadata[0][attr_name]
                if isinstance(attr_type, torch.Tensor):
                    self.assertEqual(actual.dtype, torch.int32)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertEqual(actual.tolist(), lengths)
                else:
                    self.assertEqual(actual, lengths)
                graph.replay.assert_called_once_with()
                self.assertIs(output, backend._outputs["shape"])
                backend._device_module.set_device.assert_called_with(2)

    def test_explicit_multistep_metadata_updates_only_selected_graph(self):
        backend = object.__new__(NPUCudaGraphBackend)
        selected, untouched = Mock(), Mock()
        backend._graphs = {"small": untouched, "large": selected}
        backend._outputs = {"small": object(), "large": object()}
        backend._device_module = Mock()
        backend._device_id = 0
        metadata = [
            {"actual_seq_kvlen": [127, 0]},
            {"actual_seq_kvlen": [128, 0]},
            {"actual_seq_kvlen": [129, 0]},
        ]
        output = backend.replay_with_input_update(
            "large", [999], attr_name="unused", cpu_update_input=metadata
        )
        self.assertIs(selected.update.call_args.kwargs["cpu_update_input"], metadata)
        selected.replay.assert_called_once_with()
        untouched.update.assert_not_called()
        untouched.replay.assert_not_called()
        self.assertIs(output, backend._outputs["large"])


if __name__ == "__main__":
    unittest.main()
