"""Verify graph metadata selection and update-before-replay ordering."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
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


if __name__ == "__main__":
    unittest.main()
