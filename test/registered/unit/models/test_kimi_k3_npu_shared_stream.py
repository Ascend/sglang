"""Keep attention-TP collectives off the shared-expert MLP stream."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKimiK3SharedStream(CustomTestCase):
    def test_collectives_stay_on_main_and_shared_mlp_overlaps_routed(self):
        events = []
        current = ["main"]

        def record(name):
            events.append((name, current[0]))

        @contextmanager
        def on_stream(stream):
            current[0] = "shared"
            try:
                yield
            finally:
                current[0] = "main"

        hidden = Mock(shape=(2, 4))
        shared_input = Mock()
        shared_input.record_stream.side_effect = lambda _: record("record_input")
        alt = Mock()
        alt.wait_stream.side_effect = lambda _: record("wait_input")
        main = Mock()
        main.wait_event.side_effect = lambda _: record("join")

        def shared(value):
            self.assertIs(value, shared_input)
            record("shared_mlp")
            return torch.ones(2, 4)

        def routed(*args):
            record("routed_mlp")
            return torch.full((2, 4), 2.0)

        def reduce_scatter(dst, src):
            record("reduce_scatter")
            dst.copy_(src)

        owner = SimpleNamespace(
            shared_experts=shared,
            _sbo_shared_overlap=True,
            _shared_experts_attn_tp_comm=True,
            alt_stream=alt,
            _ep_front=lambda _: None,
            _ep_front_overlap=lambda _: None,
            gate=lambda _: record("gate"),
            topk=Mock(),
            use_latent_moe=False,
            experts=routed,
            tp_size=1,
        )
        with (
            kimi_k3.get_parallel().override(attn_tp_group=Mock()),
            patch.object(kimi_k3, "_is_npu", True),
            patch.object(kimi_k3, "get_local_dp_buffer", return_value=shared_input),
            patch.object(
                kimi_k3,
                "attn_tp_all_gather_into_tensor",
                side_effect=lambda *args: record("all_gather"),
            ),
            patch.object(
                kimi_k3, "attn_tp_reduce_scatter_tensor", side_effect=reduce_scatter
            ),
            patch.object(torch.cuda, "current_stream", return_value=main),
            patch.object(torch.cuda, "stream", side_effect=on_stream),
            patch.object(torch, "empty_like", return_value=torch.empty(2, 4)),
        ):
            result = kimi_k3.KimiK3MoE._forward_unfused(owner, hidden, prefix_sum=None)
        torch.testing.assert_close(result, torch.full((2, 4), 3.0))
        self.assertIn(("shared_mlp", "shared"), events)
        self.assertIn(("all_gather", "main"), events)
        self.assertIn(("reduce_scatter", "main"), events)
        names = [name for name, _ in events]
        self.assertLess(names.index("all_gather"), names.index("shared_mlp"))
        self.assertLess(names.index("shared_mlp"), names.index("gate"))
        self.assertLess(names.index("routed_mlp"), names.index("join"))
        self.assertLess(names.index("join"), names.index("reduce_scatter"))


if __name__ == "__main__":
    unittest.main()
