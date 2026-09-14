"""Full-rank K3 Q/K/V/G checkpoint mapping must not use the low-rank flag."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKimiK3FusedWeights(CustomTestCase):
    def test_full_rank_loads_all_four_shards_with_low_rank_fusion_disabled(self):
        param = torch.nn.Parameter(torch.empty(8, 4))

        def load_shard(param, weight, shard_id):
            param.data[2 * shard_id : 2 * (shard_id + 1)].copy_(weight)

        param.weight_loader = Mock(side_effect=load_shard)
        name = "model.layers.0.self_attn.fused_qkvg_proj.weight"
        owner = SimpleNamespace(
            config=SimpleNamespace(
                linear_attn_config={"use_full_rank_gate": True},
                is_moe=False,
                num_hidden_layers=1,
                is_kda_layer=lambda layer_id: layer_id == 0,
            ),
            model=SimpleNamespace(
                start_layer=0,
                end_layer=1,
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            use_full_rank_gate=True, do_fuse_qkvbfg=False
                        )
                    )
                ],
            ),
            named_parameters=lambda: [(name, param)],
            post_load_weights=Mock(),
        )
        weights = [
            (
                f"model.layers.0.self_attn.{proj}_proj.weight",
                torch.full((2, 4), float(i)),
            )
            for i, proj in enumerate(("q", "k", "v", "g"))
        ]
        loaded = KimiK3LinearForCausalLM.load_weights(owner, weights)
        self.assertEqual(loaded, {name})
        self.assertEqual(
            [call.args[2] for call in param.weight_loader.call_args_list], [0, 1, 2, 3]
        )
        torch.testing.assert_close(
            param, torch.cat([weight for _, weight in weights]), rtol=0, atol=0
        )
        owner.post_load_weights.assert_called_once()


if __name__ == "__main__":
    unittest.main()
