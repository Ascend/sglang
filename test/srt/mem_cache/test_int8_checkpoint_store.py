"""Regression tests for native conv-state layouts in int8 mamba checkpoints."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    MambaCheckpointPool,
    maybe_init_int8_mamba_checkpoint_pool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

H, V, K, L = 32, 128, 128, 4


class TestInt8CheckpointConvLayout(unittest.TestCase):
    def test_extended_conv_state_round_trip_preserves_native_layout(self):
        conv = torch.arange(L * 8 * 7 * 5, dtype=torch.float32).reshape(L, 8, 7, 5)
        temporal = torch.randn(L, 8, H, V, K)
        active = SimpleNamespace(
            mamba_cache=SimpleNamespace(conv=[conv], temporal=temporal)
        )
        checkpoint = MambaCheckpointPool(
            num_layers=L,
            num_slots=4,
            num_heads=H,
            head_v_dim=V,
            head_k_dim=K,
            conv_shapes=[(7, 5)],
            conv_dtype=conv.dtype,
            temporal_dtype=temporal.dtype,
            device="cpu",
        )
        source_slot, checkpoint_slot, dst_slot = (
            torch.tensor([2]),
            torch.tensor([1]),
            torch.tensor([4]),
        )
        expected = conv[:, source_slot].clone()
        checkpoint.store_from_active(active, source_slot, checkpoint_slot)
        self.assertTrue(torch.equal(checkpoint.conv[0][:, checkpoint_slot], expected))
        conv[:, dst_slot] = 0
        checkpoint.load_to_active(active, checkpoint_slot, dst_slot)
        self.assertTrue(torch.equal(conv[:, dst_slot], expected))

    def test_factory_uses_active_conv_shape_for_allocation_and_estimate(self):
        ckpt_size, active_shape = 3, (5, 5)
        cache_params = SimpleNamespace(
            shape=SimpleNamespace(temporal=(H, V, K), conv=[(3, 5)]),
            dtype=SimpleNamespace(conv=torch.bfloat16, temporal=torch.bfloat16),
        )
        server_args = SimpleNamespace(
            enable_int8_mamba_checkpoint=True,
            int8_mamba_ckpt_size=ckpt_size,
        )
        with patch(
            "sglang.srt.mem_cache.mamba_checkpoint_pool.get_global_server_args",
            return_value=server_args,
        ):
            pool = maybe_init_int8_mamba_checkpoint_pool(
                mamba_size=2,
                cache_params=cache_params,
                conv_shapes=[active_shape],
                mamba_layer_ids=list(range(L)),
                device="cpu",
            )

        self.assertEqual(tuple(pool.conv[0].shape), (L, ckpt_size + 1, *active_shape))
        estimate = MambaCheckpointPool.estimate_mem_usage_bytes(
            num_layers=L,
            num_slots=ckpt_size,
            num_heads=H,
            head_v_dim=V,
            head_k_dim=K,
            conv_shapes=[active_shape],
            conv_dtype=torch.bfloat16,
            temporal_dtype=torch.bfloat16,
        )
        self.assertEqual(estimate["conv"], L * (ckpt_size + 1) * 25 * 2)
        self.assertEqual(pool.mem_usage_bytes(), estimate["total"])


if __name__ == "__main__":
    unittest.main()
