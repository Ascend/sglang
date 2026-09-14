"""CPU regressions for ordinary MLA writes into PA-NZ storage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMLATokenToKVPool,
    _mla_fia_nz_scatter_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_pool(is_nz, page_size=16, device="cpu"):
    pool = object.__new__(NPUMLATokenToKVPool)
    pool.use_fia_nz = is_nz
    pool.page_size = page_size
    pool.kv_lora_rank = 512
    pool.qk_rope_head_dim = 64
    pool.dtype = pool.store_dtype = torch.bfloat16
    pool.start_layer = 3
    pool.k_buffer = torch.full(
        (2, 4, page_size, 1, 512), -1, dtype=pool.dtype, device=device
    )
    pool.v_buffer = torch.full(
        (2, 4, page_size, 1, 64), -1, dtype=pool.dtype, device=device
    )
    return pool


class TestMLANZWrite(CustomTestCase):
    def test_scatter_addresses(self):
        for page_size in (16, 128):
            for head_dim in (64, 512):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        page_size=page_size, head_dim=head_dim, dtype=dtype
                    ):
                        slots = [2 * page_size, page_size - 1, page_size + 1, 0]
                        loc = torch.tensor(slots, dtype=dtype)
                        rows = _mla_fia_nz_scatter_indices(loc, head_dim, page_size)
                        expected = []
                        for slot in slots:
                            page, offset = divmod(slot, page_size)
                            for tile in range(head_dim // 16):
                                expected.append(
                                    page * page_size * (head_dim // 16)
                                    + tile * page_size
                                    + offset
                                )
                        self.assertEqual(rows.flatten().tolist(), expected)
                        self.assertEqual(rows.dtype, dtype)
                        self.assertEqual(
                            _mla_fia_nz_scatter_indices(
                                loc[:0], head_dim, page_size
                            ).shape,
                            (0, 1),
                        )

    def test_invalid_geometry(self):
        loc = torch.tensor([0], dtype=torch.int32)
        for head_dim, page_size in ((63, 16), (64, 0), (64, -1)):
            with self.subTest(head_dim=head_dim, page_size=page_size):
                with self.assertRaises(ValueError):
                    _mla_fia_nz_scatter_indices(loc, head_dim, page_size)

    def test_combined_and_separate_writes_round_trip(self):
        def scatter(dst, indices, src):
            dst.index_copy_(0, indices.flatten().long(), src)

        npu = SimpleNamespace(npu_scatter_nd_update_=scatter)
        for is_nz in (False, True):
            for combined in (False, True):
                with self.subTest(is_nz=is_nz, combined=combined):
                    pool = make_pool(is_nz)
                    # Out-of-order writes straddle page boundaries. A second
                    # write replaces one slot without disturbing its neighbors.
                    loc = torch.tensor([32, 15, 17, 0], dtype=torch.int32)
                    values = (
                        torch.arange(4 * 576, dtype=torch.float32).reshape(4, 1, 576)
                        % 127
                    )
                    with patch.object(memory_pool_npu, "torch_npu", npu, create=True):
                        k, v = values.split((512, 64), dim=-1)
                        pool.set_kv_buffer(
                            SimpleNamespace(layer_id=4),
                            loc,
                            values if combined else k,
                            None if combined else v,
                        )
                        pool.set_kv_buffer(
                            SimpleNamespace(layer_id=4), loc[:1], values[:1] + 1, None
                        )
                    values[0] += 1
                    for dim, cache, source in (
                        (512, pool.k_buffer, values[..., :512]),
                        (64, pool.v_buffer, values[..., 512:]),
                    ):
                        expected = torch.full((4, 16, 1, dim), -1, dtype=pool.dtype)
                        expected.view(-1, 1, dim)[loc.long()] = source.to(pool.dtype)
                        ids = torch.tensor([2, 0, 1, 2], dtype=torch.int32)
                        actual = gather_mla_cache_pages(cache[1], ids, is_nz=is_nz)
                        torch.testing.assert_close(
                            actual, expected[ids.long()], rtol=0, atol=0
                        )
                        self.assertTrue(torch.all(cache[0] == -1))


if __name__ == "__main__":
    unittest.main()
