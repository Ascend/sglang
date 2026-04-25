"""Unit test for CPU precomputation of GDN chunked-prefill indices.

Validates that `build_gdn_chunked_prefill_meta` produces tensors that are
value- and dtype-equal to the in-tree lazy helpers `prepare_chunk_indices`
and `prepare_chunk_offsets` over a `cu_seqlens` of the requested dtype.
"""

import unittest

import torch

from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.linear.gdn_chunk_meta import (
    build_gdn_chunked_prefill_meta,
)


def _cu_seqlens_from(seq_lens, dtype):
    return torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=dtype
    )


class TestGDNChunkMeta(unittest.TestCase):

    DTYPES = (torch.int32, torch.int64)

    def _check(self, seq_lens, chunk_size, dtype):
        device = torch.device("cpu")
        cu_seqlens = _cu_seqlens_from(seq_lens, dtype)

        expected_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        expected_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)

        meta = build_gdn_chunked_prefill_meta(
            extend_seq_lens_cpu=list(seq_lens),
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
            use_pinned_memory=False,
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.chunk_indices.dtype, expected_indices.dtype)
        self.assertEqual(meta.chunk_offsets.dtype, expected_offsets.dtype)
        self.assertEqual(meta.chunk_indices.dtype, dtype)
        self.assertEqual(meta.chunk_offsets.dtype, dtype)
        torch.testing.assert_close(meta.chunk_indices.cpu(), expected_indices.cpu())
        torch.testing.assert_close(meta.chunk_offsets.cpu(), expected_offsets.cpu())

    def _check_all_dtypes(self, seq_lens, chunk_size):
        for dtype in self.DTYPES:
            with self.subTest(seq_lens=seq_lens, chunk_size=chunk_size, dtype=dtype):
                self._check(seq_lens, chunk_size, dtype)

    def test_single_seq_exact_multiple(self):
        self._check_all_dtypes([128], chunk_size=64)

    def test_single_seq_ragged(self):
        self._check_all_dtypes([100], chunk_size=64)

    def test_mixed_batch(self):
        self._check_all_dtypes([64, 65, 130, 1], chunk_size=64)

    def test_small_chunk(self):
        self._check_all_dtypes([3, 7, 1], chunk_size=2)

    def test_empty_batch_returns_none(self):
        meta = build_gdn_chunked_prefill_meta(
            extend_seq_lens_cpu=[],
            chunk_size=64,
            device=torch.device("cpu"),
            use_pinned_memory=False,
        )
        self.assertIsNone(meta)

    def test_default_dtype_is_int32(self):
        meta = build_gdn_chunked_prefill_meta(
            extend_seq_lens_cpu=[10],
            chunk_size=4,
            device=torch.device("cpu"),
            use_pinned_memory=False,
        )
        self.assertEqual(meta.chunk_indices.dtype, torch.int32)
        self.assertEqual(meta.chunk_offsets.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
