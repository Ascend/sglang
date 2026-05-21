"""CPU-side precomputation of GDN chunked-prefill metadata.

Motivation: the fla `prepare_chunk_indices` / `prepare_chunk_offsets` helpers
call `.tolist()` on a GPU/NPU tensor inside every layer's forward, which
introduces a D2H sync on the hot path. For GDN non-spec extend on Ascend NPU,
all inputs (`extend_seq_lens_cpu`) are already known on the host at
`init_forward_metadata` time, so we can build these indices once per step on
the CPU and asynchronously H2D-copy them into pinned memory, making them free
to consume later in every GDN layer.

Scope: this module covers the metadata that SGLang can derive from already
known host sequence lengths: `chunk_indices`, `chunk_offsets`, and host shape
values used by Ascend NPU kernels to avoid device-to-host scalar reads.
Other precomputed tensors (e.g. for large-block triu / cumulative block
indices in vllm-ascend's PR) depend on the consuming kernel's exact signature
and are intentionally left for a follow-up once the matching `sgl_kernel_npu`
side lands.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18


@dataclass
class GDNChunkedPrefillMetadata:
    """Precomputed chunk metadata for GDN chunked prefill.

    `chunk_indices` and `chunk_offsets` are compatibility aliases for the
    chunk-64 fields. They mirror the return values of
    `sglang.srt.layers.attention.fla.index.prepare_chunk_indices` and
    `prepare_chunk_offsets`, respectively. The host fields let external
    kernels build launch shapes without reading scalar values from device
    tensors.
    """

    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    max_T: int
    cu_seq_len: int
    query_start_loc_cpu: List[int]


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _chunk_counts(seq_lens: List[int], chunk_size: int) -> List[int]:
    return [_cdiv(seq_len, chunk_size) for seq_len in seq_lens]


def _chunk_indices_rows(chunk_counts: List[int]) -> List[List[int]]:
    rows: List[List[int]] = []
    for seq_idx, n_chunks in enumerate(chunk_counts):
        for chunk_idx in range(n_chunks):
            rows.append([seq_idx, chunk_idx])
    return rows


def _chunk_offsets_list(chunk_counts: List[int]) -> List[int]:
    offsets = [0]
    running = 0
    for n_chunks in chunk_counts:
        running += n_chunks
        offsets.append(running)
    return offsets


def _update_chunk_offsets_list(chunk_counts: List[int]) -> List[int]:
    offsets = [0]
    running = 0
    for n_chunks in chunk_counts:
        running += n_chunks + 1
        offsets.append(running)
    return offsets


def _final_chunk_indices_list(chunk_counts: List[int]) -> List[int]:
    running = 0
    indices = []
    for n_chunks in chunk_counts:
        running += n_chunks + 1
        indices.append(running - 1)
    return indices


def _to_device_tensor(
    values,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    use_pinned_memory: bool,
) -> torch.Tensor:
    if values:
        tensor_cpu = torch.tensor(values, dtype=dtype)
    else:
        tensor_cpu = torch.empty(shape, dtype=dtype)

    if use_pinned_memory:
        try:
            tensor_cpu = tensor_cpu.pin_memory()
        except (RuntimeError, NotImplementedError):
            # Pinning not supported on this backend; fall back silently.
            pass

    return tensor_cpu.to(device, non_blocking=True)


def build_gdn_chunked_prefill_meta(
    extend_seq_lens_cpu: List[int],
    chunk_size: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.int32,
    use_pinned_memory: bool = True,
    num_heads: Optional[int] = None,
) -> Optional[GDNChunkedPrefillMetadata]:
    """Build chunk metadata and host shape values from per-sequence lengths.

    Returns None when there is nothing to build (empty batch), letting callers
    fall back to the legacy lazy path.

    The CPU construction exactly matches `prepare_chunk_indices` /
    `prepare_chunk_offsets` over a `cu_seqlens` whose diffs equal
    `extend_seq_lens_cpu`; see `test_gdn_chunk_meta.py`.
    """
    if not extend_seq_lens_cpu:
        return None

    seq_lens = [int(seq_len) for seq_len in extend_seq_lens_cpu]
    query_start_loc_cpu: List[int] = [0]
    cu_seq_len = 0
    max_T = 0
    for seq_len in seq_lens:
        cu_seq_len += seq_len
        max_T = max(max_T, seq_len)
        query_start_loc_cpu.append(cu_seq_len)

    chunk_counts_chunk64 = _chunk_counts(seq_lens, chunk_size)
    chunk_counts_large = _chunk_counts(seq_lens, _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE)
    if num_heads is None:
        cumsum_block_size = chunk_size
    else:
        cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (num_heads * chunk_size))
        cumsum_block_size = _next_power_of_2(cumsum_chunks)
    chunk_counts_cumsum = _chunk_counts(seq_lens, cumsum_block_size)

    chunk_indices_chunk64 = _to_device_tensor(
        _chunk_indices_rows(chunk_counts_chunk64),
        shape=(0, 2),
        dtype=dtype,
        device=device,
        use_pinned_memory=use_pinned_memory,
    )
    chunk_offsets_chunk64 = _to_device_tensor(
        _chunk_offsets_list(chunk_counts_chunk64),
        shape=(0,),
        dtype=dtype,
        device=device,
        use_pinned_memory=use_pinned_memory,
    )
    update_chunk_offsets_chunk64 = _to_device_tensor(
        _update_chunk_offsets_list(chunk_counts_chunk64),
        shape=(0,),
        dtype=dtype,
        device=device,
        use_pinned_memory=use_pinned_memory,
    )
    final_chunk_indices_chunk64 = _to_device_tensor(
        _final_chunk_indices_list(chunk_counts_chunk64),
        shape=(0,),
        dtype=dtype,
        device=device,
        use_pinned_memory=use_pinned_memory,
    )
    chunk_indices_large_block = _to_device_tensor(
        _chunk_indices_rows(chunk_counts_large),
        shape=(0, 2),
        dtype=dtype,
        device=device,
        use_pinned_memory=use_pinned_memory,
    )
    block_indices_cumsum = _to_device_tensor(
        _chunk_indices_rows(chunk_counts_cumsum),
        shape=(0, 2),
        dtype=dtype,
        device=device,
        use_pinned_memory=use_pinned_memory,
    )

    return GDNChunkedPrefillMetadata(
        chunk_indices_chunk64=chunk_indices_chunk64,
        chunk_offsets_chunk64=chunk_offsets_chunk64,
        update_chunk_offsets_chunk64=update_chunk_offsets_chunk64,
        final_chunk_indices_chunk64=final_chunk_indices_chunk64,
        chunk_indices_large_block=chunk_indices_large_block,
        block_indices_cumsum=block_indices_cumsum,
        chunk_indices=chunk_indices_chunk64,
        chunk_offsets=chunk_offsets_chunk64,
        max_T=max_T,
        cu_seq_len=cu_seq_len,
        query_start_loc_cpu=query_start_loc_cpu,
    )
