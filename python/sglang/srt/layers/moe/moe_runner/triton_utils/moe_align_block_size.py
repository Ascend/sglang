from __future__ import annotations

from typing import Tuple

import torch
import triton

from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_npu = is_npu()

if _is_cuda or _is_hip or _is_xpu or _is_musa:
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    if _is_npu:
        return _moe_align_block_size_python(topk_ids, block_size, num_experts)

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )

    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def _moe_align_block_size_python(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure Python implementation of moe_align_block_size for NPU."""
    # topk_ids shape: [total_tokens, top_k]
    # Flatten to get all expert assignments
    flat_ids = topk_ids.view(-1)
    total_tokens = flat_ids.size(0)

    # Count tokens per expert
    token_counts = torch.bincount(flat_ids, minlength=num_experts + 1)

    # Calculate padded tokens per expert (round up to block_size)
    padded_counts = ((token_counts + block_size - 1) // block_size) * block_size

    # Calculate cumulative sum for positioning
    cum_counts = torch.zeros(num_experts + 2, dtype=torch.int32, device=topk_ids.device)
    cum_counts[1:] = torch.cumsum(padded_counts, dim=0)

    total_padded = cum_counts[-1].item()
    max_num_m_blocks = triton.cdiv(total_padded, block_size)

    # Build sorted_token_ids and expert_ids
    sorted_token_ids = torch.full(
        (total_padded,), num_experts, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )

    # Place tokens for each expert
    for expert_id in range(num_experts + 1):
        start_pos = cum_counts[expert_id].item()
        # Find tokens belonging to this expert
        expert_mask = flat_ids == expert_id
        expert_token_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
        count = expert_token_indices.size(0)
        if count > 0:
            sorted_token_ids[start_pos : start_pos + count] = expert_token_indices.to(
                torch.int32
            )
        # Fill expert_ids for each block
        num_blocks = (padded_counts[expert_id].item()) // block_size
        block_start = start_pos // block_size
        expert_ids[block_start : block_start + num_blocks] = expert_id

    num_tokens_post_padded = torch.tensor(
        [total_padded], dtype=torch.int32, device=topk_ids.device
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded
