import os

import deep_ep
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

os.environ.setdefault("DEEP_USE_MODE", "default")


WORLD_SIZE = 16
NUM_EXPERTS = 256
NUM_TOPK = 8


def _run_rank(local_rank: int, world_size: int) -> None:
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=local_rank,
        world_size=world_size,
    )
    group = dist.new_group(list(range(world_size)))

    try:
        buffer = deep_ep.Buffer(
            group,
            int(2e9),
            0,
            low_latency_mode=False,
            num_qps_per_rank=1,
            normal_strategy="default",
            low_latency_strategy="default",
        )

        # This matches SGLang's DP-attention idle path: only the active DP rank
        # owns a token, while every idle rank still participates in DeepEP with
        # a contiguous [0, num_topk] routing tensor.
        num_tokens = 1 if local_rank == 0 else 0
        if num_tokens:
            topk_idx = torch.arange(
                NUM_TOPK, dtype=torch.int64, device="npu"
            ).reshape(1, NUM_TOPK)
        else:
            topk_idx = torch.empty(
                (0, NUM_TOPK), dtype=torch.int64, device="npu"
            )

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(topk_idx, NUM_EXPERTS)

        assert num_tokens_per_rdma_rank is None
        assert isinstance(event, deep_ep.EventOverlap)
        assert tuple(num_tokens_per_rank.shape) == (world_size,)
        assert tuple(is_token_in_rank.shape) == (num_tokens, world_size)

        expected_routes = NUM_TOPK if num_tokens else 0
        assert int(num_tokens_per_expert.sum().item()) == expected_routes
        if num_tokens == 0:
            assert int(num_tokens_per_rank.sum().item()) == 0
            assert is_token_in_rank.numel() == 0

        print(
            f"rank={local_rank} num_tokens={num_tokens} layout passed",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


def test_deepep_dispatch_layout_accepts_idle_ranks() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29567")
    torch.multiprocessing.spawn(
        _run_rank,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    test_deepep_dispatch_layout_accepts_idle_ranks()
