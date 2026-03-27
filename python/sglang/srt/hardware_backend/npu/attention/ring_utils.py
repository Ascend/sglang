from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
)

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    lse_trans: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if lse_trans:
        block_lse = block_lse.transpose(1, 2)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    lse_trans: bool = True,
    last_lse_trans: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = block_out
        if lse_trans:
            lse = block_lse.transpose(1, 2)
        else:
            lse = block_lse
    else:
        if last_lse_trans:
            lse = lse.transpose(1, 2)
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse, lse_trans)
    return out, lse


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.pcp_rank = get_attention_cp_rank()
        self.pcp_size = get_attention_cp_size()
        self._reqs = None

        pcp_ranks = dist.get_process_group_ranks(self._process_group)
        self.send_rank = pcp_ranks[(self.pcp_rank + 1) % self.pcp_size]
        self.recv_rank = pcp_ranks[self.pcp_rank - 1]

    def send_recv(
        self,
        to_send: torch.Tensor,
    ) -> torch.Tensor:
        res = torch.zeros_like(to_send)

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")

        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k), self.send_recv(v)
        self.commit()
        return next_k, next_v
