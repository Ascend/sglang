import inspect

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
elif is_cpu():
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu

    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )


def _kernel_accepts_prebuilt_meta(fn) -> bool:
    """Detect whether a chunk_gated_delta_rule implementation accepts a
    `prebuilt_meta` keyword argument. Used to stay compatible with older
    external `sgl_kernel_npu` builds that don't yet support the fast path."""
    return _kernel_accepts_kwarg(fn, "prebuilt_meta")


def _kernel_accepts_kwarg(fn, kwarg: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if kwarg in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


try:
    _CHUNK_GDR_ACCEPTS_PREBUILT_META = _kernel_accepts_prebuilt_meta(
        chunk_gated_delta_rule
    )
    _CHUNK_GDR_ACCEPTS_MAX_T = _kernel_accepts_kwarg(chunk_gated_delta_rule, "max_T")
    _CHUNK_GDR_ACCEPTS_CU_SEQ_LEN = _kernel_accepts_kwarg(
        chunk_gated_delta_rule, "cu_seq_len"
    )
    _CHUNK_GDR_ACCEPTS_QUERY_START_LOC_CPU = _kernel_accepts_kwarg(
        chunk_gated_delta_rule, "query_start_loc_cpu"
    )
except NameError:
    # CPU+CUDA-only build imports were skipped above; leave detection off.
    _CHUNK_GDR_ACCEPTS_PREBUILT_META = False
    _CHUNK_GDR_ACCEPTS_MAX_T = False
    _CHUNK_GDR_ACCEPTS_CU_SEQ_LEN = False
    _CHUNK_GDR_ACCEPTS_QUERY_START_LOC_CPU = False


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: fuse QKV extraction + gating + recurrent
        update into a single Triton kernel, eliminating intermediate tensors
        and extra kernel launches.

        Args:
            mixed_qkv: [B, qkv_dim] packed projection output after conv1d.
            a, b: [B, HV] gating inputs.
            A_log: [HV] log-space decay parameter.
            dt_bias: [HV] time-step bias.
            scale: attention scale factor (typically head_k_dim ** -0.5).
            ssm_states: [num_slots, HV, V, K] full state pool.
            cache_indices: [B] per-request state slot indices.
            num_v_heads: number of value heads (after TP sharding).
            head_v_dim: dimension per value head.

        Returns:
            output tensor of shape [1, B, HV, V] matching the existing
            decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # Packed kernel expects output shape [B, 1, HV, V]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

        # Convert [B, 1, HV, V] → [1, B, HV, V] to match existing output
        # layout. transpose() returns a view — zero cost.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        prebuilt_meta=None,
        **kwargs,
    ) -> tuple:
        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        if is_npu() or is_cpu():
            recurrent_state = ssm_states[cache_indices]
            recurrent_state_indices_args = {}
        extra_kwargs = {}
        if prebuilt_meta is not None:
            if _CHUNK_GDR_ACCEPTS_PREBUILT_META:
                extra_kwargs["prebuilt_meta"] = prebuilt_meta
            if _CHUNK_GDR_ACCEPTS_MAX_T:
                extra_kwargs["max_T"] = prebuilt_meta.max_T
            if _CHUNK_GDR_ACCEPTS_CU_SEQ_LEN:
                extra_kwargs["cu_seq_len"] = prebuilt_meta.cu_seq_len
            if _CHUNK_GDR_ACCEPTS_QUERY_START_LOC_CPU:
                extra_kwargs["query_start_loc_cpu"] = prebuilt_meta.query_start_loc_cpu
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **recurrent_state_indices_args,
            **extra_kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            # target_verify specific parameters
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
