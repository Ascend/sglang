# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from functools import partial
from typing import Callable, Optional

import torch

from sglang.srt.layers.attention.nsa.utils import (
    cp_extract_local_tokens,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
    use_pcp,
)
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_attention_cp_group,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_tp_group,
    get_local_dp_buffer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import get_current_device_stream_fast


def nsa_enable_prefill_cp():
    # After using cp, the communication mode of this part changes.
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer
    # no longer require additional communication for reduce, scatter, etc.
    return is_nsa_enable_prefill_cp()


class NSACPLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
    ):
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )

    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert (
                self._context.attn_dp_size == 1
            ), f"dp_size should be 1 when moe_runner_backend is none"
        self._communicate_simple_fn = NSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = NSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_summable_tensor_pair_fn = NSACPCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )


class NSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return NSACPCommunicateSimpleFn._trivial

        raise NotImplementedError(f"{input_mode=} {output_mode=}")


class NSACPCommunicateWithAllReduceAndLayerNormFn(
    CommunicateWithAllReduceAndLayerNormFn
):
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        assert hidden_states_input_mode == ScatterMode.SCATTERED
        assert residual_input_mode == ScatterMode.SCATTERED
        assert residual_output_mode == ScatterMode.SCATTERED
        if hidden_states_output_mode == ScatterMode.SCATTERED:
            return NSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                NSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        # for prefill: attn tp scattered -> full
        # for decode: attn tp full -> full
        if nsa_use_prefill_cp(forward_batch):
            if hidden_states.shape[0] != 0:
                hidden_states, residual = layernorm(hidden_states, residual)
            assert context.attn_dp_size == 1
            hidden_states, local_hidden_states = (
                get_local_dp_buffer(),
                hidden_states,
            )
            attn_cp_all_gather_into_tensor(
                hidden_states,
                local_hidden_states,
            )
            return hidden_states, residual
        elif use_pcp(forward_batch) and forward_batch.cp_metadata.is_gqa:
            cp_metadata = forward_batch.cp_metadata
            assert cp_metadata is not None
            pcp_size = get_attention_cp_size()
            max_len = (
                cp_metadata.max_rank_len[0]
                if cp_metadata.max_rank_len is not None
                else hidden_states.shape[0]
            )
            if hidden_states.shape[0] != 0:
                hidden_states = get_attention_tp_group().all_reduce(hidden_states)
                hidden_states, residual = layernorm(hidden_states, residual)
            local_len = hidden_states.shape[0]
            local_hidden_states = hidden_states.new_zeros(
                (max_len, hidden_states.shape[1]), dtype=hidden_states.dtype
            )
            if local_len > 0:
                local_hidden_states[:local_len].copy_(hidden_states)
            gathered_hidden_states = torch.empty(
                pcp_size,
                max_len,
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            get_attention_cp_group().cp_all_gather_into_tensor_async(
                gathered_hidden_states,
                local_hidden_states,
                get_current_device_stream_fast(),
            )
            reshaped_hidden_states = gathered_hidden_states.reshape(
                -1, gathered_hidden_states.shape[-1]
            )
            if (
                cp_metadata.per_rank_actual_token is not None
                and len(cp_metadata.per_rank_actual_token) == pcp_size
            ):
                gathered_parts = []
                for rank_idx, per_rank_len in enumerate(
                    cp_metadata.per_rank_actual_token
                ):
                    if per_rank_len <= 0:
                        continue
                    start = rank_idx * max_len
                    gathered_parts.append(
                        reshaped_hidden_states[start : start + per_rank_len]
                    )
                if gathered_parts:
                    reshaped_hidden_states = torch.cat(gathered_parts, dim=0)
                else:
                    reshaped_hidden_states = reshaped_hidden_states[:0]
            return reshaped_hidden_states, residual
        else:
            if hidden_states.shape[0] != 0:
                hidden_states, residual = layernorm(hidden_states, residual)
            return hidden_states, residual


class NSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return NSACPCommunicateSummableTensorPairFn._trivial

        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return NSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        layer_norm: Optional[torch.nn.Module] = None,
        allow_reduce_scatter: bool = False,
    ):
        # for prefill: full -> attn tp scattered
        # for decode: full -> attn tp full
        if nsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            input_hidden_states = hidden_states
            hidden_states = hidden_states.tensor_split(context.attn_cp_size)[
                context.attn_cp_rank
            ]
            attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
            return hidden_states, residual
        elif use_pcp(forward_batch):
            if hidden_states.shape[0] != 0:
                if not forward_batch.cp_metadata.is_gqa:
                    hidden_states = get_attention_tp_group().all_reduce(hidden_states)
                    hidden_states, residual = layer_norm(hidden_states, residual)
                else:
                    pcp_size = get_attention_cp_size()
                    pcp_rank = get_attention_cp_rank()
                    cp_metadata = forward_batch.cp_metadata
                    if (
                        cp_metadata is not None
                        and cp_metadata.per_rank_actual_token is not None
                        and len(cp_metadata.per_rank_actual_token) == pcp_size
                    ):
                        start = sum(cp_metadata.per_rank_actual_token[:pcp_rank])
                        end = start + cp_metadata.per_rank_actual_token[pcp_rank]
                        hidden_states = hidden_states[start:end].contiguous()
                    else:
                        hidden_states = hidden_states.tensor_split(pcp_size)[pcp_rank]
                    if residual is not None and cp_metadata is not None:
                        residual = cp_extract_local_tokens(forward_batch, residual)
            return hidden_states, residual
        else:
            return hidden_states, residual
