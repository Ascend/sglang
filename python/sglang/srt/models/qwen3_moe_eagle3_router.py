"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import copy
from sglang.srt.utils import add_prefix
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from transformers.activations import ACT2FN



# Qwen3-MoE configuration class (matches the actually deployed config.json, fixes the default value of draft_vocab_size)
class Qwen3MoEConfig(PretrainedConfig):
    model_type = "qwen3_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size=151936,
        draft_vocab_size=32000,  # Aligned with 32000 in the actual config file
        hidden_size=2048,
        intermediate_size=12288,
        moe_intermediate_size=768,  # Aligned with 768 in the actual config file
        num_hidden_layers=1,  # Aligned with 1 layer in the actual config file
        num_attention_heads=32,  # Aligned with 32 heads in the actual config file
        num_key_value_heads=4,  # Aligned with 4 heads in the actual config file
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,  # Aligned with 1000000.0 in the actual config file
        attention_bias=False,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        num_experts_per_tok=8,  # Aligned with 8 in the actual config file
        num_experts=128,  # Aligned with 128 in the actual config file
        norm_topk_prob=False,  # Aligned with false in the actual config file
        mlp_only_layers=None,
        rope_scaling=None,
        head_dim=128,  # Aligned with 128 in the actual config file
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=48,
        transformers_version="4.57.1",
        dtype="bfloat16",
        bos_token_id=151643,
        eos_token_id=151645,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings,** kwargs)
        # Basic configuration
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.decoder_sparse_step = decoder_sparse_step
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim if head_dim else hidden_size // num_attention_heads
        
        # Extended configuration (aligned with actual config.json)
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.transformers_version = transformers_version
        self.dtype = dtype
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


# Decoder layer: Core fixes - embeds parameter input + NPU global alignment + intra-layer dual input concatenation + EAGLE3 + MoE
class Qwen3MoEDecoderLayer(Qwen2DecoderLayer):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size
        
        # Replace with MoE layer to implement Qwen3-MoE core features
        self.mlp = Qwen3MoeSparseMoeBlock(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Core fixes: Add embeds parameter input + NPU global device/type alignment + intra-layer concatenation logic
    def forward(
        self,
        positions: torch.Tensor,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NPU global alignment: unified device and data type

        residual = hidden_states
        input_emb = self.input_layernorm(input_emb)
        hidden_states = self.hidden_norm(hidden_states)
        
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        # Self Attention: input is concatenated 2*hidden_size, matching QKV layer input dimension
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states, _ = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter,
        )
        return x


# Qwen3Moe Sparse MoE Block
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        ):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp", prefix),
                    )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            num_tokens, hidden_dim
        )
        return final_hidden_states, router_logits



# Qwen3-MoE EAGLE3 model core: Fix embeds parameter passing + NPU adaptation + full logic alignment
class Qwen3MoEModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoEConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.is_mrope_enabled = (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and "mrope_section" in config.rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            config.rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )
        
        # MoE decoder layer list
        self.midlayer =  Qwen3MoEDecoderLayer(
            config=config,
            layer_id=0,
            quant_config=quant_config,
            prefix=prefix,
        )
        
        # Output normalization layer
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
            # print(f"398 hidden_statesfc {hidden_states[0][0]} ")

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]



# Top-level model: EAGLE3 framework interface + weight loading + NPU input alignment
class Qwen3MoERouterForCausalLMEagle3(LlamaForCausalLMEagle3):
    # EAGLE3 weight mapping table
    def __init__(
        self,
        config: Qwen3MoEConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        self.model = Qwen3MoEModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = (
            config_.draft_vocab_size
        )  # draft logits processor has it's own vocab size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


# SGLang framework entry class: Must be defined
EntryClass = [Qwen3MoERouterForCausalLMEagle3]