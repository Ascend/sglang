# LLM 架构参考

## Decoder-only LLM 核心组件

```
Input tokens
  ↓ embedding
  ↓
  ├── for layer in num_layers:
  │     ├── input_layernorm  (RMSNorm / LayerNorm)
  │     ├── self_attention
  │     │     ├── q_proj, k_proj, v_proj (Linear, 可能带 bias)
  │     │     ├── rotary embedding (RoPE)
  │     │     ├── attention(q, k, v) → o
  │     │     ├── o_proj
  │     ├── residual add
  │     ├── post_attention_layernorm
  │     ├── mlp
  │     │     ├── gate_proj, up_proj (SwiGLU 等 gated 结构)
  │     │     ├── activation (SiLU / GELU)
  │     │     ├── down_proj
  │     ├── residual add
  ↓
  ↓ final_norm
  ↓ lm_head (有时与 embedding tie weights)
  → logits
```

## 关键 config 字段(适配时必须看)

| 字段 | 意义 | 影响适配 |
|------|------|---------|
| `architectures` | HF 架构名 | **决定参考哪个 SGLang 模型文件** |
| `hidden_size` | d_model | 影响 tp 分配是否整除 |
| `num_hidden_layers` | 层数 | 影响 pipeline parallel(目前 SGLang 默认 tp-only) |
| `num_attention_heads` | head 数 | 必须能被 tp_size 整除 |
| `num_key_value_heads` | KV head 数(GQA) | GQA: num_kv_heads < num_heads;MQA: num_kv_heads=1 |
| `intermediate_size` | MLP 中间维度 | 必须能被 tp_size 整除 |
| `vocab_size` | 词表大小 | 影响 lm_head 显存 |
| `rope_theta` | RoPE base(频率基) | 长上下文模型常用更大值(1e6) |
| `max_position_embeddings` | 最大上下文 | 决定 `--context-length` 上限 |
| `tie_word_embeddings` | embedding 与 lm_head 是否共享权重 | 影响 weight loading 逻辑 |
| `torch_dtype` | 默认精度 | bf16 在 NPU 上首选 |
| `rms_norm_eps` | RMSNorm epsilon | NPU 上数值敏感 |
| `attention_bias` | qkv 是否带 bias | Llama=False, Qwen=True |
| `hidden_act` | 激活函数 | "silu"(SwiGLU), "gelu" 等 |

## 常见架构家族

| 家族 | HF 架构 | 主要特征 |
|------|---------|--------|
| Llama | `LlamaForCausalLM` | RoPE + GQA + SwiGLU + RMSNorm, q/k/v 无 bias |
| Llama-3 | `LlamaForCausalLM` | 同上,但 rope_theta=500000 |
| Qwen2 | `Qwen2ForCausalLM` | Llama 类,但 q/k/v **有 bias** |
| Qwen2-MoE | `Qwen2MoeForCausalLM` | Qwen2 + MoE MLP |
| Qwen3 | `Qwen3ForCausalLM` | Qwen2 + 加 q_norm/k_norm |
| Mixtral | `MixtralForCausalLM` | Llama + MoE MLP(top-2 routing) |
| DeepSeek-V2/V3 | `DeepseekV2/V3ForCausalLM` | MLA(Multi-head Latent Attention)+ MoE + MTP |
| Mistral | `MistralForCausalLM` | Llama + sliding window |
| Phi | `PhiForCausalLM` / `Phi3ForCausalLM` | qkv_proj 合并 / SwiGLU |
| Gemma | `Gemma2ForCausalLM` | RoPE + sliding window 间隔 + 大 vocab |

## similarity 评级标准

| similarity | 标准 |
|------------|------|
| **high** | HF 架构名完全匹配 + 关键字段(num_heads / num_kv_heads / intermediate_size / rope_theta / hidden_act / attention_bias / norm_type)全部一致 |
| **medium** | HF 架构名同族(如都是 LlamaForCausalLM)但 1-3 处差异 |
| **low** | 无同族参考,或差异 >3 处,或有 MoE/MLA/Sliding-Window 等需要不同实现路径的结构差异 |

## 决策示例

| 目标模型 | 参考实现 | similarity | adapter_strategy |
|---------|---------|-----------|-----------------|
| Qwen2.5-7B | `python/sglang/srt/models/qwen2.py` | high | direct_use |
| Qwen2.5-VL | `python/sglang/srt/models/qwen2_vl.py` | high | direct_use |
| 某新版 Qwen3(加了 q_norm) | `python/sglang/srt/models/qwen3.py`(若已存在) | high | direct_use; 否则 medium → extend_existing |
| 全新 architecture(自创 attention 变体) | 无 | low | new_implementation |

## tp 整除性检查

```
num_attention_heads % tp_size == 0       # 必须
num_key_value_heads % tp_size == 0       # 必须(GQA 时;若 num_kv_heads < tp_size 需 KV 复制)
intermediate_size % tp_size == 0         # 必须
hidden_size % tp_size == 0               # 必须
vocab_size 可以不整除(用 pad)
```

不满足时,要么换 tp_size,要么报 `requires_new_adapter=true` 加 padding 逻辑。
