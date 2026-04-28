# MLA架构识别知识

## 1. MLA概述

Multi-Head Latent Attention (MLA) 是DeepSeek提出的创新注意力机制，通过KV压缩大幅减少内存占用。

### 1.1 MLA vs 标准Attention

| 特性 | 标准Attention | MLA |
|------|--------------|-----|
| KV Cache | 原始维度 | 压缩维度 |
| 内存占用 | 较大 | 较小（减少90%+） |
| 计算复杂度 | 标准压缩/解压缩 | 略高 |
| 代表模型 | Llama, Qwen | DeepSeek-V2/V3 |

---

## 2. MLA核心原理

### 2.1 KV压缩

将KV压缩到低维潜在空间：
```
KV压缩: hidden_states → [kv_lora_rank] → 缓存
KV解压缩: [kv_lora_rank] → [num_heads, head_dim]
```

### 2.2 Query压缩

Query也可以压缩：
```
Query压缩: hidden_states → [q_lora_rank] → [num_heads, head_dim]
```

### 2.3 RoPE处理

MLA中RoPE只应用于部分维度：
```
qk_rope_head_dim: RoPE应用的维度
```

---

## 3. MLA配置字段

| 字段 | 说明 | 示例值 |
|------|------|--------|
| kv_lora_rank | KV压缩维度 | 512 |
| q_lora_rank | Query压缩维度 | 1536 |
| qk_rope_head_dim | RoPE维度 | 64 |
| v_head_dim | Value头维度 | 128 |

---

## 4. MLA内存计算

### 4.1 KV Cache大小对比

**标准Attention**：
```
KV Cache = 2 × batch × num_heads × seq_len × head_dim × dtype_size
```

**MLA**：
```
KV Cache = batch × seq_len × kv_lora_rank × dtype_size
```

### 4.2 示例计算

DeepSeek-V2配置：
- num_heads = 128, head_dim = 128
- kv_lora_rank = 512

标准Attention KV Cache: 2 × 128 × 128 = 32768 维
MLA KV Cache: 512 维

**压缩比**: 32768 / 512 = 64x

---

## 5. MLA适配要点

### 5.1 检查MLA配置

```python
def is_mla_model(config):
    return hasattr(config, 'kv_lora_rank') and config.kv_lora_rank is not None
```

### 5.2 MLA Attention实现

需要实现压缩和解压缩层，参考实现：`python/sglang/srt/models/deepseek_v2.py`

### 5.3 权重加载

MLA模型的权重结构不同，需要特殊处理。

---

## 6. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/models/deepseek_v2.py` | DeepSeek-V2/V3 MLA实现 |
| `python/sglang/srt/models/deepseek.py` | DeepSeek-V1实现 |
