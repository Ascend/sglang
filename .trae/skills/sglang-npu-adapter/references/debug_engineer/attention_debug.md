# Attention相关问题

## 1. NPU Attention后端

### 1.1 Ascend后端配置

NPU使用 `ascend` 作为Attention后端：

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --attention-backend ascend \
    --device npu
```

### 1.2 Ascend后端初始化失败排查

**错误现象**：
```
RuntimeError: Failed to initialize ascend attention backend
```

**排查步骤**：
1. 检查CANN版本是否兼容
2. 检查NPU驱动是否正常
3. 确认torch_npu正确安装

**检查命令**：
```bash
# 检查NPU设备
npu-smi info

# 检查torch_npu
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

### 1.3 NPU Attention算子限制

| 算子 | 支持状态 | 备注 |
|------|----------|------|
| FlashAttention | ✅ 支持 | 通过ascend后端 |
| PagedAttention | ✅ 支持 | KV Cache管理 |
| 滑动窗口 | ⚠️ 部分支持 | 需检查具体配置 |
| ALiBi | ⚠️ 部分支持 | 需检查具体配置 |

---

## 2. GQA相关问题

### 2.1 num_heads与num_kv_heads的关系

**GQA核心概念**：
```
num_heads: Query头数
num_kv_heads: Key/Value头数
num_heads / num_kv_heads = 每组KV对应的Query头数
```

**常见配置**：
| 模型 | num_heads | num_kv_heads | 分组比 |
|------|-----------|--------------|--------|
| Llama2-70B | 64 | 8 | 8:1 |
| Qwen2-7B | 28 | 4 | 7:1 |
| Mistral-7B | 32 | 8 | 4:1 |

### 2.2 TP分片后的head分配

**问题**：TP分片时，每个rank的head数必须是整数

**检查条件**：
```python
# 必须满足
assert num_heads % tp_size == 0
assert num_kv_heads % tp_size == 0 or tp_size % num_kv_heads == 0
```

**常见错误**：
```
AssertionError: num_kv_heads must be divisible by tp_size
```

**解决方案**：
- 调整TP大小
- 或使用DP Attention

### 2.3 GQA配置错误案例

**错误现象**：输出完全错误，但无报错

**原因**：`num_kv_heads` 配置错误

**排查方法**：
1. 检查config中的 `num_key_value_heads`
2. 确认与模型权重匹配
3. 检查QKVParallelLinear参数

---

## 3. MLA架构问题

### 3.1 MLA vs GQA

| 特性 | GQA | MLA |
|------|-----|-----|
| KV Cache | 原始维度 | 压缩维度 |
| 内存占用 | 较大 | 较小 |
| 实现复杂度 | 简单 | 复杂 |
| 代表模型 | Llama, Qwen | DeepSeek-V2/V3 |

### 3.2 MLA配置字段

| 字段 | 说明 |
|------|------|
| kv_lora_rank | KV压缩维度 |
| q_lora_rank | Query压缩维度 |
| qk_rope_head_dim | RoPE维度 |

### 3.3 MLA适配注意事项

1. 需要实现压缩/解压缩层
2. RoPE处理方式不同
3. 参考实现：`python/sglang/srt/models/deepseek_v2.py`

---

## 4. Sliding Window Attention

### 4.1 概念

限制每个位置只能看到最近的W个位置：
```
attention_mask[i, j] = 1 if |i - j| <= window_size else 0
```

### 4.2 SGLang支持

通过配置启用：
```python
# 在config中
sliding_window = 4096
```

### 4.3 常见问题

**问题**：启用sliding window后输出错误

**排查**：
1. 确认RadixAttention支持sliding window
2. 检查window_size配置是否正确

---

## 5. ALiBi位置编码

### 5.1 概念

Attention with Linear Biases，不使用RoPE，而是通过偏置编码位置：
```
attention_score += -m * |i - j|
```

### 5.2 与RoPE的区别

| 特性 | RoPE | ALiBi |
|------|------|-------|
| 位置信息 | 通过旋转编码 | 通过偏置编码 |
| 外推能力 | 需要scaling | 天然支持 |
| 实现复杂度 | 较高 | 较低 |

### 5.3 实现注意事项

ALiBi模型不需要RoPE，在Attention初始化时：
```python
# ALiBi模型
if has_alibi:
    self.rotary_emb = None  # 不使用RoPE
    # ALiBi偏置在RadixAttention中处理
```

---

## 6. KV Cache相关问题

### 6.1 KV Cache大小计算

```
KV Cache大小 = 2 × batch × num_kv_heads × seq_len × head_dim × dtype_size
```

**示例**（Qwen2-7B, BF16）：
```
batch=16, seq_len=4096, num_kv_heads=4, head_dim=128
KV Cache = 2 × 16 × 4 × 4096 × 128 × 2 = 1GB
```

### 6.2 KV Cache OOM

**解决方案**：
| 策略 | 命令 | 效果 |
|------|------|------|
| 减小batch | `--max-running-requests 8` | 减少并发 |
| 减小序列长度 | `--context-length 4096` | 减少KV Cache |
| KV Cache量化 | `--kv-cache-dtype fp8` | 减少内存 |

### 6.3 KV Cache形状不匹配

**错误现象**：
```
RuntimeError: shape '[16, 4, 1024, 128]' is invalid for input of size ...
```

**原因**：num_kv_heads或head_dim配置错误

**排查**：检查Attention初始化参数

---

## 7. 常见错误案例

### 案例1：GQA配置错误导致输出错误

**现象**：模型输出完全乱码，无报错

**原因**：`num_key_value_heads` 从config读取失败，使用了默认值

**修复**：
```python
num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
```

### 案例2：NPU Attention后端初始化失败

**现象**：
```
RuntimeError: Failed to initialize ascend backend
```

**原因**：CANN版本不兼容

**修复**：
1. 检查CANN版本
2. 更新torch_npu
3. 检查NPU驱动

### 案例3：KV Cache大小不匹配

**现象**：
```
RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
```

**原因**：TP配置与权重不匹配

**修复**：检查TP配置，确保与预训练权重一致

---

## 8. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/layers/radix_attention.py` | RadixAttention实现 |
| `python/sglang/srt/layers/linear.py` | QKVParallelLinear |
| `python/sglang/srt/models/qwen2.py` | GQA示例 |
| `python/sglang/srt/models/deepseek_v2.py` | MLA示例 |
| `python/sglang/srt/hardware_backend/npu/` | NPU后端实现 |
