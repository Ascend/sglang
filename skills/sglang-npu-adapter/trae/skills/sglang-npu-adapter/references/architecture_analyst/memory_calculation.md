# 内存计算方法

## 1. 模型权重内存

### 1.1 参数量计算

```
总参数量 = Embedding + N × (Attention + MLP) + LM_Head

Embedding = vocab_size × hidden_size
Attention = 4 × hidden_size × hidden_size (QKV + O)
MLP = 3 × hidden_size × intermediate_size (gate, up, down)
LM_Head = vocab_size × hidden_size
```

### 1.2 权重内存计算

```
权重内存 = 参数量 × dtype_size

dtype_size:
- FP32: 4 bytes
- FP16/BF16: 2 bytes
- FP8: 1 byte
- INT8: 1 byte
```

### 1.3 TP分片后每卡内存

```
每卡权重内存 = 总权重内存 / TP_size
```

---

## 2. KV Cache内存

### 2.1 KV Cache大小公式

```
KV Cache = 2 × batch × num_kv_heads × seq_len × head_dim × dtype_size
```

### 2.2 MLA KV Cache

```
KV Cache = batch × seq_len × kv_lora_rank × dtype_size
```

### 2.3 影响因素

| 因素 | 影响 |
|------|------|
| batch_size | 线性增加 |
| seq_len | 线性增加 |
| num_kv_heads | 线性增加 |
| head_dim | 线性增加 |
| dtype | FP16比FP32减半 |

---

## 3. 激活内存

### 3.1 激活内存估算

```
激活内存 ≈ batch × seq_len × hidden_size × num_layers × activation_factor
```

### 3.2 激活重计算

启用激活重计算可以减少激活内存，但增加计算量：
```
--enable-flashinfer --flashinfer-reduce-buffer-size
```

---

## 4. 总内存估算

### 4.1 公式

```
总内存 = 权重内存 + KV Cache内存 + 激活内存 + 开销
开销 ≈ 10-20% 的总内存
```

### 4.2 示例计算

Qwen2-7B (BF16, TP=1):
```
参数量 ≈ 7B
权重内存 = 7B × 2 = 14GB

batch=16, seq_len=4096, num_kv_heads=4, head_dim=128
KV Cache = 2 × 16 × 4 × 4096 × 128 × 2 = 1GB

激活内存 ≈ 2GB
开销 ≈ 2GB

总内存 ≈ 14 + 1 + 2 + 2 = 19GB
```

---

## 5. 并行配置建议

### 5.1 TP选择

| 模型大小 | 建议TP | 说明 |
|----------|--------|------|
| 7B | 1-2 | 单卡或双卡 |
| 13B | 2-4 | 双卡或四卡 |
| 70B | 4-8 | 四卡或八卡 |

### 5.2 Batch Size选择

根据可用内存动态调整：
```
max_batch = (可用内存 - 权重内存) / (每token KV Cache × seq_len)
```

### 5.3 序列长度限制

```
max_seq_len = (可用内存 - 权重内存) / (KV Cache per token × batch)
```

---

## 6. NPU内存特性

### 6.1 NPU内存限制

| NPU型号 | HBM大小 | 可用内存 |
|---------|---------|----------|
| Ascend 910B1 | 64GB | ~50GB |
| Ascend 910B2 | 64GB | ~50GB |

### 6.2 NPU内存优化

- 使用量化减少权重内存
- 使用KV Cache量化
- 减小batch size
- 减小序列长度
