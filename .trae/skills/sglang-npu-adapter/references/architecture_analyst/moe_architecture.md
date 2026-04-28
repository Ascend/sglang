# MoE架构识别知识

## 1. MoE核心概念

### 1.1 什么是MoE

Mixture of Experts (MoE) 是一种稀疏激活的模型架构：
- **核心思想**：每个token只激活部分专家，而非全部
- **优势**：在增加参数量的同时保持计算效率
- **代表模型**：Mixtral、Qwen2-MoE、DeepSeek-V2/V3

### 1.2 MoE vs Dense 对比

| 特性 | Dense模型 | MoE模型 |
|------|----------|---------|
| 参数激活 | 全部激活 | 部分激活 |
| 计算量 | 与参数量成正比 | 与激活专家数成正比 |
| 参数效率 | 较低 | 较高 |
| 内存占用 | 较小 | 较大（需存储所有专家） |

---

## 2. MoE架构组件

### 2.1 专家(Expert)

每个专家是一个独立的MLP网络：
```
Expert = gate_proj + up_proj + down_proj
```
- 与标准MLP结构相同
- 但每个专家有独立的参数
- 专家数量通常为8-256个

### 2.2 路由器(Router/Gate)

决定每个token应该由哪些专家处理：
```
router_logits = gate(hidden_states)  # [batch, seq, num_experts]
```
- 输出每个专家的得分
- 通过Top-K选择激活的专家

### 2.3 Top-K选择

选择得分最高的K个专家：
```
topk_weights, topk_indices = torch.topk(router_logits, k)
```
- K通常为2-8
- 权重通常需要归一化（softmax）

---

## 3. MoE变体识别

### 3.1 纯MoE

所有Decoder Layer都使用MoE：
- 代表：Mixtral 8x7B
- 特点：每层都有专家路由

### 3.2 混合MoE

部分层使用MoE，部分层使用Dense：
- 代表：某些定制模型
- 需要检查每层的配置

### 3.3 共享专家(Shared Expert)

除了路由专家外，还有一个共享专家处理所有token：
```
output = routed_experts_output + shared_expert_output
```
- 代表：Qwen2-MoE、DeepSeek-V2/V3
- 优势：提高模型表达能力

### 3.4 共享专家门控

共享专家的输出可以通过门控调节：
```
shared_output = sigmoid(shared_expert_gate(x)) * shared_expert(x)
```
- 代表：Qwen2-MoE

---

## 4. MoE并行策略

### 4.1 Tensor Parallel (TP)

将每个专家的权重切分到多个设备：
```
专家权重: [hidden_size, intermediate_size]
    ↓ TP=2
设备0: [hidden_size, intermediate_size/2]
设备1: [hidden_size, intermediate_size/2]
```

**限制**：
- TP大小不能超过专家的intermediate_size
- 所有设备都需要存储所有专家的部分权重

### 4.2 Expert Parallel (EP)

将完整的专家分配到不同设备：
```
num_experts = 8, EP = 2
设备0: 专家 0-3
设备1: 专家 4-7
```

**优势**：
- 每个设备只存储部分专家
- 支持更大的专家数量
- 减少单设备内存压力

**通信模式**：
- All-to-All通信：将token发送到对应专家所在的设备
- 计算完成后，再将结果发回原设备

### 4.3 TP + EP 混合并行

结合两种策略：
```
总设备数 = TP × EP
每个设备存储: (num_experts / EP) 个专家的 (1/TP) 权重
```

---

## 5. DeepEP详解

### 5.1 什么是DeepEP

DeepEP是SGLang中用于MoE专家并行的高性能通信库，专门优化了All-to-All通信：

**核心功能**：
- 高效的token分发(dispatch)和收集(combine)
- 支持NVLink和InfiniBand
- 支持FP8/BF16精度
- 支持对称内存(symm-mem)优化

### 5.2 DeepEP通信流程

```
┌─────────────────────────────────────────────────────────────┐
│                     DeepEP通信流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Dispatch阶段（token分发）                                │
│     ┌─────────┐    All-to-All    ┌─────────┐               │
│     │ Device0 │ ───────────────→ │ Device0 │               │
│     │ token_a │                  │ 专家0-3 │               │
│     └─────────┘                  └─────────┘               │
│     ┌─────────┐    All-to-All    ┌─────────┐               │
│     │ Device1 │ ───────────────→ │ Device1 │               │
│     │ token_b │                  │ 专家4-7 │               │
│     └─────────┘                  └─────────┘               │
│                                                             │
│  2. Expert计算                                               │
│     每个设备计算自己负责的专家                                 │
│                                                             │
│  3. Combine阶段（结果收集）                                   │
│     将计算结果发回原设备                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 DeepEP配置

**启动参数**：
```bash
python -m sglang.launch_server \
    --model-path /path/to/moe_model \
    --ep-size 4 \              # Expert Parallel大小
    --moe-a2a-backend deepep   # 使用DeepEP后端
```

**代码配置**：
```python
from sglang.srt.layers.moe import get_moe_a2a_backend

moe_backend = get_moe_a2a_backend()
if moe_backend.is_deepep():
    # 使用DeepEP路径
    ...
```

### 5.4 DeepEP性能优化

| 优化项 | 说明 |
|--------|------|
| **对称内存** | `--enable-symm-mem` 预分配通信缓冲区 |
| **FP8通信** | 减少通信数据量 |
| **NVLink优化** | 利用NVLink高带宽 |
| **流水线重叠** | 通信与计算重叠 |

### 5.5 DeepEP vs 标准EP

| 特性 | 标准EP | DeepEP |
|------|--------|--------|
| 通信库 | NCCL/HCCL | 定制优化 |
| 延迟 | 较高 | 较低 |
| 带宽利用率 | 一般 | 高 |
| 支持精度 | FP16/BF16 | FP8/BF16/FP16 |
| 对称内存 | 不支持 | 支持 |

---

## 6. SGLang MoE实现

### 6.1 SparseMoeBlock

SGLang的MoE核心组件，参考实现：`python/sglang/srt/models/qwen2_moe.py` 中的 `Qwen2MoeSparseMoeBlock`

### 6.2 FusedMoE

高性能融合MoE实现，参考实现：`python/sglang/srt/layers/moe/fused_moe_triton.py`

### 6.3 TopK路由

参考实现：`python/sglang/srt/layers/moe/topk.py`

### 6.4 DeepEP后端

参考实现：`python/sglang/srt/layers/moe/ep_moe/`

---

## 7. 配置文件关键字段

### 7.1 MoE必要字段

| 字段 | 说明 | 示例值 |
|------|------|--------|
| num_experts | 专家总数 | 64 |
| num_experts_per_tok | 每个token激活的专家数 | 8 |
| moe_intermediate_size | 专家的intermediate_size | 1408 |

### 7.2 共享专家字段

| 字段 | 说明 | 示例值 |
|------|------|--------|
| shared_expert_intermediate_size | 共享专家的intermediate_size | 4096 |

### 7.3 路由配置字段

| 字段 | 说明 | 示例值 |
|------|------|--------|
| norm_topk_prob | 是否归一化Top-K权重 | true |
| router_aux_loss_coef | 路由辅助损失系数 | 0.02 |

---

## 8. 权重加载特殊处理

### 8.1 专家权重映射

MoE模型的专家权重需要特殊处理：

```
HuggingFace格式:
  model.layers.N.mlp.experts.0.gate_proj.weight
  model.layers.N.mlp.experts.0.up_proj.weight
  model.layers.N.mlp.experts.0.down_proj.weight

SGLang格式:
  model.layers.N.mlp.experts.0.gate_up_proj.weight (merged)
  model.layers.N.mlp.experts.0.down_proj.weight
```

### 8.2 expert_params_mapping

参考实现：`python/sglang/srt/models/qwen2_moe.py` 中的 `load_weights` 方法

---

## 9. MoE内存与计算分析

### 9.1 参数量计算

```
总参数量 = Embedding + N × (Attention + MoE) + LM_Head

MoE参数量 = num_experts × (gate_proj + up_proj + down_proj)
         + shared_expert参数量（如有）
         + gate参数量
```

### 9.2 计算量计算

```
每个token的计算量 ≈ num_experts_per_tok × 单专家计算量
```

### 9.3 内存优化策略

| 策略 | 说明 |
|------|------|
| Expert Parallel | 将专家分布到多个设备 |
| DeepEP | 高效的EP通信优化 |
| 权重量化 | 对专家权重量化 |
| KV Cache优化 | 减少KV Cache占用 |

---

## 10. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/models/qwen2_moe.py` | Qwen2-MoE完整实现 |
| `python/sglang/srt/models/mixtral.py` | Mixtral实现 |
| `python/sglang/srt/models/deepseek_v2.py` | DeepSeek MoE实现 |
| `python/sglang/srt/layers/moe/` | MoE层实现目录 |
| `python/sglang/srt/layers/moe/ep_moe/` | DeepEP实现目录 |
