# MoE相关问题

## 1. 专家路由问题

### 1.1 Top-K选择错误

**错误现象**：
```
RuntimeError: Invalid expert indices
```

**原因**：Top-K选择逻辑错误

**排查**：
1. 检查num_experts配置
2. 检查num_experts_per_tok配置
3. 检查TopK实现

### 1.2 路由权重归一化

**问题**：权重未正确归一化导致输出异常

**检查**：
```python
# 检查norm_topk_prob配置
norm_topk_prob = getattr(config, "norm_topk_prob", True)
```

### 1.3 专家负载不均衡

**现象**：某些专家使用率过高，其他专家闲置

**原因**：路由策略问题

**解决方案**：
- 检查router_aux_loss_coef配置
- 考虑负载均衡策略

---

## 2. 专家并行问题

### 2.1 EP配置错误

**错误现象**：
```
RuntimeError: EP size must divide num_experts
```

**原因**：EP大小与专家数量不匹配

**解决方案**：
```python
# 必须满足
assert num_experts % ep_size == 0
```

### 2.2 专家权重sharding

**问题**：专家权重分配不正确

**检查**：
1. 确认每个rank的专家数量
2. 检查权重加载逻辑

### 2.3 DeepEP通信问题

**错误现象**：
```
RuntimeError: DeepEP communication failed
```

**排查**：
1. 检查HCCL是否正确安装
2. 检查网络配置
3. 检查NPU互联

---

## 3. 共享专家问题

### 3.1 shared_expert配置

**问题**：共享专家未正确加载

**检查**：
```python
shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", 0)
if shared_expert_intermediate_size > 0:
    # 需要加载共享专家
    ...
```

### 3.2 shared_expert_gate配置

**问题**：门控权重未正确加载

**检查**：
1. 确认shared_expert_gate权重存在
2. 检查权重名称映射

---

## 4. 权重加载问题

### 4.1 专家权重名称映射

**问题**：专家权重名称不匹配

**解决方案**：
```python
expert_params_mapping = FusedMoE.make_expert_params_mapping(
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=config.num_experts,
)
```

### 4.2 专家权重加载失败

**错误现象**：
```
WARNING: Parameter mlp.experts.0.xxx not found
```

**排查**：
1. 检查权重文件结构
2. 检查expert_params_mapping配置

---

## 5. NPU MoE特定问题

### 5.1 DeepEP后端兼容性

**检查**：
```bash
# 检查HCCL支持
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"
```

### 5.2 MoE算子支持

| 算子 | NPU支持状态 |
|------|-------------|
| FusedMoE | ✅ 支持 |
| TopK | ✅ 支持 |
| DeepEP | ⚠️ 需检查HCCL |

---

## 6. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/models/qwen2_moe.py` | Qwen2-MoE实现 |
| `python/sglang/srt/layers/moe/` | MoE层实现 |
| `python/sglang/srt/layers/moe/ep_moe/` | DeepEP实现 |
