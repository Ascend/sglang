# MoE 架构参考

## MoE 与稠密的差异

```
Dense MLP:
    x → gate_proj, up_proj → SiLU(gate) * up → down_proj → out

MoE MLP:
    x → router(linear) → top-k experts 选择 + softmax 权重
       ↓
    parallel: for each selected expert:
        x → expert.gate_proj, up_proj → SiLU(gate) * up → expert.down_proj
       ↓
    weighted sum → out
```

## 关键 config 字段

| 字段 | 意义 |
|------|------|
| `num_experts` / `num_local_experts` | 专家数 |
| `num_experts_per_tok` / `num_routed_experts` | top-k |
| `moe_intermediate_size` | 每个 expert 的中间维度 |
| `n_shared_experts` | 共享专家数(DeepSeek 风格) |
| `n_routed_experts` | 路由专家数 |
| `router_aux_loss_coef` | 训练时用,推理可忽略 |
| `norm_topk_prob` | top-k 权重是否做 softmax 归一化 |

## SGLang MoE 已有支持(检查时 grep `class.*MoE` in `python/sglang/srt/models/`)

| 家族 | HF 架构 | SGLang 文件 |
|------|---------|------------|
| Mixtral | `MixtralForCausalLM` | `mixtral.py` |
| Qwen2-MoE | `Qwen2MoeForCausalLM` | `qwen2_moe.py` |
| DeepSeek-V2 | `DeepseekV2ForCausalLM` | `deepseek_v2.py` |
| DeepSeek-V3 | `DeepseekV3ForCausalLM` | `deepseek_v3.py`(若存在) |

## 并行策略

MoE 的并行比 dense 复杂一些:

| 维度 | 含义 |
|------|------|
| `tp_size` | tensor parallel(切 attention 头 / MLP 维度) |
| `ep_size` | expert parallel(切 expert 数;每张卡只放部分 expert) |
| `dp_size` | data parallel(常与 dp_attention 配合) |

**典型配置**:
- 7B MoE(类 Qwen2-MoE-14B-A2B):`tp=8 ep=1 dp=1`
- 大型 MoE(DeepSeek-V3 671B):`tp=8 ep=8 dp=1`,跨机
- 加 DP attention(注意:不是所有 MoE 模型都支持):`tp=8 dp=2 ep=2`

## DeepEP 适用范围

DeepEP 是 NPU 上的 expert parallel 优化通信库。**仅 MoE 模型适用**,且需要:
- 模型架构支持(参考 deepseek_v3 的实现)
- `ep_size > 1`

非 MoE 模型直接标 `deepep: na`。

## NPU 上 MoE 已知坑

1. **Token 分发**:NPU 上 all-to-all 通信比 GPU 慢,建议小 batch 用 broadcast 替代
2. **专家不均衡**:某些 expert 永远不被路由(冷启动期),不一定是 bug,跑一段时间会均衡
3. **fused MoE kernel**:NPU 上通常没有 fused 实现,SGLang 默认拆分为多个 GEMM(性能损失但功能正确)
