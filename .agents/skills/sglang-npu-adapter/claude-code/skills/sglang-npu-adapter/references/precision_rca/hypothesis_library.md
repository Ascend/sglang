# Hypothesis Library

按 first_bad_module 类型查实验序列。每个实验有明确的:
- hypothesis(假设是什么)
- command / code(怎么跑)
- pass criteria(verdict 判据)

## verdict 判据(共用)

跑完实验后比对 drift_after 与 drift_before:
- `supports`:drift 下降 ≥ 80%
- `rejects`:drift 下降 < 20%
- `inconclusive`:中间值

每个实验记录到 `precision/experiments/exp_NNN_<name>.json`。

## Linear / Linear-like(q_proj / k_proj / v_proj / o_proj / gate / up / down)

| ID | 实验 | category | 验证假设 | 实施要点 |
|---|---|---|---|---|
| exp_001 | 强制 fp32 cast 输入/输出 | dtype | dtype 截断 / 累积误差 | hook 该模块 forward,在 input 处 `.float()`,output 处 `.bfloat16()` 还原 |
| exp_002 | 比对 weight tensor (SGLang vs HF) | weight_loading | 权重加载错(shard split / layout / scale 维度) | 直接 `torch.allclose(sgl.weight, hf.weight)`;有 quant 时 dequant 后比 |
| exp_003 | force quant=None 这一层 | quantization | W8A8 scale dtype/shape 错 | 启动加 `--quantization-config-overrides` 或 monkeypatch 该层的 `process_weights_after_loading` |
| exp_004 | 关 fused kernel,走 torch fallback | fused_kernel | fused 实现数值差(只对 fused 模块如 attention) | 启动加 `--attention-backend torch_native`(或对应模块的 backend 切换) |
| exp_005 | 关 ACLGraph (`--disable-cuda-graph`) | graph_capture | graph capture 引入的偏差 | 启动加 `--disable-cuda-graph` |

## Attention(self_attn 整体)

跑 Linear 列表的 exp_001 / exp_004 / exp_005,加上:

| ID | 实验 | category | 验证假设 |
|---|---|---|---|
| exp_006 | mask 类型对齐(causal vs full vs sliding) | fused_kernel | mask 实现差 |
| exp_007 | 关 flash attention,走 vanilla math attn | fused_kernel | flash attn 数值差 |
| exp_008 | 比对 RoPE 输出(sin/cos table) | rope_implementation | RoPE 实现差 |

## RoPE / rotary_emb

| ID | 实验 | category | 验证假设 |
|---|---|---|---|
| exp_010 | theta 比对(SGLang vs HF) | rope_implementation | theta 不一致(默认 10000) |
| exp_011 | sin/cos table dtype 比对 | rope_implementation | table dtype 影响精度 |
| exp_012 | neox-style vs gpt-j-style 切换 | rope_implementation | rotary 旋转方式实现差 |

## LayerNorm / RMSNorm

| ID | 实验 | category | 验证假设 |
|---|---|---|---|
| exp_020 | eps 比对(SGLang vs HF) | dtype | eps 默认值不一致 |
| exp_021 | 强制 fp32 norm 计算 | dtype | norm 内部累加精度 |
| exp_022 | residual add 顺序检查(pre vs post norm) | dtype | residual order |

> **MVP 不含**:MoE 专用实验(routing / fused MoE)与 multi-host communication 实验(allreduce / all2all)。
> MoE 中 expert 是 Linear,exp_001-003 已覆盖;multi-host 是 v2 范围。

## 跨模块(累积漂移场景)

当层级 diff 显示**多层连续漂移**(不是单点突变)时,默认跑这组:

- exp_001(在 first_bad_layer 入口 fp32 cast)
- exp_020(eps 比对)
- exp_022(residual 顺序)

## 实验执行约定

1. **只改本模块行为**:每个实验是 minimal patch,不影响其他层 / 其他模块
2. **每个实验单独跑一次完整 P3**:重新 dump 该层输出,与 HF 比 drift
3. **同 prompt 集**:用 P1 验证过能复现的 prompts,不更换
4. **实验上限**:语义上限,默认最多 8 个实验。**不设时间预算** —— 定位问题正确性优先于墙钟时间
5. **失败处理**:某个实验本身跑不起来(比如 `--quantization-config-overrides` 不存在该选项)→ 该实验记 verdict=inconclusive,继续下一个

## 实验日志 schema(per `precision/experiments/exp_NNN_<name>.json`)

```json
{
  "exp_id": "exp_004",
  "name": "disable_fused_attn",
  "hypothesis": "fused attention 在该模块上数值偏差",
  "category": "fused_kernel",
  "command": "python -m sglang.launch_server ... --attention-backend torch_native",
  "drift_before": { "max_abs": 0.42, "cosine": 0.81 },
  "drift_after":  { "max_abs": 0.001, "cosine": 0.9999 },
  "verdict": "supports",
  "elapsed_seconds": 124,
  "notes": ""
}
```
