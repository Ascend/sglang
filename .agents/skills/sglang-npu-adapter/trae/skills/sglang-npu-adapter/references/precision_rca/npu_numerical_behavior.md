# NPU Numerical Behavior Reference

本文件覆盖 NPU 上**对精度有影响**的数值行为细节(数值视角)。NPU 硬件规格(memory/算力)请查询厂商文档,不在本 skill 范围内。

NPU(Ascend 910 A3)上目标 RCA 涉及的 dtype:`fp32 / bf16 / fp16 / W8A8 (int8)`。

## Accumulator 精度

| 数据类型 | 累加器 | 备注 |
|---|---|---|
| fp32 | fp32 | full precision |
| bf16 | fp32 in, bf16 out | matmul 默认 |
| fp16 | fp32 accumulator(部分 op),fp16 accumulator(其它 op,如 attention reduction) | softmax/attention 部分路径有 fp16 累加,精度可能下降 |
| W8A8 (int8) | fp32 accumulator,scale 走 fp16/bf16 | 量化 linear 路径 |

## Fused 实现的已知数值差(与 vanilla math 比)

### Fused Attention

- 不同 head_dim 路径精度可能不同(64 / 128 / 256 各有独立优化)
- mask 类型(causal / full / sliding)切换路径,某些组合走不同 kernel
- 长 seq_len 时,**block-wise softmax** 的归一化可能与 vanilla 软极大累积不一致(数值上等价但浮点不等)

### Fused MoE

- top-k routing 结果在 fp16 路径下可能与 fp32 路径选不同 expert(top-k 的 tie-breaking 受 dtype 影响)
- expert dispatching 的 reorder 与原顺序不一致(数值上等价但不 bit-exact)

## Quantization

### W8A8

- weight 8-bit 整数,activation 8-bit 整数,scale fp16/bf16
- per-channel weight scale 是常态;per-tensor activation scale 也常见
- scale 加载顺序:加载 weight 后 process_weights_after_loading 中调整 scale 维度

## 通信原语数值行为

- `allreduce(sum)`:多 rank 求和顺序不确定(NCCL 不保证 deterministic),fp16 求和误差累积可见
- `all2all`:专家分发,数值上无操作但可能影响后续 norm 的输入分布
- 建议:涉及 reduce 的层用 fp32 reduce dtype(`--reduce-dtype fp32` 若支持)

## ACLGraph 捕获限制

- 动态 shape(prefill 期变化)与 ACLGraph 不兼容
- 自定义 op 若未注册到 graph,运行时 fallback
- 量化路径(W8A8)在 ACLGraph 下个别 op 可能捕获不完整,运行时 fallback 到 eager

---

# 已知算子侧问题清单(Operator-Side Issues)

来源:**CANN release notes** + **Ascend 开发者社区精度专题帖**。这些是公开知识,不是从 git 挖出来的。

## 当前已知(模板;实施时按当时情况填充)

```yaml
# 每条 entry 填入:
# - issue_id: 简短标识
# - applies_to: { chip, cann_range, torch_npu_range }
# - symptom: 在哪种 first_bad_module 类型上出现,drift 特征
# - workaround: sglang 侧关哪个开关 / 改哪个配置
# - vendor_reference: CANN release note URL / 社区帖 URL
# - resolved_in: 哪个版本之后修复(若已修)

- issue_id: "TEMPLATE_fused_attn_head128_pre_rc2"
  applies_to:
    chip: "910B*"
    cann_range: "<8.0.RC2"
    torch_npu_range: "*"
  symptom: "first_bad_module 在 self_attn,head_dim==128,cosine < 0.95"
  workaround: "--attention-backend torch_native"
  vendor_reference: "<填入 CANN release note URL>"
  resolved_in: "CANN 8.0.RC2"
```

**实施时(M4 之前)需要做**:由用户/同事查阅当前 CANN release note,填充这个清单的真实条目。本设计不强行写死具体 issue,因为该清单会随 CANN 版本演进。
