# 模型架构分析报告

## 1. 基本信息
- 模型名称：[必填]
- 架构类型：[Dense / MoE / MoE-MLA / VLM]
- 架构名称：[必须与 HF config 中的 architectures 字段一致]

## 2. 模型配置
| 参数 | 取值 | 说明 |
|----|----|----|
| hidden_size | | |
| num_hidden_layers | | |
| num_attention_heads | | |
| num_key_value_heads | | |
| intermediate_size | | |
| vocab_size | | |
| max_position_embeddings | | |

## 3. MoE 配置（MoE 模型必填）
| 参数 | 取值 | 说明 |
|----|----|----|
| n_routed_experts | | 路由专家数 |
| n_shared_experts | | 共享专家数 |
| num_experts_per_tok | | 每 token 激活的专家数 |
| moe_intermediate_size | | 专家中间层维度 |

## 4. MLA 配置（MLA 模型必填）
| 参数 | 取值 | 说明 |
|----|----|----|
| q_lora_rank | | Query LoRA rank |
| kv_lora_rank | | KV LoRA rank |
| qk_nope_head_dim | | QK 非 RoPE 维度 |
| qk_rope_head_dim | | QK RoPE 维度 |
| v_head_dim | | Value 维度 |

## 5. 并行配置推导
### 推导过程
1. 基于 hidden_size=xxx 推出 min_tp=xxx
2. 基于 n_experts=xxx，EP 候选值为 [...]
3. 选择满足 TP % EP == 0 的 TP=xxx、EP=xxx
4. 校验设备数：需要 xxx，可用 xxx

### 配置结果
| 参数 | 推荐值 | 说明 |
|----|-----|----|
| TP | | Tensor Parallel |
| EP | | Expert Parallel |
| PP | | Pipeline Parallel |
| 总设备数 | | TP × PP |

### 约束校验
| 检查项 | 结果 | 详情 |
|-----|----|----|
| 设备数 | ✅/❌ | |
| TP/EP 可除性 | ✅/❌ | |
| 专家分配 | ✅/❌ | |
| Attention head | ✅/❌ | |

## 6. 资源评估
- 权重大小：xxx GB
- 最低设备数：xxx
- 推荐设备数：xxx
- 每设备显存：xxx GB

## 7. NPU 兼容性
- 兼容性：✅/⚠️/❌
- Attention backend：ascend
- MoE backend：fused_moe
- 已知问题：[...]
- 规避方案：[...]

## 8. 风险评估
| 等级 | 类别 | 描述 | 缓解措施 |
|----|----|----|------|

## 9. 参考模型
- SGLang 实现：[文件路径]
- 相似度：high / medium / low
- 主要差异：[...]

## 10. 下一步动作
- proceed：进入代码修改阶段
- call_debug_engineer：需要 debug 支持
