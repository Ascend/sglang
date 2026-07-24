# NPU 特性兼容性矩阵

下表是**当前 PoC 分支**(release/PoC_*)下的已知支持情况。**有把握的标 supported/unsupported,没把握标 unknown,不要瞎猜。**

## 特性支持矩阵(参考)

| 特性 | NPU 支持 | 备注 | 验证方法 |
|------|---------|------|---------|
| **basic_inference** | supported | 基础前向 + KV cache | 任意模型不加额外 flag 启动 |
| **TP** | supported | tensor parallel | `--tp <N>` |
| **DP** | supported | data parallel | `--dp <N>` |
| **EP** | supported(MoE only) | expert parallel | `--ep <N>` 仅 MoE 模型 |
| **DP-Attention** | supported(部分模型) | DP attention + EP MoE | `--enable-dp-attention`;DeepSeek-V3 类已验证 |
| **ACLGraph** | supported(decode-only / mtp 部分场景) | NPU 上的 graph capture,类似 CUDA Graph | `--enable-aclgraph`;**编译期较慢,首次启动会卡 30s+** |
| **DeepEP** | supported(MoE only) | NPU 优化的 all-to-all | `--enable-deepep`;依赖 deepep 编译 |
| **MTP** | supported(部分模型,decode-only fuse 已合入) | Multi-Token Prediction speculative decoding | `--speculative-algorithm EAGLE3` 或同等开关;参考 commit `6fc07c09a` 的 `causal_conv1d_update_mtp_npu` |
| **多模态(VLM)** | supported(qwen2_vl 类) | image input | 用 multimodal example 脚本 |
| **flash_attn (GPU)** | unsupported | GPU-only 实现 | —— |
| **flashinfer (GPU)** | unsupported | GPU-only 库 | —— |
| **fp8** | unknown | NPU 对 fp8 支持有限 | —— |
| **awq / gptq 量化** | unknown | 部分量化 kernel 不支持 | —— |
| **chunk prefill** | supported | 参考 commit `5e3e693ab` | `--chunked-prefill-size <N>` |

## 模型架构 × 特性 矩阵(已知)

| 模型 | ACLGraph | DeepEP | DP-Attn | MTP | 多模态 |
|------|---------|--------|---------|-----|--------|
| Qwen2 系列 | supported | na(非 MoE) | unknown | unknown | na |
| Qwen2-MoE | unknown | supported | unknown | unknown | na |
| Qwen2-VL | unknown | na | unknown | unknown | supported |
| DeepSeek-V2/V3 | supported | supported | supported | supported | na |
| Llama 系列 | supported | na | unknown | unknown | na |
| Mixtral | unknown | supported | unknown | unknown | na |

**unknown 不等于 unsupported**——意味着没在当前分支上验证过,主流程跑 test-validator 时会实际试一遍。

## 推断 supported / unsupported 的依据

- Grep `python/sglang/srt/models/<model>.py` 看有没有 NPU 分支或 `is_npu()` 检查
- 看 git log 找近期 commit(`git log --grep="ACLGraph"` / `--grep="MTP"`)
- 看 `python/sglang/srt/distributed/` 下是否有 NPU 对应实现

## 不要瞎填的字段

新模型(从未见过的架构)的所有特性默认 **unknown**,test-validator 会实际验证后才更新。**不要凭"应该支持吧"就写 supported**。
