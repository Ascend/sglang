# Attention 调试参考

Attention 在 SGLang 里有多个 backend,NPU 上能用的是子集。

## SGLang Attention 抽象

```
RadixAttention(高层接口)
   ↓ dispatch by backend
AttentionBackend (e.g. flashinfer, triton, torch_native, npu_pfa)
   ↓
具体 kernel 实现
```

`--attention-backend <name>` 切换。NPU 上能用的:
- `torch_native`(默认 fallback,慢但通用)
- 部分 NPU 专用 backend(看 `python/sglang/srt/layers/attention/` 下有没有 `*_npu` 文件)

## 典型错误

### 1. Backend 不存在

```
KeyError: 'flashinfer'
Backend 'xxx' is not registered
```

**根因**:开了 GPU-only backend(`--attention-backend flashinfer`)在 NPU 上跑。

**修复**:删 flag(用默认),或换成 NPU 兼容的(如果不知道有哪些,Grep `class.*AttentionBackend` 找)。

### 2. Shape mismatch in attention forward

```
RuntimeError: shape '[bs, num_heads, seqlen, head_dim]' is invalid for input of size ...
```

**根因排序**:
1. tp split 错了 num_heads:`num_heads_per_tp = num_heads // tp_size`,但代码里取了全量 num_heads
2. GQA 的 KV head 复制逻辑没适配:NPU 某些 backend 需要 KV head 数等于 Q head 数,需要 expand
3. RoPE 维度匹配错:rope_theta / max_position 在 NPU backend 下处理路径不同

### 3. KV cache 索引越界

```
IndexError: index out of bounds for token_to_kv_pool
req_index_to_mamba_index_mapping index out of bounds
```

**根因**:KV pool 大小估算错了,常见原因是 `num_kv_heads` 没正确除以 tp。

参考 PoC 分支 commit `bef287c84` 已修复一处类似 bug。

### 4. Causal mask 不对

```
现象:输出能跑但 attention 看起来"看见了未来",输出乱
```

**调试**:
- 看是否用了正确的 `is_causal=True`
- 看 NPU backend 的 mask 构造逻辑是否与 GPU 路径一致
- 用一个短 prompt(<10 tokens)+ greedy decode 看 logits 第一个 token 是否合理

### 5. RoPE 实现差异

NPU 上某些算子(如 `apply_rotary_pos_emb`)的实现与 GPU 不一定 bit-exact。

- 若输出**完全乱**(NaN / 全同一个 token)→ 多半是 RoPE 没正确应用
- 若输出**接近但不一致** → 可能是 RoPE 数值精度差异(精度问题,交 precision_rca)

## 调试套路

### 第一招:short prompt + greedy

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0,
        "max_tokens": 10
    }'
```

greedy(temperature=0)能消除采样随机性,输出是否合理可以快速判定 attention 是否基本正确。

### 第二招:对比 HF eager 路径

如果怀疑 attention 实现,临时用 HF transformers 加载同样的 model,跑同样的 prompt,greedy,看前 10 个 token 是否一致。**不一致就是 attention 实现差异**(此时应升级 precision_rca,而非继续在 debug_engineer 里查)。

### 第三招:看 attention backend 源码

`python/sglang/srt/layers/attention/<backend>.py` 通常 200-500 行,Read 一遍能确定:
- 是否考虑了 NPU 特殊情况
- 是否有 device 分支
- KV cache 读写路径
