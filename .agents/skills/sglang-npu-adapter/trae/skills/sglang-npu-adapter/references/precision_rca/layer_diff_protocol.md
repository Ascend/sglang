# Layer Diff Protocol

## Trap A:dtype 一致性约定(强制)

HF reference 与 SGLang 的模型加载**必须使用 `precision_context.dtype` 指定的同一 dtype**。默认从 `output/output_summary.json` 的 dtype 字段继承(由 `scripts/probe_model_arch.py` 写入)。

绝不允许 HF 跑 fp32 而 SGLang 跑 bf16 后做 diff —— 这只会反映 dtype 截断,不反映真实漂移。

## Hook 协议

### Where to hook

每个 transformer block 的**出口**:

```python
hook_targets = [f"model.layers.{i}" for i in range(num_layers)]
```

### What to dump

- 输出张量的 last-token hidden state(避免大文件)
- 多 prompt 拼为一个 npz:`shape = [n_prompts, hidden_size]`
- 文件:`precision/{hf,sgl}_layer_outputs/layer_{i}.npz`

### Forward hook 实现要点

```python
def make_hook(layer_idx, store):
    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            out = outputs[0]
        else:
            out = outputs
        # last-token only
        last_tok = out[:, -1, :].detach().cpu().to(torch.float32).numpy()
        store[layer_idx].append(last_tok)
    return hook
```

每个 prompt 跑结束后将 `store` 拼接保存为 npz。

## Trap B:模块命名对不上 → fallback to by-index

某些 SGLang 模型重命名了 HF 的子模块(如 `self_attn` → `attn`,或合并 `q_proj/k_proj/v_proj` 为 `fused_qkv`)。

### Strategy 1 (default):by name

维护已知映射表:

| 模型架构 | HF 命名 | SGLang 命名 |
|---|---|---|
| llama-style (qwen2/qwen3) | `model.layers.{i}.self_attn` | `model.layers.{i}.self_attn`(一致) |
| dsv4 | `model.layers.{i}.self_attn` | `model.layers.{i}.attn`(重命名) |
| qwen3-next | `model.layers.{i}.linear_attn` | `model.layers.{i}.linear_attn`(一致) |

(此表持续维护,新增架构时追加)

### Strategy 2:fallback to by-index

名字对不上 → 按 forward hook 调用顺序配对(HF 第 N 个 hook 输出 ↔ SGLang 第 N 个 hook 输出)。前提:两边的 transformer block 数量一致。

退化标志写入 `layer_diff.json`:

```json
{
  "naming_strategy": "by_name | by_index",
  "naming_fallback_reason": "<if by_index>"
}
```

## 二分扫描算法

```python
def find_first_bad_layer(num_layers, hf_outputs, sgl_outputs, rtol, atol):
    """
    返回第一个让 torch.allclose 失败的 layer index。
    O(log N) NPU forward 时间(若两端的 dump 都已存好,只是文件 IO)。

    若所有层都 fail → 返回 0
    若所有层都 pass → 返回 num_layers(标记 inconclusive)
    """
    lo, hi = 0, num_layers - 1
    # 先检查 layer 0 是否就坏了
    if not torch.allclose(hf_outputs[0], sgl_outputs[0], rtol=rtol, atol=atol):
        return 0
    if torch.allclose(hf_outputs[-1], sgl_outputs[-1], rtol=rtol, atol=atol):
        return num_layers  # 全过,无 first bad layer
    # 二分:寻找第一个让 allclose 失败的下标
    while lo < hi:
        mid = (lo + hi) // 2
        if not torch.allclose(hf_outputs[mid], sgl_outputs[mid], rtol=rtol, atol=atol):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

## NPU 非确定性

某些 NPU op(尤其 fp16 reduction)非 bit-exact deterministic。

### 检测

每个 prompt 跑两次同条件 → 若两次输出差异 > tolerance/10 则该 layer 标 *NPU non-deterministic*。

### 处理

- 比对取**两次的中位数**(per-element)
- `layer_diff.json` 中标记 `non_deterministic_layers: [...]`
- 报告里 surface 这一事实

## layer_diff.json 输出 schema

```json
{
  "tolerance": { "rtol": 1e-3, "atol": 1e-3 },
  "naming_strategy": "by_name",
  "naming_fallback_reason": null,
  "first_bad_layer": 7,
  "first_bad_layer_method": "binary_search | sequential | not_found",
  "layers": [
    { "i": 0, "max_abs": 0.001, "cosine": 0.9999, "passed": true, "non_deterministic": false },
    { "i": 1, "...": "..." }
  ],
  "non_deterministic_layers": []
}
```
