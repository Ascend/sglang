# RoPE相关问题

## 1. RoPE基础概念

### 1.1 什么是RoPE

旋转位置编码(Rotary Position Embedding)通过旋转向量来编码位置信息：
```
f(x, m) = x * e^{imθ}
```
- m: 位置索引
- θ: 频率，θ_i = 10000^{-2i/d}

### 1.2 RoPE的优势
- 相对位置编码
- 支持外推（通过scaling）
- 计算效率高

---

## 2. RoPE Scaling类型

### 2.1 Linear Scaling

线性插值位置索引：
```
position_new = position_original / factor
```

**配置格式**：
```json
{"type": "linear", "factor": 2.0}
```

**效果**：简单有效，可能损失精度

### 2.2 Dynamic NTK

动态调整RoPE频率：
```
θ_new = θ * (factor * seq_len / original_seq_len) ^ (-dim/(dim-2))
```

**配置格式**：
```json
{"type": "dynamic", "factor": 2.0}
```

**效果**：更好的外推能力

### 2.3 Yarn

结合多种策略的混合方法：

**配置格式**：
```json
{
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 4096
}
```

**效果**：最佳外推效果

### 2.4 LongRoPE

优化的插值策略：

**配置格式**：
```json
{
    "type": "longrope",
    "long_factor": [...],
    "short_factor": [...]
}
```

**效果**：支持超长上下文

---

## 3. 格式转换问题

### 3.1 HuggingFace vs SGLang格式

| 来源 | 格式 |
|------|------|
| HuggingFace | `{"rope_type": "linear", "factor": 2.0}` |
| SGLang | `{"type": "linear", "factor": 2.0}` |

### 3.2 转换代码

```python
def convert_rope_scaling(rope_scaling):
    if rope_scaling is None:
        return None
    
    # HF使用 "rope_type"，SGLang使用 "type"
    if "rope_type" in rope_scaling:
        result = {
            "type": rope_scaling["rope_type"],
            "factor": rope_scaling.get("factor", 1.0),
        }
        # 复制其他字段
        for key in ["original_max_position_embeddings", "long_factor", "short_factor"]:
            if key in rope_scaling:
                result[key] = rope_scaling[key]
        return result
    
    return rope_scaling
```

### 3.3 在模型中使用

```python
rope_scaling = getattr(config, "rope_scaling", None)
rope_scaling = convert_rope_scaling(rope_scaling)

self.rotary_emb = get_rope(
    self.head_dim,
    rotary_dim=self.head_dim,
    max_position=max_position_embeddings,
    base=rope_theta,
    rope_scaling=rope_scaling,
)
```

---

## 4. 常见错误

### 4.1 rope_type vs type

**错误现象**：
```
RuntimeError: RoPE scaling parameter 'rope_type' not recognized
```

**原因**：HuggingFace格式与SGLang格式不一致

**修复**：添加格式转换（见3.2节）

### 4.2 factor值不正确

**错误现象**：输出质量下降或位置信息丢失

**原因**：`factor` 值与模型训练时不一致

**排查**：
1. 检查原始模型的 `rope_scaling` 配置
2. 确认factor值正确传递

### 4.3 max_position_embeddings不匹配

**错误现象**：长序列输出错误

**原因**：`max_position_embeddings` 与scaling配置不匹配

**修复**：
```python
max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
# 如果有rope_scaling，可能需要调整
```

---

## 5. 特殊RoPE变体

### 5.1 M-RoPE（多模态）

用于多模态模型（如Qwen2-VL）：
- 支持图像、视频等多模态输入
- 需要特殊的位置编码处理

**参考实现**：`python/sglang/srt/layers/rotary_embedding/mrope.py`

### 5.2 3D-RoPE（视频）

用于视频模型：
- 时间维度 + 空间维度
- 需要三维位置编码

---

## 6. NPU RoPE实现

### 6.1 NPU RoPE支持

| RoPE类型 | NPU支持状态 |
|----------|-------------|
| 标准RoPE | ✅ 支持 |
| Linear Scaling | ✅ 支持 |
| Dynamic NTK | ✅ 支持 |
| Yarn | ⚠️ 部分支持 |
| LongRoPE | ⚠️ 部分支持 |
| M-RoPE | ⚠️ 需检查 |

### 6.2 NPU RoPE性能优化

NPU上RoPE计算可能需要特殊优化：
- 使用NPU专用算子
- 注意精度问题

---

## 7. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/layers/rotary_embedding/factory.py` | get_rope函数 |
| `python/sglang/srt/layers/rotary_embedding/` | RoPE实现目录 |
| `python/sglang/srt/models/qwen2.py` | RoPE使用示例 |
