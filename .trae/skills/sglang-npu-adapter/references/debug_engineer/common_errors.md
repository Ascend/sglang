# 常见错误及解决方案

## 1. 错误模式快速参考

### 1.1 错误模式库

| 错误ID | 关键词 | 错误类型 | 类别 | 诊断 | 根本原因 | 解决方向 |
|--------|--------|----------|------|------|----------|----------|
| E001 | cuda_graph_runner, Capture npu graph | npu_graph_error | hardware | NPU Graph捕获失败 | 算子不支持graph capture或动态shape问题 | 检查算子兼容性，考虑禁用graph |
| E002 | ZeroDivisionError, integer division or modulo by zero | parallel_config_error | config | 并行配置约束不满足 | TP/EP配置不满足整除约束 | 重新计算TP/EP满足tp_size % ep_size == 0 |
| E003 | KeyError, model_type | config_load_error | config | transformers不识别模型类型 | 模型配置未注册到transformers | 创建自定义配置类并注册到_CONFIG_REGISTRY |
| E004 | out of memory, OOM, CUDA out of memory | memory_error | resource | 内存不足 | 权重或KV Cache超出设备内存 | 减小context_length、batch_size或增加设备数 |
| E005 | device count, RuntimeError: device | device_error | hardware | 设备数量问题 | 配置的设备数超过可用设备数 | 减少TP/PP或检查设备可用性 |
| E006 | operator not supported, not implemented | operator_error | hardware | 算子不支持 | NPU不支持该算子 | 寻找替代实现或绕过方案 |
| E007 | ACL error, ASCEND error | npu_driver_error | hardware | NPU驱动错误 | NPU底层驱动或硬件问题 | 检查驱动版本、设备状态、内存清理 |

## 2. 错误分类概览

模型适配过程中的错误可分为以下几类：

| 类别 | 占比 | 典型错误 |
|------|------|----------|
| 配置相关 | 30% | 字段缺失、格式不匹配 |
| 权重加载 | 25% | 名称不匹配、形状不匹配 |
| 算子兼容 | 20% | NPU不支持、后端不兼容 |
| 内存问题 | 15% | OOM、内存碎片 |
| 其他 | 10% | 环境问题、版本问题 |

---

## 2. 配置相关错误

### 2.1 rope_scaling参数格式不匹配

**错误现象**：
```
RuntimeError: RoPE scaling parameter 'rope_type' not recognized
```

**原因分析**：
HuggingFace新版本使用 `rope_type`，而SGLang期望 `type`

**解决方案**：
在模型初始化时转换格式：
```python
rope_scaling = getattr(config, "rope_scaling", None)
if rope_scaling is not None and "rope_type" in rope_scaling:
    rope_scaling = {
        "type": rope_scaling["rope_type"],
        "factor": rope_scaling.get("factor", 1.0),
    }
```

### 2.2 配置字段缺失

**错误现象**：
```
AttributeError: 'Qwen2Config' object has no attribute 'head_dim'
```

**原因分析**：
新模型可能使用老版本transformers不支持的配置字段

**解决方案**：
在模型代码中添加默认值：
```python
head_dim = getattr(config, "head_dim", None)
if head_dim is None:
    head_dim = config.hidden_size // config.num_attention_heads
```

### 2.3 num_key_value_heads缺失

**错误现象**：
```
AttributeError: 'Config' object has no attribute 'num_key_value_heads'
```

**原因分析**：
老模型使用MHA，没有GQA配置

**解决方案**：
```python
num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
```

---

## 3. 权重加载错误

### 3.1 权重名称不匹配

**错误现象**：
```
WARNING: Parameter model.layers.0.self_attn.q_proj.weight not found in params_dict
```

**原因分析**：
HuggingFace权重名称与SGLang期望不一致

**解决方案**：
在 `load_weights` 方法中添加名称映射：
```python
# 定义映射关系
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
]

for name, loaded_weight in weights:
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name in name:
            name = name.replace(weight_name, param_name)
            # 使用weight_loader加载
            ...
```

### 3.2 权重形状不匹配

**错误现象**：
```
RuntimeError: size mismatch for model.layers.0.self_attn.qkv_proj.weight: 
copy_() shape '[4096, 4096]' is invalid for input of size '12288'
```

**原因分析**：
- TP（Tensor Parallel）配置不正确
- 权重sharding方式不匹配

**解决方案**：
1. 检查TP配置是否与权重匹配
2. 确认QKVParallelLinear的参数正确：
```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    head_dim,           # 每个头的维度
    total_num_heads,    # 总头数
    total_num_kv_heads, # KV头数
    ...
)
```

### 3.3 权重加载顺序问题

**错误现象**：
```
RuntimeError: Expected all tensors to be on the same device
```

**原因分析**：
权重加载时设备不一致

**解决方案**：
确保权重加载在正确的设备上：
```python
for name, loaded_weight in weights:
    loaded_weight = loaded_weight.to(device)  # 确保设备一致
    ...
```

---

## 4. 算子兼容性错误

### 4.1 NPU算子不支持

**错误现象**：
```
NotImplementedError: No operator found for `flash_attn_varlen_func` on NPU
```

**原因分析**：
NPU不支持某些GPU专用算子

**解决方案**：
1. 切换到支持的Attention后端：
```bash
# 使用ascend后端
python -m sglang.launch_server ... --attention-backend ascend
```

2. 或在代码中添加fallback：
```python
if is_npu():
    # 使用NPU兼容实现
    ...
else:
    # 使用标准实现
    ...
```

### 4.2 Attention后端初始化失败

**错误现象**：
```
RuntimeError: Failed to initialize FlashInfer backend
```

**原因分析**：
- CUDA版本不兼容
- GPU架构不支持

**解决方案**：
尝试其他后端：
```bash
--attention-backend triton  # 或 flashinfer
```

### 4.3 RoPE实现不兼容

**错误现象**：
```
RuntimeError: RoPE implementation not available for the given configuration
```

**原因分析**：
特殊的RoPE配置可能不被支持

**解决方案**：
检查RoPE配置是否在支持范围内，必要时简化配置。

---

## 5. 内存相关错误

### 5.1 CUDA/NPU OOM

**错误现象**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**原因分析**：
- 模型太大
- Batch Size过大
- 序列长度过长
- KV Cache占用过多

**解决方案**：

| 策略 | 命令参数 | 效果 |
|------|----------|------|
| 增大TP | `--tp 4` 或 `--tp 8` | 分摊模型权重 |
| 减小BS | `--max-running-requests 4` | 减少并发 |
| 减小序列长度 | `--context-length 4096` | 减少KV Cache |
| 启用量化 | `--quantization fp8` | 减少权重内存 |
| 启用KV Cache量化 | `--kv-cache-dtype fp8` | 减少KV Cache |

### 5.2 内存碎片问题

**错误现象**：
```
RuntimeError: CUDA error: out of memory (even though enough memory appears available)
```

**原因分析**：
内存碎片导致无法分配连续内存

**解决方案**：
```bash
# 设置内存分配器
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 6. 运行时错误

### 6.1 模型架构不匹配

**错误现象**：
```
ValueError: Model architectures ['CustomModel'] are not supported for now.
```

**原因分析**：
模型架构名未在SGLang中注册

**解决方案**：
1. 确认EntryClass定义正确
2. 检查架构名是否与HF config匹配
3. 必要时创建新的模型文件

### 6.2 tokenizer加载失败

**错误现象**：
```
OSError: Can't load tokenizer for 'path/to/model'
```

**原因分析**：
tokenizer文件缺失或格式不正确

**解决方案**：
确保模型目录包含必要的tokenizer文件：
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json` (如适用)
- `merges.txt` (如适用)

---

## 7. 错误诊断流程

```
错误发生
    │
    ├─ 查看错误类型
    │     ├─ AttributeError → 配置字段问题
    │     ├─ RuntimeError → 运行时问题
    │     ├─ ValueError → 参数值问题
    │     └─ OutOfMemoryError → 内存问题
    │
    ├─ 检查错误堆栈
    │     └─ 定位到具体代码位置
    │
    ├─ 分析上下文
    │     ├─ 检查config配置
    │     ├─ 检查权重名称
    │     └─ 检查运行环境
    │
    └─ 应用解决方案
          └─ 验证修复效果
```

---

## 8. 参考代码

- `python/sglang/srt/models/qwen2.py` - 权重加载示例
- `python/sglang/srt/layers/rotary_embedding/` - RoPE实现
- `python/sglang/srt/layers/radix_attention.py` - Attention实现
