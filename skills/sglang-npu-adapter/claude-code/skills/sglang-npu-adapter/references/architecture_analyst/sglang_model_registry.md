# SGLang模型注册机制

## 1. 注册流程概述

SGLang使用自动发现机制注册模型，核心流程如下：

1. `ModelRegistry` 在初始化时调用 `import_model_classes` 函数
2. 遍历 `sglang.srt.models` 包下的所有模块
3. 查找每个模块中的 `EntryClass` 变量
4. 将 `EntryClass` 的类名作为架构名，注册到 `models` 字典中

### 关键代码路径
- 注册入口：`python/sglang/srt/models/registry.py`
- 模型目录：`python/sglang/srt/models/`

## 2. EntryClass定义规范

### 单模型注册
```python
# 在模型文件末尾
EntryClass = Qwen2ForCausalLM
```

### 多模型注册
当一个模块支持多个模型架构时：
```python
EntryClass = [Qwen2ForCausalLM, Qwen2ForClassification]
```

### 命名约定
- 架构名 = EntryClass的类名（如 `Qwen2ForCausalLM`）
- 必须与 HuggingFace config 中的 `architectures` 字段匹配

## 3. 模型文件结构

### 必要组件
```python
# 1. Config类（通常从transformers导入）
from transformers import Qwen2Config

# 2. 模型类（继承nn.Module）
class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        ...
    
    def forward(self, input_ids, positions, forward_batch, ...):
        ...
    
    def load_weights(self, weights):
        ...

# 3. EntryClass注册
EntryClass = Qwen2ForCausalLM
```

### 可选组件
```python
# MLP模块
class Qwen2MLP(nn.Module):
    ...

# Attention模块
class Qwen2Attention(nn.Module):
    ...

# DecoderLayer模块
class Qwen2DecoderLayer(nn.Module):
    ...
```

## 4. 权重加载机制

### load_weights方法
```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    # 1. 定义参数映射
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    
    # 2. 获取参数字典
    params_dict = dict(self.named_parameters())
    
    # 3. 遍历权重并加载
    for name, loaded_weight in weights:
        # 处理权重名称映射
        # 使用weight_loader加载
        ...
```

### 权重名称映射
| HuggingFace权重名 | SGLang权重名 |
|------------------|--------------|
| model.layers.N.self_attn.q_proj | model.layers.N.self_attn.qkv_proj (q) |
| model.layers.N.self_attn.k_proj | model.layers.N.self_attn.qkv_proj (k) |
| model.layers.N.self_attn.v_proj | model.layers.N.self_attn.qkv_proj (v) |
| model.layers.N.mlp.gate_proj | model.layers.N.mlp.gate_up_proj (0) |
| model.layers.N.mlp.up_proj | model.layers.N.mlp.gate_up_proj (1) |

## 5. 注册示例

### 简单LLM：qwen2.py
```python
# 文件：python/sglang/srt/models/qwen2.py

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        self.model = Qwen2Model(config, quant_config, ...)
        self.lm_head = ParallelLMHead(...)
    
    def forward(self, ...):
        ...
    
    def load_weights(self, weights):
        ...

EntryClass = Qwen2ForCausalLM
```

### MoE模型：qwen2_moe.py
```python
# 文件：python/sglang/srt/models/qwen2_moe.py

class Qwen2MoeForCausalLM(nn.Module):
    # 包含SparseMoeBlock
    ...

EntryClass = Qwen2MoeForCausalLM
```

### VLM模型：qwen2_vl.py
```python
# 文件：python/sglang/srt/models/qwen2_vl.py

class Qwen2VLForConditionalGeneration(nn.Module):
    # 包含vision_tower和projector
    ...

EntryClass = Qwen2VLForConditionalGeneration
```

## 6. 常见问题

### Q1: 架构名冲突怎么办？
A: 使用 `overwrite=True` 参数：
```python
ModelRegistry.register("custom_models", overwrite=True)
```

### Q2: 如何添加自定义模型路径？
A: 设置环境变量：
```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE="my_custom_models"
```

### Q3: 模型没有被识别？
A: 检查以下几点：
1. `EntryClass` 是否正确定义
2. 类名是否与HF config中的 `architectures` 匹配
3. 模块是否在正确的包路径下

## 7. 参考代码

- [registry.py](file:///home/tsz/Code_0316/sglang/python/sglang/srt/models/registry.py)
- [qwen2.py](file:///home/tsz/Code_0316/sglang/python/sglang/srt/models/qwen2.py)
- [qwen2_moe.py](file:///home/tsz/Code_0316/sglang/python/sglang/srt/models/qwen2_moe.py)
