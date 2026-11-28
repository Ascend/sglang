# SGLang基础知识

## 一、SGLang概述

SGLang是一个高性能的大语言模型服务框架，专注于：
- **高效推理**：通过RadixAttention实现KV Cache共享和重用
- **灵活部署**：支持多种后端（CUDA、NPU等）
- **模型适配**：提供统一的模型注册和加载机制

## 二、核心组件

### 2.1 RadixAttention
SGLang的核心注意力实现，支持自动KV Cache管理、前缀缓存、多轮对话优化。

**参考实现**：`python/sglang/srt/layers/radix_attention.py` 中的 `RadixAttention` 类

### 2.2 LogitsProcessor

**参考实现**：`python/sglang/srt/layers/logits_processor.py`

### 2.3 模型注册

通过 `EntryClass` 注册模型，架构名 = EntryClass的类名：

**参考实现**：`python/sglang/srt/models/registry.py`

## 三、关键导入

```python
# 注意力
from sglang.srt.layers.radix_attention import RadixAttention
# RoPE
from sglang.srt.layers.rotary_embedding import get_rope
# 归一化
from sglang.srt.layers.layernorm import RMSNorm
# 线性层（支持TP）
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
# Embedding
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
# 激活函数
from sglang.srt.layers.activation import SiluAndMul
# 分布式
from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_pp_group,
)
# 工具函数
from sglang.srt.utils import add_prefix, make_layers
from sglang.srt.layers.utils import get_layer_id, PPMissingLayer
```

## 四、模型文件结构

### 4.1 必要组件
- **Config类**：通常从transformers导入
- **模型类**：继承nn.Module，实现 `__init__`、`forward`、`load_weights`
- **EntryClass注册**：在文件末尾定义

### 4.2 可选组件
- **MLP模块**：如 `Qwen2MLP`
- **Attention模块**：如 `Qwen2Attention`
- **DecoderLayer模块**：如 `Qwen2DecoderLayer`

**完整示例参考**：`python/sglang/srt/models/qwen2.py`

## 五、模型适配要点

### 5.1 适配流程概览

```
1. 分析模型架构
   └─ 识别：Attention类型、MLP类型、Norm类型、RoPE配置

2. 选择参考模型
   └─ 找到最相似的已适配模型作为参考

3. 创建模型文件
   └─ 复制参考模型，修改差异部分

4. 实现权重加载
   └─ 处理权重名称映射

5. 测试验证
   └─ 运行推理，验证输出正确性
```

### 5.2 必须处理的差异点

| 差异点 | 检查方法 | 处理方式 |
|--------|----------|----------|
| **Attention类型** | 检查 `num_key_value_heads` | 调整QKVParallelLinear参数 |
| **RoPE配置** | 检查 `rope_scaling` | 转换格式或自定义实现 |
| **权重名称** | 对比HF权重和SGLang参数 | 添加名称映射 |
| **配置字段** | 检查config属性 | 添加默认值处理 |

### 5.3 权重名称映射规则

SGLang通常将独立的Q/K/V投影合并为单个 `qkv_proj`：

| HuggingFace | SGLang |
|-------------|--------|
| q_proj, k_proj, v_proj | qkv_proj (sharded) |
| gate_proj, up_proj | gate_up_proj (merged) |

**映射代码模式参考**：`python/sglang/srt/models/qwen2.py` 中的 `load_weights` 方法

### 5.4 配置字段默认值处理

新模型可能使用老版本transformers不支持的配置字段，需要添加默认值：

```python
head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
rope_theta = getattr(config, "rope_theta", 10000)
max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
```

### 5.5 RoPE Scaling格式转换

HuggingFace和SGLang的rope_scaling格式可能不同：

```python
def convert_rope_scaling(rope_scaling):
    if rope_scaling is None:
        return None
    if "rope_type" in rope_scaling:
        return {
            "type": rope_scaling["rope_type"],
            "factor": rope_scaling.get("factor", 1.0),
        }
    return rope_scaling
```

### 5.6 Pipeline Parallel支持

如果模型需要支持Pipeline Parallel，需要：
1. 使用 `PPMissingLayer` 处理非本rank的层
2. 使用 `get_pp_group()` 判断当前rank
3. 使用 `make_layers` 创建层列表

**参考实现**：`python/sglang/srt/models/qwen2.py` 中的 `Qwen2Model` 类

## 六、NPU适配要点

### 6.1 NPU检测与初始化

```python
from sglang.srt.utils import is_npu
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

_is_npu = is_npu()
if _is_npu:
    init_npu_backend()
```

### 6.2 CUDA API 自动转换

`torch_npu.contrib.transfer_to_npu` 会自动将 `torch.cuda.xxx` 转换为 `torch.npu.xxx`：

| API | NPU行为 | 安全性 |
|-----|--------|--------|
| `torch.cuda.device_count()` | 返回NPU设备数量 | ✅ |
| `tensor.cuda()` | 转换到NPU | ✅ |
| `device="cuda"` | 创建在NPU上 | ✅ |
| `torch.cuda.is_available()` | **返回False** | ⚠️ 用`is_npu()`判断 |

### 6.3 注意力后端配置

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --attention-backend ascend \
    --device npu
```

## 七、常用工具函数

### 7.1 add_prefix
为参数名添加前缀，用于层级命名：
```python
prefix = add_prefix("self_attn", "model.layers.0")
# 结果: "model.layers.0.self_attn"
```

### 7.2 make_layers
创建层列表，支持Pipeline Parallel：
```python
layers, start_layer, end_layer = make_layers(
    config.num_hidden_layers,
    lambda idx, prefix: DecoderLayer(layer_id=idx, config=config, prefix=prefix),
    pp_rank=pp_rank,
    pp_size=pp_size,
    prefix=add_prefix("layers", prefix),
)
```

### 7.3 get_layer_id
从参数名中提取层ID：
```python
layer_id = get_layer_id("model.layers.5.self_attn.q_proj.weight")
# 结果: 5
```

## 八、参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/models/qwen2.py` | 完整LLM示例（GQA + Gated MLP） |
| `python/sglang/srt/models/llama.py` | Llama架构参考 |
| `python/sglang/srt/models/qwen2_moe.py` | MoE模型示例 |
| `python/sglang/srt/models/deepseek_v2.py` | MLA架构示例 |
| `python/sglang/srt/models/registry.py` | 模型注册机制 |
| `python/sglang/srt/layers/radix_attention.py` | RadixAttention实现 |
| `python/sglang/srt/layers/rotary_embedding/` | RoPE实现 |
| `python/sglang/srt/hardware_backend/npu/` | NPU后端实现 |
