# NPU特定问题

## 1. 算子兼容性

### 1.1 不支持的算子列表

| 算子 | 状态 | 替代方案 |
|------|------|----------|
| flash_attn_varlen | ❌ 不支持 | 使用ascend后端 |
| triton kernels | ❌ 不支持 | 使用NPU原生实现 |
| 某些CUDA操作 | ❌ 不支持 | 使用NPU等效操作 |

### 1.2 替代实现方案

```python
# 检测NPU环境
from sglang.srt.utils import is_npu

if is_npu():
    # 使用NPU兼容实现
    ...
else:
    # 使用标准实现
    ...
```

### 1.3 CPU Fallback策略

对于不支持的算子，可以考虑：
1. 使用CPU计算（性能损失）
2. 寻找NPU等效实现
3. 等待算子支持更新

---

## 2. 内存管理

### 2.1 NPU内存模型

NPU内存与GPU不同：
- HBM作为主内存
- 需要显式管理内存

### 2.2 内存碎片问题

**现象**：
```
RuntimeError: NPU out of memory (fragmentation)
```

**解决方案**：
```python
# 清空缓存
import torch_npu
torch_npu.npu.empty_cache()

# 同步
torch_npu.npu.synchronize()
```

### 2.3 内存优化技巧

| 技巧 | 说明 |
|------|------|
| 预分配 | 提前分配所需内存 |
| 内存池 | 使用内存池管理 |
| 量化 | 减少内存占用 |

---

## 3. NPU已知限制

### 3.1 限制列表

| 限制ID | 类别 | 描述 | 错误模式 | 解决方案 |
|--------|------|------|----------|----------|
| L001 | graph | NPU使用NPU Graph，非CUDA Graph | cuda_graph_runner, Capture cuda graph | 使用 --disable-cuda-graph 或检查NPU Graph兼容性 |
| L002 | operator | 部分算子不支持NPU Graph capture | operator not supported, graph capture failed | 禁用graph或寻找替代实现 |
| L003 | memory | 长上下文对KV Cache内存压力大 | out of memory, OOM | 减小context_length或使用MLA压缩 |
| L004 | parallel | EP配置必须满足约束 | ZeroDivisionError, expert distribution | 确保满足 tp_size % ep_size == 0 和 n_experts % ep_size == 0 |

## 4. 性能调优

### 4.1 Attention后端选择

NPU使用 `ascend` 后端：
```bash
--attention-backend ascend
```

### 4.2 MoE实现选择

| 实现 | 适用场景 |
|------|----------|
| FusedMoE | 单卡或小规模EP |
| DeepEP | 大规模EP（需HCCL） |

### 4.3 通信优化

```bash
# HCCL配置
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE=120
```

---

## 5. 常见错误案例

### 案例1：Ascend后端初始化失败

**现象**：
```
RuntimeError: Failed to initialize ascend backend
```

**排查**：
1. 检查CANN版本
2. 检查NPU驱动
3. 检查torch_npu安装

**检查命令**：
```bash
npu-smi info
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

### 案例2：算子不支持

**现象**：
```
NotImplementedError: No operator found for xxx on NPU
```

**解决方案**：
1. 检查是否有NPU等效实现
2. 考虑CPU fallback
3. 联系开发团队

### 案例3：内存不足

**现象**：
```
RuntimeError: NPU out of memory
```

**解决方案**：
| 策略 | 命令 |
|------|------|
| 减小batch | `--max-running-requests 4` |
| 减小序列长度 | `--context-length 4096` |
| 启用量化 | `--quantization fp8` |
| 增大TP | `--tp 4` |

---

## 6. 环境检查

### 6.1 检查NPU设备

```bash
# 查看NPU信息
npu-smi info

# 查看设备数量
npu-smi info -l
```

### 6.2 检查软件环境

```bash
# 检查CANN
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 检查torch_npu
python -c "import torch_npu; print(torch_npu.__version__)"
```

### 6.3 检查通信

```bash
# 检查HCCL
python -c "import torch.distributed as dist; dist.init_process_group(backend='hccl')"
```

---

## 7. 参考资源

| 资源 | 说明 |
|------|------|
| `python/sglang/srt/hardware_backend/npu/` | SGLang NPU实现 |
| `python/sglang/srt/utils.py` | is_npu等工具函数 |
| CANN文档 | https://www.hiascend.com/document |
