# NPU基础知识

## 一、NPU概述

NPU（Neural Processing Unit）是专为深度学习设计的加速器。SGLang主要支持华为昇腾（Ascend）NPU。

### 1.1 NPU vs CUDA 对比

| 特性 | CUDA (NVIDIA) | NPU (Ascend) |
|------|---------------|--------------|
| 软件栈 | CUDA, cuDNN | CANN |
| Python库 | torch.cuda | torch_npu |
| 分布式通信 | NCCL | HCCL |
| 注意力后端 | flashinfer, triton | ascend |

## 二、torch_npu 核心API

### 2.1 设备管理

```python
import torch
import torch_npu

# 检查NPU可用性
torch.npu.is_available()

# 设备数量和当前设备
torch.npu.device_count()
torch.npu.current_device()

# 设备名称
torch.npu.get_device_name()  # 如 "Ascend910_9382"

# 设置设备
torch.npu.set_device(device_id)
```

### 2.2 张量操作

```python
# 创建NPU张量
tensor = torch.randn(3, 3, device="npu")
tensor = torch.randn(3, 3).npu()

# CPU与NPU转换
cpu_tensor = tensor.cpu()
npu_tensor = cpu_tensor.npu()
```

### 2.3 内存管理

```python
# 显存使用情况
torch.npu.memory_allocated()
torch.npu.memory_reserved()

# 清空缓存
torch.npu.empty_cache()

# 同步
torch.npu.synchronize()
```

## 三、NPU格式转换

NPU支持特殊的内存布局格式以优化算子性能：

```python
from sglang.srt.hardware_backend.npu.utils import npu_format_cast, NPUACLFormat

# 转换为优化格式
tensor = npu_format_cast(tensor, NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
```

## 四、分布式通信（HCCL）

HCCL是华为的集合通信库，类似于NCCL：

```python
import torch.distributed as dist

# NPU上使用HCCL后端
dist.init_process_group(backend="hccl")

# 通信操作
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.all_gather(tensor_list, tensor)
dist.broadcast(tensor, src=0)
```

## 五、NPU特定配置

### 5.1 环境变量

```bash
# NPU设备选择
export ASCEND_DEVICE_ID=0

# HCCL配置
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
```

### 5.2 量化配置

NPU量化使用modelslim：

```bash
# 量化配置
--quantization modelslim
```

## 六、常见问题

### 6.1 显存不足

```python
# 清空缓存
torch.npu.empty_cache()

# 使用量化
--quantization modelslim
```

### 6.2 设备选择

```bash
# 指定NPU设备
export ASCEND_DEVICE_ID=0,1,2,3

# 或在代码中
torch.npu.set_device(0)
```

## 七、参考资源

| 资源 | 链接 |
|------|------|
| torch_npu文档 | https://gitcode.com/Ascend/pytorch |
| NPU算子文档 | https://gitcode.com/Ascend/op-plugin |
| CANN文档 | https://www.hiascend.com/document |
| SGLang NPU实现 | python/sglang/srt/hardware_backend/npu/ |
