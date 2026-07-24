# NPU 基础知识 (Ascend)

## 设备与软件栈

- **昇腾 NPU**:华为 Ascend 系列,常见 910B / 910C(B 是大部分 PoC 环境)
- **CANN**:Compute Architecture for Neural Networks,相当于 NPU 上的 CUDA
- **ACL**:Ascend Computing Language,运行时 API(对应 CUDA Runtime API)
- **torch_npu**:Ascend 维护的 PyTorch 后端,装上后 `torch.npu.X` 可用,API 表面对齐 `torch.cuda.X`(大部分)

## 检测 NPU

```bash
npu-smi info                              # 类似 nvidia-smi
npu-smi info | grep "Ascend"              # 数 NPU 数量
python -c "import torch_npu; import torch; print(torch.npu.is_available(), torch.npu.device_count())"
```

## 常见 API 对应

| GPU (torch.cuda)              | NPU (torch.npu)              | 备注                       |
|-------------------------------|------------------------------|--------------------------|
| `torch.cuda.is_available()`   | `torch.npu.is_available()`   |                          |
| `torch.cuda.device_count()`   | `torch.npu.device_count()`   |                          |
| `torch.cuda.empty_cache()`    | `torch.npu.empty_cache()`    |                          |
| `torch.cuda.synchronize()`    | `torch.npu.synchronize()`    |                          |
| `tensor.cuda()`               | `tensor.npu()`               |                          |
| `torch.cuda.current_device()` | `torch.npu.current_device()` |                          |
| `torch.cuda.Stream()`         | `torch.npu.Stream()`         | 部分场景行为不同 |

**注意:torch_npu 不是 100% 镜像**,以下有差异:
- `torch.cuda.amp.autocast` → NPU 用 `torch.npu.amp.autocast`,且支持的 dtype 子集略小
- `torch.cuda.graphs` → NPU 对应 `aclgraph`(独立 API,非 drop-in)
- 部分 fused kernel 在 NPU 上没有对应实现(flash_attn / paged_attention 的 CUDA 实现需替换为 torch-native 或 NPU 专用算子)

## 在 SGLang 代码里检测 NPU

```python
from sglang.srt.utils import is_npu, is_cuda

if is_npu():
    # NPU 分支
    ...
elif is_cuda():
    # GPU 分支
    ...
```

或:

```python
import torch
device_type = torch.npu.is_available() and "npu" or "cuda"
```

## 启动命令差异

```bash
# GPU
python -m sglang.launch_server --model-path ... --tp 8

# NPU
python -m sglang.launch_server --model-path ... --tp 8 --device npu
```

注意一些 GPU-only flag 在 NPU 上不可用(如 `--enable-flashinfer`,因为 flashinfer 是 GPU 库)。

## 常见坑

1. **dtype 默认值**:NPU 上 bf16 是首选;fp16 在某些算子上会有数值差异
2. **kv-cache 内存**:NPU 显存碎片化比 GPU 更严重,适当调小 `--mem-fraction-static`(0.85 → 0.8)
3. **PYTHONPATH**:`export PYTHONPATH=${PWD}/python:$PYTHONPATH` 必须先设,否则会用到老版本 sglang
4. **不要升级 transformers**:torch_npu 对 transformers 版本有兼容性约束;盲升会破坏精度

## 关键命令速查

```bash
# 看 NPU 用谁占着
npu-smi info -t process

# 看显存使用
npu-smi info | grep -A1 "Ascend"

# 杀进程(只杀自己的!)
kill $(cat {WORKSPACE_DIR}/logs/server.pid)
sleep 3
kill -9 $(cat {WORKSPACE_DIR}/logs/server.pid) 2>/dev/null
```
