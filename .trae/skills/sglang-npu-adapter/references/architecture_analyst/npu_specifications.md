# NPU规格限制

## 1. 华为昇腾NPU规格

### 1.1 主流型号

| 型号 | HBM | 算力(FP16) | 适用场景 |
|------|-----|-----------|----------|
| Ascend 910B1 | 64GB | 376 TFLOPS | 大模型训练推理 |
| Ascend 910B2 | 64GB | 376 TFLOPS | 大模型训练推理 |
| Ascend 910B3 | 64GB | 376 TFLOPS | 大模型训练推理 |

### 1.2 可用内存估算

```
可用内存 = HBM大小 - 系统预留
可用内存 ≈ 64GB - 10-15GB = 50-55GB
```

---

## 2. 算子支持限制

### 2.1 Attention后端

| 后端 | 支持状态 | 说明 |
|------|----------|------|
| ascend | ✅ 支持 | NPU默认后端 |
| flashinfer | ❌ 不支持 | GPU专用 |
| triton | ❌ 不支持 | GPU专用 |

### 2.2 RoPE实现

| RoPE类型 | 支持状态 |
|----------|----------|
| 标准RoPE | ✅ 支持 |
| Linear Scaling | ✅ 支持 |
| Dynamic NTK | ✅ 支持 |
| Yarn | ⚠️ 部分支持 |
| M-RoPE | ⚠️ 需检查 |

### 2.3 MoE实现

| 功能 | 支持状态 |
|------|----------|
| FusedMoE | ✅ 支持 |
| DeepEP | ⚠️ 需检查HCCL支持 |

---

## 3. 性能特性

### 3.1 内存带宽

| NPU型号 | 内存带宽 |
|---------|----------|
| Ascend 910B | 1.2 TB/s |

### 3.2 通信带宽

| 互联方式 | 带宽 |
|----------|------|
| HCCS | 392 GB/s |
| RoCE | 100 Gbps |

### 3.3 计算吞吐

实际吞吐受多种因素影响：
- 模型架构
- Batch size
- 序列长度
- 量化精度

---

## 4. NPU特定限制

### 4.1 不支持的操作

- 某些GPU专用算子
- 特定CUDA操作
- 部分Triton kernel

### 4.2 精度限制

| 精度 | 支持状态 |
|------|----------|
| FP32 | ✅ 支持 |
| FP16 | ✅ 支持 |
| BF16 | ✅ 支持 |
| FP8 | ⚠️ 部分支持 |
| INT8 | ✅ 支持 |

### 4.3 动态shape限制

NPU对动态shape的支持有限，可能需要padding或固定shape。

---

## 5. 环境要求

### 5.1 软件栈

| 组件 | 版本要求 |
|------|----------|
| CANN | 8.0+ |
| torch_npu | 匹配PyTorch版本 |
| HCCL | 随CANN安装 |

### 5.2 环境变量

```bash
# NPU设备选择
export ASCEND_DEVICE_ID=0

# HCCL配置
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
```

---

## 6. 参考资源

| 资源 | 链接 |
|------|------|
| CANN文档 | https://www.hiascend.com/document |
| torch_npu | https://gitcode.com/Ascend/pytorch |
| SGLang NPU实现 | `python/sglang/srt/hardware_backend/npu/` |
