# NPU特定验证

## 1. NPU功能验证

### 1.1 Ascend后端验证

```bash
# 启动服务
python -m sglang.launch_server \
    --model-path /path/to/model \
    --device npu \
    --attention-backend ascend

# 测试请求
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 1.2 算子正确性验证

| 算子 | 验证方法 |
|------|----------|
| FlashAttention | 对比GPU输出 |
| RoPE | 对比CPU实现 |
| MoE | 对比标准实现 |

### 1.3 内存使用验证

```bash
# 监控NPU内存
watch -n 1 npu-smi info
```

---

## 2. NPU性能验证

### 2.1 与GPU性能对比

```bash
# GPU测试
python -m sglang.bench_one_batch --model-path /path/to/model --device cuda

# NPU测试
python -m sglang.bench_one_batch --model-path /path/to/model --device npu
```

### 2.2 内存效率对比

| 指标 | GPU | NPU |
|------|-----|-----|
| 模型权重 | 基准 | 相近 |
| KV Cache | 基准 | 相近 |
| 激活内存 | 基准 | 可能更高 |

### 2.3 通信效率验证

```bash
# 测试HCCL通信
python -c "
import torch
import torch.distributed as dist
dist.init_process_group(backend='hccl')
# 执行all_reduce测试
...
"
```

---

## 3. NPU稳定性验证

### 3.1 长时间运行测试

```bash
# 持续运行测试
for i in {1..1000}; do
    curl http://localhost:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "default", "messages": [{"role": "user", "content": "Test '$i'"}]}'
done
```

### 3.2 大批量测试

```bash
# 并发请求测试
python -m sglang.bench_one_batch \
    --model-path /path/to/model \
    --device npu \
    --batch-size 64
```

### 3.3 边界条件测试

| 测试项 | 说明 |
|--------|------|
| 最大序列长度 | 测试max_position_embeddings |
| 最大batch | 测试内存极限 |
| 长时间运行 | 测试稳定性 |

---

## 4. 常见问题排查

### 4.1 NPU初始化失败

```bash
# 检查设备状态
npu-smi info

# 检查CANN
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

### 4.2 内存不足

```bash
# 查看内存使用
npu-smi info -t usages

# 清空缓存
python -c "import torch_npu; torch_npu.npu.empty_cache()"
```

### 4.3 通信错误

```bash
# 检查HCCL配置
echo $HCCL_CONNECT_TIMEOUT
echo $HCCL_EXEC_TIMEOUT
```

---

## 5. 参考资源

| 资源 | 说明 |
|------|------|
| `python/sglang/srt/hardware_backend/npu/` | NPU实现 |
| npu-smi | NPU管理工具 |
| CANN文档 | https://www.hiascend.com/document |
