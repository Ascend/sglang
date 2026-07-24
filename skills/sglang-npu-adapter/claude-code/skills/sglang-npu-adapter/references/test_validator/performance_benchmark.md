# 性能基准测试

## 1. 性能指标

### 1.1 核心指标

| 指标 | 说明 | 单位 |
|------|------|------|
| TTFT | Time To First Token | ms |
| 吞吐量 | tokens/s | tok/s |
| 延迟 | 单请求响应时间 | ms |
| 内存占用 | GPU/NPU内存使用 | GB |

### 1.2 指标关系

```
吞吐量 = batch_size × output_len / 总时间
TTFT = 首token生成时间
延迟 = TTFT + (output_len - 1) × inter_token_latency
```

---

## 2. 测试工具

### 2.1 SGLang Benchmark

```bash
python -m sglang.bench_one_batch \
    --model-path /path/to/model \
    --batch-size 16 \
    --input-len 1024 \
    --output-len 128
```

### 2.2 自定义测试脚本

```python
import time
import requests

def benchmark(model_url, prompts, max_tokens=100):
    times = []
    for prompt in prompts:
        start = time.time()
        response = requests.post(
            f"{model_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
        )
        end = time.time()
        times.append(end - start)
    
    return {
        "avg_latency": sum(times) / len(times),
        "throughput": len(prompts) * max_tokens / sum(times),
    }
```

---

## 3. 测试配置

### 3.1 Batch Size范围

| 配置 | 说明 |
|------|------|
| 1 | 单请求延迟测试 |
| 4-8 | 小批量测试 |
| 16-32 | 中等批量测试 |
| 64+ | 大批量吞吐测试 |

### 3.2 序列长度范围

| 配置 | 说明 |
|------|------|
| 128 | 短序列 |
| 1024 | 中等序列 |
| 4096 | 长序列 |
| 8192+ | 超长序列 |

### 3.3 并行配置

| 配置 | 说明 |
|------|------|
| TP=1 | 单卡 |
| TP=2 | 双卡 |
| TP=4 | 四卡 |
| TP=8 | 八卡 |

---

## 4. 结果分析

### 4.1 性能瓶颈识别

| 现象 | 可能瓶颈 |
|------|----------|
| TTFT高 | 模型加载、prefill慢 |
| 吞吐量低 | decode慢、内存带宽 |
| 内存不足 | KV Cache过大 |

### 4.2 优化建议

| 问题 | 优化方案 |
|------|----------|
| TTFT高 | 增大chunked prefill |
| 吞吐量低 | 增大batch、优化attention |
| 内存不足 | 量化、减小KV Cache |

---

## 5. NPU性能对比

### 5.1 与GPU对比

| 指标 | GPU (A100) | NPU (910B) |
|------|------------|------------|
| TTFT | 基准 | 约1.2x |
| 吞吐量 | 基准 | 约0.8-1.0x |

### 5.2 NPU优化要点

- 使用ascend attention后端
- 合理配置HCCL
- 注意内存管理

---

## 6. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/bench_one_batch.py` | 批量测试 |
| `benchmark/` | 基准测试目录 |
