# 基础推理测试模板

## OpenAI 兼容接口冒烟测试

SGLang server 默认在 `localhost:8000` 暴露 OpenAI 兼容 API。

### Completions 请求(无 chat template)

```bash
curl -s -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "default",
        "prompt": "The capital of France is",
        "max_tokens": 16,
        "temperature": 0
    }' | jq .
```

期望响应:
```json
{
  "id": "...",
  "choices": [
    {"text": " Paris.", "index": 0, "finish_reason": "stop"}
  ],
  "usage": {...}
}
```

判定:`choices[0].text` 非空 + 不含 `"<unk>" * N` / 乱码 / 同一 token 重复。

### Chat completions(有 chat template)

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "default",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        "max_tokens": 32,
        "temperature": 0
    }' | jq .
```

期望:`choices[0].message.content` 含 "4" 或类似数字。

## 健康检查请求

```bash
# server 已启动?
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models
# 期望 200

# 模型列表
curl -s http://localhost:8000/v1/models | jq '.data[].id'
```

## 多模态(VLM)请求

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "default",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/test.jpg"}}
            ]
        }],
        "max_tokens": 64
    }' | jq .
```

如果模型不是 VLM,会返回错误。VLM 必须能正确响应图文混合请求才算 passed。

## Batch 推理

```bash
# 用 bench_serving 跑 batch
python -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 8 \
    --random-input-len 256 \
    --random-output-len 64
```

输出含 throughput / latency,记到 capacity_baseline。

## 失败判定规则

| 现象 | 判定 |
|------|------|
| HTTP 5xx | failed |
| 响应 timeout > 30s(short prompt) | failed |
| 响应 token 全是 `<unk>` 或一个字符重复 | failed(可能精度问题,标 precision_suspect) |
| 响应是空字符串 / 只有 EOS | failed |
| 响应内容看起来合理(语法正确,主题相关) | passed |

**注意 dummy 权重的特殊情况**:`--load-format dummy` 时模型用随机权重,输出本来就乱。**dummy 验证的标准是"server 启起来 + 不报错 + 能返回响应"**,不是输出合理。
