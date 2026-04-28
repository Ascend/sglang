# 基础推理测试方法

## 1. 测试流程

### 1.1 完整测试流程

```
1. 模型加载测试
   └─ 验证模型能正确加载到内存

2. 基础推理测试
   └─ 验证模型能正常生成输出

3. 输出验证
   └─ 验证输出格式和内容正确

4. 性能测试（可选）
   └─ 测量吞吐量和延迟
```

### 1.2 测试优先级

| 优先级 | 测试项 | 说明 |
|--------|--------|------|
| P0 | 模型加载 | 必须通过 |
| P0 | 基础推理 | 必须通过 |
| P1 | 输出验证 | 建议通过 |
| P2 | 性能测试 | 可选 |

---

## 2. SGLang启动命令

### 2.1 基础启动

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --port 30000
```

### 2.2 NPU启动

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --port 30000 \
    --device npu \
    --attention-backend ascend
```

### 2.3 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model-path` | 模型路径 | `/path/to/model` |
| `--port` | 服务端口 | `30000` |
| `--tp` | Tensor Parallel大小 | `2` |
| `--device` | 设备类型 | `npu` |
| `--attention-backend` | Attention后端 | `ascend` |
| `--context-length` | 上下文长度 | `4096` |
| `--max-running-requests` | 最大并发请求数 | `16` |

---

## 3. 测试请求

### 3.1 curl测试

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

### 3.2 Python测试

```python
import requests

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
    }
)
print(response.json())
```

### 3.3 OpenAI客户端测试

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=50,
)
print(response.choices[0].message.content)
```

---

## 4. 测试模板与用例

### 4.1 测试模式

| 模式 | 描述 | 测试用例 | 预计时间(秒) |
|------|------|----------|--------------|
| quick | 快速验证 | TC001, TC002, TC003 | 120 |
| standard | 标准验证 | TC001, TC002, TC003, TC101, TC102 | 300 |
| full | 完整验证 | TC001, TC002, TC003, TC101, TC102, TC201, TC202 | 600 |

### 4.2 测试用例详细定义

#### 基础测试用例（所有模型）

**TC001: 短文本推理**
- 输入："1+1=?"
- 通过条件：输出包含"2"
- 超时：30秒

**TC002: 长文本推理**
- 输入："请写一篇关于人工智能的短文"
- 通过条件：输出长度 >= 50字符
- 超时：60秒

**TC003: 多轮对话**
- 输入："我叫张三->我叫什么名字？"
- 通过条件：输出包含"张三"
- 超时：30秒

#### MoE模型扩展测试

**TC101: 专家路由测试**
- 输入：多样化问题列表
- 通过条件：所有响应正常，无专家路由错误
- 适用模型：MoE, MoE-MLA
- 超时：60秒

**TC102: 负载均衡测试**
- 输入：连续10次推理
- 通过条件：所有请求成功
- 适用模型：MoE, MoE-MLA
- 超时：120秒

#### MLA模型扩展测试

**TC201: 长上下文测试**
- 输入：4K+ token输入
- 通过条件：响应正常，无内存错误
- 适用模型：MLA, MoE-MLA
- 超时：120秒

**TC202: KV Cache测试**
- 输入：多轮长对话
- 通过条件：无KV Cache相关错误
- 适用模型：MLA, MoE-MLA
- 超时：120秒

### 4.3 测试用例请求示例

#### 4.3.1 短文本推理

```json
{
    "model": "default",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 10
}
```

**预期**：模型输出类似 "2" 或 "2." 的简短回答

#### 4.3.2 长文本推理

```json
{
    "model": "default",
    "messages": [{"role": "user", "content": "请写一篇关于人工智能的500字文章"}],
    "max_tokens": 1000
}
```

**预期**：模型输出一篇完整的文章

#### 4.3.3 多轮对话

```json
{
    "model": "default",
    "messages": [
        {"role": "user", "content": "我叫张三"},
        {"role": "assistant", "content": "你好，张三！"},
        {"role": "user", "content": "我叫什么名字？"}
    ],
    "max_tokens": 50
}
```

**预期**：模型回答 "张三"

---

## 5. 错误诊断

### 5.1 启动失败

**现象**：服务无法启动

**排查步骤**：
1. 检查模型路径是否正确
2. 检查设备是否可用
3. 检查内存是否充足
4. 查看错误日志

**常见错误**：
| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Model not found | 模型路径错误 | 检查路径 |
| Out of memory | 内存不足 | 减小TP/BS |
| Device not available | 设备不可用 | 检查设备状态 |

### 5.2 推理失败

**现象**：请求返回错误

**排查步骤**：
1. 检查请求格式是否正确
2. 检查服务日志
3. 检查模型配置

**常见错误**：
| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Invalid request | 请求格式错误 | 检查JSON格式 |
| Context too long | 上下文超长 | 减小输入长度 |
| Internal error | 模型内部错误 | 查看详细日志 |

### 5.3 输出异常

**现象**：输出内容异常（乱码、重复等）

**排查步骤**：
1. 检查模型权重是否正确加载
2. 检查配置是否正确
3. 与HuggingFace输出对比

---

## 6. 日志查看

### 6.1 启动日志

启动时关注以下信息：
```
[INFO] Loading model from /path/to/model
[INFO] Model architecture: Qwen2ForCausalLM
[INFO] Using attention backend: ascend
[INFO] Server started at http://0.0.0.0:30000
```

### 6.2 运行日志

运行时关注：
- 请求处理时间
- 内存使用情况
- 错误和警告信息

---

## 7. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/launch_server.py` | 服务启动入口 |
| `python/sglang/srt/server_args.py` | 服务参数定义 |
| `python/sglang/srt/hardware_backend/npu/` | NPU后端实现 |

