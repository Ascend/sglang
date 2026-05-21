# NPU Interface Testing Knowledge Base / NPU 接口测试知识库

## Overview / 概述

This knowledge base documents the comprehensive test coverage for SGLang NPU (Ascend) interface APIs. The test suite validates various API endpoints, parameters, and functionalities on the Ascend NPU backend.

本知识库记录了 SGLang NPU (Ascend) 接口 API 的全面测试覆盖。该测试套件验证了 Ascend NPU 后端上的各种 API 端点、参数和功能。

**Test Directory / 测试目录**: `test/registered/ascend/interface/`

**Test Files / 测试文件**: 8 files / 8 个文件

**Supported Backend / 支持后端**: Ascend NPU (华为昇腾 NPU)

**Test Models / 测试模型**:
- Llama-3.2-1B-Instruct
- Llama-3.1-8B-Instruct
- Qwen3-VL-4B-Instruct
- Qwen3-30B-A3B

---

## Core Parameters / 核心参数

| Parameter / 参数 | Description / 描述 | Test Coverage / 测试覆盖 |
|------------------|-------------------|-------------------------|
| `--attention-backend` | Attention backend (ascend) / 注意力后端 (ascend) | ✅ All test files / 所有测试文件 |
| `--disable-cuda-graph` | Disable CUDA graph / 禁用 CUDA 图 | ✅ Most test files / 大多数测试文件 |
| `--enable-return-hidden-states` | Return hidden states / 返回隐藏状态 | ✅ test_npu_api.py |
| `--tp-size` | Tensor parallelism size / 张量并行大小 | ✅ test_npu_api_encode.py |
| `--is-embedding` | Embedding model mode / 嵌入模型模式 | ✅ test_npu_api_encode.py |
| `--mem-fraction-static` | Static memory fraction / 静态内存比例 | ✅ test_npu_enable_thinking.py, test_npu_matched_stop.py |
| `--tp` | Tensor parallelism / 张量并行 | ✅ test_npu_enable_thinking.py |
| `--reasoning-parser` | Reasoning parser / 推理解析器 | ✅ test_npu_enable_thinking.py |
| `--tool-call-parser` | Tool call parser / 工具调用解析器 | ✅ test_npu_openai_function_calling.py |
| `--max-running-requests` | Max running requests / 最大运行请求数 | ✅ test_npu_matched_stop.py |

---

## Test Function Points / 测试功能点

### 1. Basic API Test / 基础 API 测试 (test_npu_api.py) 🔗

**Test Goal / 测试目标**: Verify that the basic functions of the API interfaces work properly and the returned parameters are consistent with the configurations. / 验证 API 接口的基本功能正常工作，返回的参数与配置一致。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--attention-backend ascend`
- `--enable-return-hidden-states`

**Function Points / 功能点**:
- Health check endpoints (/health, /health_generate, /ping) / 健康检查端点
- Model info retrieval (/model_info) / 模型信息获取
- Server info retrieval (/server_info) / 服务器信息获取
- Load monitoring (/v1/loads) / 负载监控
- Model listing (/v1/models) / 模型列表
- Model details (/v1/models/{model}) / 模型详情
- Text generation (/generate) / 文本生成
- Chat completions (/v1/chat/completions) / 聊天补全
- Completions (/v1/completions) / 补全接口
- Profile management (/start_profile, /stop_profile) / 性能分析管理

**Observable Points / 可观察点**:
- HTTP status code (200) / HTTP 状态码
- Response JSON structure / 响应 JSON 结构
- Model path consistency / 模型路径一致性
- Tokenizer path consistency / 分词器路径一致性
- Generation token count / 生成令牌数
- Logprobs presence / 对数概率存在性
- Hidden states presence / 隐藏状态存在性
- Stream response format / 流式响应格式

---

### 2. Abort Request Test / 请求中止测试 (test_npu_api_abort_request.py) 🔗

**Test Goal / 测试目标**: Verify the functionality of /abort_request API to terminate a running /generate request on Ascend backend. / 验证 /abort_request API 在 Ascend 后端上终止正在运行的 /generate 请求的功能。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--attention-backend ascend`

**Function Points / 功能点**:
- Concurrent request handling / 并发请求处理
- Request abortion mechanism / 请求中止机制
- Thread-safe operations / 线程安全操作

**Observable Points / 可观察点**:
- Request termination success / 请求终止成功
- Thread synchronization / 线程同步
- Response collection / 响应收集

---

### 3. Encode API Test / 编码 API 测试 (test_npu_api_encode.py) 🔗

**Test Goal / 测试目标**: Verify the availability and correctness of the /encode API on Ascend backend with embedding models. / 验证 Ascend 后端上 /encode API 的可用性和正确性（使用嵌入模型）。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--attention-backend ascend`
- `--disable-cuda-graph`
- `--tp-size 2`
- `--is-embedding`

**Function Points / 功能点**:
- Plain text encoding / 纯文本编码
- Input IDs encoding / 输入 ID 编码
- Multimodal encoding (text + image) / 多模态编码（文本 + 图像）

**Observable Points / 可观察点**:
- Response status code / 响应状态码
- Response keys presence / 响应键存在性
- Request ID matching / 请求 ID 匹配
- Meta info structure / 元信息结构

---

### 4. Enable Thinking Test / 启用思考测试 (test_npu_enable_thinking.py) 🔗

**Test Goal / 测试目标**: Testing with the 'enable_thinking' feature enabled/disabled, both streaming and non-streaming input requests successful. / 测试启用/禁用 'enable_thinking' 功能，流式和非流式输入请求都能成功。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--attention-backend ascend`
- `--disable-cuda-graph`
- `--mem-fraction-static 0.95`
- `--tp 2`
- `--reasoning-parser qwen3`

**Function Points / 功能点**:
- Non-streaming with reasoning enabled / 非流式启用推理
- Non-streaming with reasoning disabled / 非流式禁用推理
- Streaming with reasoning enabled / 流式启用推理
- Streaming with reasoning disabled / 流式禁用推理
- Reasoning content separation / 推理内容分离

**Observable Points / 可观察点**:
- reasoning_content presence/absence / reasoning_content 存在/不存在
- content presence in response / 响应中 content 存在性
- Stream chunk structure / 流式块结构
- Delta content parsing / Delta 内容解析

---

### 5. Matched Stop Test / 匹配停止测试 (test_npu_matched_stop.py) 🔗

**Test Goal / 测试目标**: Test configuring 'matched_stop' to different values (string, EOS token, length) correctly identifies it as a stop signal. / 测试将 'matched_stop' 配置为不同值（字符串、EOS 令牌、长度）正确识别为停止信号。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--max-running-requests 10`
- `--attention-backend ascend`
- `--disable-cuda-graph`
- `--mem-fraction-static 0.8`

**Function Points / 功能点**:
- Stop string matching / 停止字符串匹配
- EOS token stop / EOS 令牌停止
- Length-based stop / 基于长度的停止
- /v1/completions interface / /v1/completions 接口
- /v1/chat/completions interface / /v1/chat/completions 接口

**Observable Points / 可观察点**:
- finish_reason value / finish_reason 值
- matched_stop value / matched_stop 值
- Stop condition accuracy / 停止条件准确性

---

### 6. OpenAI Function Calling Test / OpenAI 函数调用测试 (test_npu_openai_function_calling.py) 🔗

**Test Goal / 测试目标**: Verify the correctness of full-scenario OpenAI-style function calling with llama3 and pythonic parsers. / 验证使用 llama3 和 pythonic 解析器的全场景 OpenAI 风格函数调用的正确性。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--attention-backend ascend`
- `--disable-cuda-graph`
- `--tool-call-parser llama3` / `--tool-call-parser pythonic`

**Function Points / 功能点**:
- Function call format validation / 函数调用格式验证
- Multi-turn function calls / 多轮函数调用
- Streaming function calls / 流式函数调用
- Multi-parameter tool_choice / 多参数 tool_choice
- JSON parsing validity / JSON 解析有效性
- Strict mode function calling / 严格模式函数调用
- Required tool choice / 必需工具选择
- Specific tool choice / 特定工具选择
- Multiple choices with tools / 多选项工具调用
- Pythonic tool call format / Pythonic 工具调用格式
- Parallel tool calls / 并行工具调用

**Observable Points / 可观察点**:
- tool_calls presence / tool_calls 存在性
- Function name correctness / 函数名称正确性
- Arguments JSON validity / 参数 JSON 有效性
- finish_reason (tool_calls/stop) / 完成原因
- Streaming chunk integrity / 流式块完整性
- Tool call index presence / 工具调用索引存在性

---

### 7. Ignore EOS Test / 忽略 EOS 测试 (test_npu_openai_server_ignore_eos.py) 🔗

**Test Goal / 测试目标**: Test 'ignore_eos' is True, the EOS is ignored and continue reasoning. / 测试 'ignore_eos' 为 True 时，忽略 EOS 并继续推理。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--attention-backend ascend`
- `--disable-cuda-graph`

**Function Points / 功能点**:
- ignore_eos=False behavior / ignore_eos=False 行为
- ignore_eos=True behavior / ignore_eos=True 行为
- Token count comparison / 令牌数比较

**Observable Points / 可观察点**:
- Generated token count / 生成的令牌数
- finish_reason value / finish_reason 值
- Response length difference / 响应长度差异

---

### 8. Penalty Mechanisms Test / 惩罚机制测试 (test_npu_penalty.py) 🔗

**Test Goal / 测试目标**: Verify successful processing of inference requests with three specific mechanisms (frequency_penalty, presence_penalty, min_new_tokens). / 验证使用三种特定机制（frequency_penalty、presence_penalty、min_new_tokens）成功处理推理请求。

**Test Type / 测试类型**: Integration Test / 集成测试

**Covered Parameters / 覆盖参数**:
- `--attention-backend ascend`
- `--disable-cuda-graph`

**Function Points / 功能点**:
- Default values (no penalty) / 默认值（无惩罚）
- Frequency penalty application / 频率惩罚应用
- Presence penalty application / 存在惩罚应用
- Min new tokens enforcement / 最小新令牌强制执行
- Mixed penalty combinations / 混合惩罚组合
- Concurrent penalty requests / 并发惩罚请求

**Observable Points / 可观察点**:
- Response status code / 响应状态码
- Logprobs structure / 对数概率结构
- Text generation quality / 文本生成质量
- Thread pool execution / 线程池执行

---

## Test File Summary / 测试文件汇总

| # | Test File / 测试文件 | Main Function / 主函数 | Test Type / 测试类型 | Category / 类别 |
|---|---------------------|----------------------|---------------------|-----------------|
| 1 | test_npu_api.py | Basic API validation / 基础 API 验证 | Integration / 集成 | Interface / 接口 |
| 2 | test_npu_api_abort_request.py | Request abortion / 请求中止 | Integration / 集成 | Interface / 接口 |
| 3 | test_npu_api_encode.py | Encoding API / 编码 API | Integration / 集成 | Interface / 接口 |
| 4 | test_npu_enable_thinking.py | Reasoning/thinking / 推理/思考 | Integration / 集成 | Interface / 接口 |
| 5 | test_npu_matched_stop.py | Stop conditions / 停止条件 | Integration / 集成 | Interface / 接口 |
| 6 | test_npu_openai_function_calling.py | Function calling / 函数调用 | Integration / 集成 | Interface / 接口 |
| 7 | test_npu_openai_server_ignore_eos.py | EOS handling / EOS 处理 | Integration / 集成 | Interface / 接口 |
| 8 | test_npu_penalty.py | Penalty mechanisms / 惩罚机制 | Integration / 集成 | Interface / 接口 |

---

## Observable Points Summary / 可观察点汇总

### Server-side Observables / 服务端可观察点
- Server launch status / 服务器启动状态
- Health check responses / 健康检查响应
- Model loading status / 模型加载状态
- Profile directory creation / 性能分析目录创建

### Inference Observables / 推理可观察点
- Response status codes (200, 400, 500) / 响应状态码
- Generated text content / 生成的文本内容
- Token counts (input/output/completion) / 令牌数
- Finish reasons (stop/length/tool_calls) / 完成原因
- Matched stop conditions / 匹配的停止条件
- Logprobs presence and structure / 对数概率存在性和结构
- Hidden states presence / 隐藏状态存在性
- Reasoning content separation / 推理内容分离

### Performance Observables / 性能可观察点
- Request latency / 请求延迟
- Concurrent request handling / 并发请求处理
- Profile trace generation / 性能分析跟踪生成
- Memory usage / 内存使用

### Error Observables / 错误可观察点
- Error response format / 错误响应格式
- Request abortion success / 请求中止成功
- Invalid parameter handling / 无效参数处理

---

## API Endpoints Coverage / API 端点覆盖

| Endpoint / 端点 | Test Files / 测试文件 | Description / 描述 |
|----------------|---------------------|-------------------|
| /health | test_npu_api.py | Health check / 健康检查 |
| /health_generate | test_npu_api.py | Generate health check / 生成健康检查 |
| /ping | test_npu_api.py | Ping check / Ping 检查 |
| /model_info | test_npu_api.py | Model information / 模型信息 |
| /server_info | test_npu_api.py | Server information / 服务器信息 |
| /v1/loads | test_npu_api.py | Load information / 负载信息 |
| /v1/models | test_npu_api.py | Models list / 模型列表 |
| /v1/models/{model} | test_npu_api.py | Model details / 模型详情 |
| /generate | test_npu_api.py, test_npu_penalty.py | Text generation / 文本生成 |
| /v1/chat/completions | test_npu_api.py, test_npu_enable_thinking.py, test_npu_matched_stop.py, test_npu_openai_function_calling.py, test_npu_openai_server_ignore_eos.py | Chat completions / 聊天补全 |
| /v1/completions | test_npu_api.py, test_npu_matched_stop.py | Completions / 补全 |
| /encode | test_npu_api_encode.py | Encoding / 编码 |
| /abort_request | test_npu_api_abort_request.py | Abort request / 中止请求 |
| /start_profile | test_npu_api.py | Start profiling / 开始性能分析 |
| /stop_profile | test_npu_api.py | Stop profiling / 停止性能分析 |

---

## Sampling Parameters Coverage / 采样参数覆盖

| Parameter / 参数 | Test Files / 测试文件 | Description / 描述 |
|-----------------|---------------------|-------------------|
| temperature | test_npu_api.py | Sampling temperature / 采样温度 |
| max_new_tokens | test_npu_api.py, test_npu_penalty.py | Maximum new tokens / 最大新令牌数 |
| max_tokens | test_npu_api.py, test_npu_matched_stop.py, test_npu_openai_function_calling.py | Maximum tokens / 最大令牌数 |
| max_completion_tokens | test_npu_api.py | Maximum completion tokens / 最大补全令牌数 |
| top_p | test_npu_api_encode.py | Top-p sampling / Top-p 采样 |
| top_k | test_npu_api.py | Top-k sampling / Top-k 采样 |
| stream | test_npu_api.py, test_npu_enable_thinking.py, test_npu_openai_function_calling.py | Streaming mode / 流式模式 |
| stop | test_npu_matched_stop.py | Stop sequences / 停止序列 |
| stop_token_ids | test_npu_api.py | Stop token IDs / 停止令牌 ID |
| frequency_penalty | test_npu_penalty.py | Frequency penalty / 频率惩罚 |
| presence_penalty | test_npu_penalty.py | Presence penalty / 存在惩罚 |
| min_new_tokens | test_npu_penalty.py | Minimum new tokens / 最小新令牌数 |
| ignore_eos | test_npu_openai_server_ignore_eos.py | Ignore EOS token / 忽略 EOS 令牌 |
| return_logprob | test_npu_api.py, test_npu_penalty.py | Return logprobs / 返回对数概率 |
| return_hidden_states | test_npu_api.py | Return hidden states / 返回隐藏状态 |
| n | test_npu_openai_function_calling.py | Number of choices / 选择数量 |
| tools | test_npu_openai_function_calling.py | Tool definitions / 工具定义 |
| tool_choice | test_npu_openai_function_calling.py | Tool choice / 工具选择 |
| separate_reasoning | test_npu_enable_thinking.py | Separate reasoning content / 分离推理内容 |
| chat_template_kwargs | test_npu_enable_thinking.py | Chat template kwargs / 聊天模板参数 |

---

## CI Registration Information / CI 注册信息

| Test File / 测试文件 | Suite / 套件 | Estimated Time / 估计时间 | Nightly / 夜间测试 |
|---------------------|-------------|-------------------------|-------------------|
| test_npu_api.py | nightly-npu-a3-merged | 1600s | ✅ |
| test_npu_api_abort_request.py | nightly-1-npu-a3 | 400s | ✅ |
| test_npu_api_encode.py | nightly-1-npu-a3 | 400s | ✅ |
| test_npu_enable_thinking.py | nightly-2-npu-a3 | 400s | ✅ |
| test_npu_matched_stop.py | nightly-1-npu-a3 | 400s | ✅ |
| test_npu_openai_function_calling.py | nightly-1-npu-a3 | 400s | ✅ |
| test_npu_openai_server_ignore_eos.py | nightly-2-npu-a3 | 400s | ✅ |
| test_npu_penalty.py | nightly-1-npu-a3 | 400s | ✅ |

---

## Notes / 备注

1. All tests use the Ascend NPU backend (`--attention-backend ascend`) / 所有测试使用 Ascend NPU 后端
2. Most tests disable CUDA graph (`--disable-cuda-graph`) for compatibility / 大多数测试禁用 CUDA 图以确保兼容性
3. Tests are organized by CI suites for efficient parallel execution / 测试按 CI 套件组织以实现高效并行执行
4. The test suite covers both internal SGLang APIs and OpenAI-compatible APIs / 测试套件覆盖内部 SGLang API 和 OpenAI 兼容 API
5. Function calling tests include both llama3 and pythonic parsers / 函数调用测试包括 llama3 和 pythonic 解析器
