# NPU 参数测试设计文档

## 1. Model and Tokenizer 特性

### 特性概述

Model and Tokenizer 特性负责模型加载和 token 转换。

```
用户请求(text) → [Tokenizer] → token_ids → [Engine] → token_ids → [Detokenizer] → text
```

涉及参数：

| 类别 | 参数 |
|---|---|
| Tokenizer 选型 | `--tokenizer-backend`、`--tokenizer-mode` |
| 并行度控制 | `--tokenizer-worker-num`、`--detokenizer-worker-num` |
| 模型加载 | `--model-config-parser`、`--model-impl` |

本次覆盖 `--tokenizer-backend`、`--model-config-parser`。`--detokenizer-worker-num` 已识别，GPU 移植另任务完成。

---

### 逐参数分析

#### 2.1 `--tokenizer-backend`

##### 2.1.1 业务理解

**定义** | `server_args.py:434-442`，控制 tokenizer 底层库选型

**取值** |

| 值 | 含义 |
|---|---|
| `"huggingface"` (默认) | 使用 HuggingFace 原生 tokenizers 库（`AutoTokenizer.from_pretrained`） |
| `"fastokens"` | 通过猴子补丁 `_ensure_fastokens_patched()` 将底层替换为 fastokens 库 |

**作用链路** (`tokenizer.py:459-476`)

```
tokenizer_backend="huggingface" → AutoTokenizer.from_pretrained(model) → 返回标准 tokenizer
tokenizer_backend="fastokens"   → _ensure_fastokens_patched() 补丁 transformers
                                → AutoTokenizer.from_pretrained(model)
                                → tokenizer._tokenizer 替换为 _TokenizerShim 实例
```

**依赖**

| 类型 | 内容 |
|---|---|
| 下游 | TokenizerManager、DetokenizerManager 的 tokenizer 初始化 |
| 关联参数 | `--tokenizer-mode`（fast/slow 模式，与 backend 正交） |
| 前置条件 | fastokens 需 `pip install fastokens` |

##### 2.1.2 通俗理解

Tokenizer 把人类语言翻译成模型能理解的数字。`--tokenizer-backend` 选择哪个翻译官。

- **`huggingface`** — 官方翻译官，最稳定，所有模型可用
- **`fastokens`** — 快手翻译官，高并发时更快，需额外安装

**什么时候需要改**: 高并发推理时切换 fastokens 可降低 tokenization 延迟。

##### 2.1.3 GPU 社区用例分析

| 文件 | 类型 | 参数角色 | 覆盖值 | 测试内容 | 可移植 |
|---|---|---|---|---|---|
| `test/registered/unit/utils/test_hf_transformers_fastokens.py` | UT (CPU) | 被测试对象 | `"fastokens"` | T1: 验证 fastokens 后端注入后 `tokenizer._tokenizer` 为 `_TokenizerShim` 实例。T2: 验证 encode→decode 回环正确。`@unittest.skipUnless(HAS_FASTOKENS)` | ✅ 移植，去 skip |

##### 2.1.4 E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `tokenizer_backend` | `"huggingface"` | 走 `AutoTokenizer.from_pretrained(model)`，标准路径，所有不指定该参数的测试都默认走此路径 |
| | `"fastokens"` | 走 `_ensure_fastokens_patched()` 猴子补丁 → `AutoTokenizer.from_pretrained(model)` → `tokenizer._tokenizer` 替换为 `_TokenizerShim`，与 huggingface 是完全不同的代码路径 |
| 模型类型 | Llama / Qwen | tokenizer class 由 auto-map 决定，不影响 backend 选择逻辑 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 默认后端 | 用户不关心 tokenizer 选型，直接启动服务推理，走 HuggingFace 标准 tokenizer | `huggingface` (默认) | 所有不传 `--tokenizer-backend` 的 NPU 测试均默认走此路径，已隐式覆盖。不单独新增 E2E——因为显式传 `huggingface` 与隐式默认走的是同一条代码路径，测了不增加覆盖 |
| fastokens 替换后端 | 用户需降低高并发场景下 tokenization 延迟，指定 fastokens 后端替换标准实现 | `fastokens` | ① UT 验证 `_TokenizerShim` 注入成功（T1）；② UT 验证 encode→decode 回环正确（T2）；③ E2E 验证 server 启动 + 推理正确（T3）

##### 2.1.5 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_fastokens_shim_is_applied_npu` | 验证 fastokens 后端注入 `_TokenizerShim` | UT | 注入验证 | 移植 GPU `test/registered/unit/utils/test_hf_transformers_fastokens.py` | P1 |
| T2 | `test_fastokens_encode_decode_roundtrip_npu` | 验证 fastokens encode→decode 回环正确 | UT | 回环验证 | 移植 GPU `test/registered/unit/utils/test_hf_transformers_fastokens.py` | P1 |
| T3 | `test_tokenizer_backend_concurrent` | 并发场景下 fastokens tokenization 延迟低于 huggingface，验证核心加速价值 | E2E | 性能对比 | 新增 | P0 |

**T001–T002** (移植) — 移植 GPU 对应方法，去 `@unittest.skipUnless`，`register_cpu_ci` → `register_npu_ci`。

---

**T003: `test_tokenizer_backend_concurrent`** (新增)

**验证目标**: 高并发场景下 `fastokens` 的 tokenization 延迟低于 `huggingface`，证明 fastokens 在高并发时的加速价值。

**测试步骤**:
1. 以 `--tokenizer-backend huggingface` 启动 Llama 3.2 1B server，并发发送 N 个 `/generate` 请求，记录 tokenization 平均延迟
2. 重启 server，以 `--tokenizer-backend fastokens` 并发发送同量请求，记录 tokenization 平均延迟

**断言**: 所有请求 `status_code == 200` 且 `response.text` 含 `"Paris"`；fastokens 平均延迟 < huggingface 平均延迟

---

#### 2.2 `--detokenizer-worker-num`

> **范围说明**: GPU 已有完整 E2E 用例，移植工作另任务完成，本次不创建文件。以下分析供移植任务参考。

##### 2.2.1 业务理解

**定义** | `server_args.py:444`，控制 detokenizer 进程并行度

**取值** |

| 值 | 含义 |
|---|---|
| `1` (默认) | 单进程 DetokenizerManager，所有请求串行处理 |
| `> 1` (如 4) | N 个 DetokenizerManager worker + 1 个 MultiDetokenizerRouter，按 key hash 分发 |

**作用链路** (`engine.py:720-740`)

```
detokenizer_worker_num <= 1 → 单进程，直接监听 detokenizer_ipc_name
detokenizer_worker_num > 1  → N 个 worker（独立 IPC）+ Router 分发
                              Router 按 http_worker_ipc hash 路由到对应 worker
```

##### 2.2.2 通俗理解

Detokenizer 把 token ids 翻译回人类语言。`--detokenizer-worker-num` 控制多少个解码员同时工作。`=1` 一个解码员串行处理，`>1` 多个解码员并行处理。高并发时增大该值可减少排队等待。

##### 2.2.3 GPU 社区用例分析

| 文件 | 类型 | 参数角色 | 覆盖值 | 测试内容 | 可移植 |
|---|---|---|---|---|---|
| `test/registered/tokenizer/test_multi_detokenizer.py` | E2E (CUDA+AMD) | 被测试对象 | `4` | `--detokenizer-worker-num=4 --tokenizer-worker-num=8`，100 prompts benchmark (`random`, input_len=4096, output_len=2048, request_rate=1)，验证 E2E 延迟 <11000ms, TTFT <86ms, ITL <10ms。MMLUMixin 精度≥0.65 | ✅ 移植适配 |

##### 2.2.4 E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `detokenizer_worker_num` | `1` | engine.py:720 走单进程分支，直接创建 DetokenizerManager 监听原始 IPC |
| | `> 1` | engine.py:733 走多 worker 分支，创建 N 个 DetokenizerManager + MultiDetokenizerRouter，hash 分发请求 |
| 并发量 | 低/高 | 多 worker 价值在高并发时体现，低并发下 Router 分发无实际并行收益 |
| `skip_tokenizer_init` | True | 强制 `detokenizer_worker_num=1`，与多 worker 互斥 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 高并发部署 | 用户需多 detokenizer 并行降低排队延迟 | `> 1` (如 4) | server 启动成功 + benchmark 延迟在合理范围 |

默认值 `=1` 已被所有不指定该参数的测试隐式覆盖。`test_npu_model_tokenizer.py::TestNpuSkipTokenizerInit` 额外覆盖了 `skip_tokenizer_init=True` 强制 `=1` 的路径。

##### 2.2.5 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_multi_detokenizer_ttft_npu` | 多 worker detokenizer 配置下 benchmark 推理延迟 | E2E | 边界/性能 | 移植适配 GPU `test/registered/tokenizer/test_multi_detokenizer.py` | P1 |

---

#### 2.3 `--model-config-parser`

##### 2.3.1 业务理解

**定义** | `server_args.py:510-520`，控制模型配置加载策略

**取值** |

| 值 | 含义 |
|---|---|
| `"auto"` (默认) | 自动选择：模型名含 "Mistral" → mistral 解析器；否则 → hf 解析器 |
| `"hf"` | 强制走 `AutoConfig.from_pretrained(config.json)`，忽略模型名启发式 |

**解析链路** (`config.py:215-248`)

```
"auto" → is_mistral_model() 判断模型名
         → 含 "Mistral" → 走 mistral 专用解析器
         → 不含 → 走 hf 通用解析器（AutoConfig.from_pretrained）
"hf"   → 跳过启发式判断，直接 AutoConfig.from_pretrained(config.json)
```

##### 2.3.2 通俗理解

每个模型有份 config.json 说明书，`--model-config-parser` 选择谁来解读。

- **`auto`** — 自动选。Mistral 模型派专用阅读器，其他模型派通用阅读器
- **`hf`** — 强制用 HuggingFace 通用阅读器。当 auto 选错导致加载失败时手动指定

**用户什么时候需要改**: 默认 `auto` 对绝大多数模型都正确。只在模型加载报 config 错误时需要手动指定 `hf`。

##### 2.3.3 GPU 社区用例分析

| 文件 | 类型 | 参数角色 | 覆盖值 | 测试内容 | 可移植 |
|---|---|---|---|---|---|
| `test/registered/unit/configs/test_model_config_parser_registry.py` | UT (CPU) | 被测试对象 | 注册表 API | T1: register→get 回环正确。T2: 非子类注册抛 ValueError。T3: 未知名抛 ValueError 含已注册列表。setUp/tearDown 隔离全局注册表 | ✅ 移植 |

##### 2.3.4 E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `model_config_parser` | `"auto"` | 走 `is_mistral_model()` 启发式判断。模型名含 "Mistral" 时选 mistral 解析器（读取 Mistral config 特有字段），不含时退回 hf（等同于 `AutoConfig.from_pretrained`） |
| | `"hf"` | 跳过启发式判断，直接调用 `AutoConfig.from_pretrained(config.json)` 解析标准 HuggingFace config |
| 模型名 | 含 "Mistral" | auto 和 hf 走**不同解析器**，覆盖两条不同代码路径 |
| | 不含 | auto → 退回 hf，两条路最终都是 `AutoConfig.from_pretrained`，路径相同 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 模型 | 验证点 |
|---|---|---|---|---|
| Mistral 默认加载 | 用户使用 Mistral 模型，不指定 parser，依赖 auto 自动选 mistral 解析器正确加载 | `auto` | Mistral 7B | 模型加载成功 + 推理正确 |
| 强制通用解析器 | 非标准 Mistral 变体 auto 选错了解析器，用户显式指定 hf 退回通用解析器 | `hf` | Mistral 7B | hf 覆盖与 auto→mistral 不同的解析路径，推理正确 |

##### 2.3.5 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_register_then_get_roundtrip_npu` | 验证注册→获取回环，返回正确实例 | UT | 注册验证 | 移植 GPU `test/registered/unit/configs/test_model_config_parser_registry.py` | P1 |
| T2 | `test_register_rejects_non_subclass_npu` | 验证非子类注册抛 ValueError | UT | 异常验证 | 移植 GPU `test/registered/unit/configs/test_model_config_parser_registry.py` | P1 |
| T3 | `test_unknown_name_raises_with_registered_list_npu` | 验证未知名报错含已注册列表 | UT | 异常验证 | 移植 GPU `test/registered/unit/configs/test_model_config_parser_registry.py` | P1 |
| T4 | `test_model_config_parser_auto` | auto 在 Mistral 上路由到 mistral 解析器，推理成功 | E2E | 正常路径 | 新增 | P0 |
| T5 | `test_model_config_parser_hf` | hf 覆盖 auto 路由走 AutoConfig，推理成功 | E2E | 边界 | 新增 | P0 |

**T001–T003** (移植) — 去 `register_cpu_ci`，保留 setUp/tearDown 注册表隔离。

---

**T004: `test_model_config_parser_auto`** (新增)

**验证目标**: `--model-config-parser auto` 在 Mistral 模型上经 `is_mistral_model()` 判断后路由到 mistral 解析器，模型加载成功、推理正确。

**测试步骤**: 1. `--model-config-parser auto` 启动 Mistral 7B server；2. `/generate` 请求

**断言**: `status_code == 200`；`response.text` 含 `"Paris"`

---

**T005: `test_model_config_parser_hf`** (新增)

**验证目标**: `--model-config-parser hf` 在 Mistral 模型上跳过 auto 的启发式判断，直接走 `AutoConfig.from_pretrained`，覆盖与 auto 不同的解析路径。

**测试步骤**: 1. `--model-config-parser hf` 启动 Mistral 7B server；2. `/generate` 请求

**断言**: `status_code == 200`；`response.text` 含 `"Paris"`

---

#### 2.4 `--load-format`

##### 2.4.1 业务理解

**定义** | `server_args.py`，控制模型权重加载格式。GGUF 路径强制 `model_config_parser=hf`。

**取值** |

| 值 | 含义 |
|---|---|
| `"auto"` (默认) | 自动检测，GGUF 文件自动识别为 `gguf` |
| `"gguf"` | 显式指定 GGUF 量化格式，强制 hf config parser |

##### 2.4.2 通俗理解

GGUF 是量化压缩格式，文件小、省显存。传 `--load-format gguf` 告诉框架按 GGUF 格式读取权重。

##### 2.4.3 GPU 社区用例分析

GPU 社区无 `--load-format` 的独立 E2E 测试。

##### 2.4.4 E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `load_format` | `gguf` | 走 GGUF 加载路径，强制 `model_config_parser=hf` |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| GGUF 量化加载 | 用户使用 GGUF 量化模型节省显存，显式指定格式加载 | `gguf` | server 正常加载 GGUF 模型 + 推理成功 |

##### 2.4.5 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_load_format_gguf` | `--load-format gguf` 加载 GGUF 模型 + 推理成功 | E2E | 新增 | P0 |

**T001: `test_load_format_gguf`** (新增)

**验证目标**: `--load-format gguf` 显式指定 GGUF 格式加载 Qwen3 4B GGUF 量化模型，推理成功。

**测试步骤**:
1. `--load-format gguf` 启动 server，模型路径指定 GGUF 权重
2. `/generate` 请求: `"The capital of France is"`, `temperature=0, max_new_tokens=32`

**断言**: `status_code == 200`；`response.text` 含 `"Paris"`

---

### 最终结论

#### 文件清单

```
test/registered/ascend/basic_function/model_tokenizer/
├── test_npu_model_tokenizer.py             (已有，追加 --load-format gguf)
├── test_npu_model_tokenizer_multimodal.py  (已有)
├── test_npu_openai_embedding.py            (已有)
├── test_npu_tokenizer_backend.py           (新建: --tokenizer-backend)
└── test_npu_model_config_parser.py         (新建: --model-config-parser)
```

#### CI 注册汇总

| 参数 | 来源 | 文件 | CI Suite |
|---|---|---|---|
| `--tokenizer-backend` | 移植 GPU UT + 新增 E2E | `test_npu_tokenizer_backend.py` | `full-1-npu-a3` |
| `--detokenizer-worker-num` | 移植 GPU E2E | `test/registered/tokenizer/test_multi_detokenizer.py` | — |
| `--model-config-parser` | 移植 GPU UT + 新增 E2E | `test_npu_model_config_parser.py` | `full-1-npu-a3` |
| `--load-format` | 在已有文件追加 gguf 值的 E2E 验证 | `test_npu_model_tokenizer.py` | `full-1-npu-a3` |

---

## 2. Optimization/debug options 特性

### 特性概述

Optimization/debug options 覆盖 CUDA Graph、Attention 优化、Context Parallel、NCCL 预热、Torch Compile 等调试和性能调优参数。

以下参数已有 NPU E2E 测试（位于 `test/registered/ascend/basic_function/optimization_debug_options/`），测试模式为：传参 → 启动 server → 请求 generate → 断言 200+Paris。

---

### 逐参数分析

#### 2.1 `--cuda-graph-config`

##### 业务理解

- **定义**: `server_args.py:1979-1983`，JSON 格式统一配置 decode/prefill 阶段的 CUDA Graph 行为
- **核心功能**: 最高优先级。JSON 设置的字段会锁定，独立参数 无法覆盖
- **优先级**（同时指定多个时，排前面的生效）: `--cuda-graph-config` JSON 配置 > `--cuda-graph-backend-decode` 等独立参数 > `--disable-decode-cuda-graph` 等旧别名 > 默认值
- **取值**: JSON 字符串，如 `{"decode":{"backend":"full"}}`

##### 通俗理解

`--cuda-graph-config` 是 CUDA Graph 的"总控面板"。同时设了独立参数 和 JSON 时，JSON 说了算。

##### GPU 社区用例分析

6 个 UT 文件引用（`test_server_args.py`、`test_deepseek_v4.py`、`test_self_unit_capacities.py` 等），均为内部参数传递，**无 GPU E2E 测试**。

##### E2E 测试分析

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `cuda_graph_config` | `{"decode":{"backend":"full"}}` | JSON 解析后覆盖 cuda_graph_config.decode.backend 为 full，同时锁定该字段 |
| 配合独立参数 | `--cuda-graph-backend-decode disabled` | 被 JSON 覆盖为 full，验证 JSON 优先级 |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| JSON 覆盖独立参数 | 用户通过 JSON 统一管理 graph 配置，JSON 优先级高于独立参数 | `{"decode":{"backend":"full"}}` + 独立参数 disabled | JSON 生效（非 disabled），推理成功 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_cuda_graph_config_override` | JSON `{"decode":{"backend":"full"}}` 覆盖独立参数 disabled | E2E | 优先级验证 | 新增 | P1 |

**T001: `test_cuda_graph_config_override`** (新增)

**验证目标**: `--cuda-graph-config` JSON 配置优先级高于独立参数，JSON 中设置 `decode.backend=full` 覆盖独立参数 `--cuda-graph-backend-decode disabled`。

**测试步骤**:
1. 同时传 `--cuda-graph-backend-decode disabled` 和 `--cuda-graph-config '{"decode":{"backend":"full"}}'` 启动 Llama 3.2 1B server，捕获 stderr 日志
2. `/generate` 请求: `"The capital of France is"`, `temperature=0, max_new_tokens=32`

**断言**: `status_code == 200`；`response.text` 含 `"Paris"`；stderr 日志含 `cuda_graph_config` 解析记录，证明 JSON 被解析并覆盖了独立参数

---

#### 2.2 `--cuda-graph-backend-decode`

##### 业务理解

- **定义**: `server_args.py:1986-1992`，decode 阶段 CUDA Graph 编译后端
- **取值**:

| 值 | 含义 |
|---|---|
| `full` (默认) | 完整 CUDA Graph，一次编译覆盖所有 batch size |
| `disabled` | 跳过 graph，直接 eager 推理 |
| `breakable` | 可打断 graph，支持动态 batch size 变化 |
| `tc_piecewise` | 分段 torch.compile 编译，逐段捕获 graph |

- **折叠规则**: 合并入 `cuda_graph_config.decode.backend`，被 `--cuda-graph-config` JSON 覆盖
- **别名**: `--disable-decode-cuda-graph` = `disabled`

##### 通俗理解

decode 阶段是模型逐 token 生成的过程。`--cuda-graph-backend-decode` 控制用什么策略编译 graph 来加速这个过程。`full` 最完整，`breakable` 支持动态 batch，`tc_piecewise` 用分段编译，`disabled` 不加速。

##### GPU 社区用例分析

| 文件 | 类型 | 覆盖值 | 测试内容 |
|---|---|---|---|
| `test_no_extra_forked_cuda_context.py` | E2E | `disabled` | 验证禁用 decode graph 后不创建多余 CUDA context fork — **副作用验证模式** |
| `test_kimi_k25_mxfp4_bcg_mi35x.py` | E2E | `full` | Kimi K25 + full graph + GSM8K — **模型专项验证模式** |

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `cuda_graph_backend_decode` | `full` | 完整 CUDA Graph 编译 decode，一次编译覆盖所有 batch size。GPU 有 Kimi K25 + GSM8K 模型专项验证 |
| | `disabled` | API 层跳过 graph，decode 直接 eager 推理。GPU 有副作用验证（CUDA context fork 不增加） |
| | `breakable` | 可打断 graph，编译时预留动态 batch size 切换能力。GPU 未覆盖此值 |
| | `tc_piecewise` | 分段 torch.compile，逐段捕获而非一次性全图。GPU 未覆盖此值 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 完整加速 | 用户使用默认配置，decode 走完整 graph 获得最佳性能 | `full` | 完整 graph 编译成功 + 推理正确 |
| 禁用加速 | 用户调试或适配特殊模型，需绕过 graph 直接 eager 推理 | `disabled` | 跳过 graph + eager 推理成功 + CUDA context fork 数正常（GPU 副作用验证） |
| 动态 batch | 用户需运行时动态调整 batch size，使用可打断 graph | `breakable` | 双层验证：stderr 日志证明 breakable 编译生效 + MMLU 精度不退化 |
| 分段编译 | 用户模型过大无法一次编译完整 graph，需分段 torch.compile | `tc_piecewise` | 双层验证：stderr 日志证明 tc_piecewise 编译生效 + MMLU 精度不退化 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_decode_backend_breakable` | breakable 编译生效 + MMLU 精度不退化 | E2E | 新增（参考 GPU 双层验证模式） | P1 |
| T2 | `test_decode_backend_tc_piecewise` | tc_piecewise 编译生效 + MMLU 精度不退化 | E2E | 新增（参考 GPU 双层验证模式） | P1 |

---

**T001: `test_decode_backend_breakable`** (新增)

**验证目标**: `--cuda-graph-backend-decode=breakable` 下推理精度不退化，参照 GPU `test_breakable_cuda_graph.py` 双层验证模式（机制验证 + 精度基准）。

**测试步骤**:
1. `--cuda-graph-backend-decode breakable` 启动 Llama 3.2 1B server，捕获 stderr
2. 运行 MMLU 精度 benchmark（参考 `MMLUMixin`，num_examples=64, num_threads=32）
3. 验证 stderr 日志

**断言**: stderr 日志含 breakable 编译记录，证明 breakable 后端生效；MMLU 精度 ≥ 基准值

---

**T002: `test_decode_backend_tc_piecewise`** (新增)

**验证目标**: `--cuda-graph-backend-decode=tc_piecewise` 下推理精度不退化，参照 GPU 双层验证模式。

**测试步骤**:
1. `--cuda-graph-backend-decode tc_piecewise` 启动 Llama 3.2 1B server，捕获 stderr
2. 运行 MMLU 精度 benchmark（参考 `MMLUMixin`，num_examples=64, num_threads=32）
3. 验证 stderr 日志

**断言**: stderr 日志含 tc_piecewise 编译记录，证明分段编译后端生效；MMLU 精度 ≥ 基准值

---

#### 2.3 `--cuda-graph-backend-prefill`

##### 业务理解

- **定义**: `server_args.py:1993-1999`，prefill 阶段 CUDA Graph 编译后端
- **取值**:

| 值 | 含义 |
|---|---|
| `disabled` | 跳过 graph，直接 eager |
| `breakable` | 可打断 graph，支持动态 batch |
| `tc_piecewise` | 分段 torch.compile 编译 |

> 别名 `--disable-prefill-cuda-graph` = `disabled`，走相同代码路径。

##### 通俗理解

prefill 阶段一次性处理整个输入 prompt。`--cuda-graph-backend-prefill` 控制 prefill 用什么策略编译加速。

##### GPU 社区用例分析

| 文件 | 类型 | 覆盖值 | 测试内容 |
|---|---|---|---|
| `test_no_extra_forked_cuda_context.py` | E2E | `disabled` | 同时传 prefill+decode disabled，验证 CUDA context fork 数正常 |
| `test_breakable_cuda_graph.py` | E2E | `breakable` | 专门测 breakable graph 捕获和推理正确性 — **双层验证**（GSM8K 精度 ≥0.80） |
| `test_bcg_with_speculative_decoding.py` | E2E | `breakable` | breakable + EAGLE3 共存 — **组合验证** |

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `cuda_graph_backend_prefill` | `disabled` | prefill 跳过 graph 直接 eager。GPU 有副作用验证（CUDA context fork 数正常） |
| | `breakable` | prefill 走可打断 graph。GPU 有双层验证（GSM8K 精度 ≥0.80）和组合验证（+EAGLE3） |
| | `tc_piecewise` | prefill 走分段 torch.compile。GPU 未覆盖此值 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 禁用 prefill 加速 | 用户调试或适配特殊模型，绕过 prefill graph | `disabled` | 跳过 graph + eager prefill 成功 + CUDA context fork 正常 |
| 动态 batch prefill | 用户需运行时动态调整 prefill batch size | `breakable` | breakable graph 编译成功 + 推理正确 + GSM8K 精度不退化 |
| 分段 prefill 编译 | 用户模型过大无法一次编译完整 prefill graph | `tc_piecewise` | 双层验证：stderr 日志证明 tc_piecewise 编译生效 + MMLU 精度不退化 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_prefill_backend_tc_piecewise` | tc_piecewise 分段编译生效 + MMLU 精度不退化 | E2E | 新增（参考 GPU 双层验证模式） | P1 |

---

**T001: `test_prefill_backend_tc_piecewise`** (新增)

**验证目标**: `--cuda-graph-backend-prefill=tc_piecewise` 分段编译后端下推理精度不退化，参照 GPU `test_breakable_cuda_graph.py` 双层验证模式（机制验证 + 精度基准）。

**测试步骤**:
1. `--cuda-graph-backend-prefill tc_piecewise` 启动 Llama 3.2 1B server，捕获 stderr
2. 运行 MMLU 精度 benchmark（参考 `MMLUMixin`，num_examples=64, num_threads=32）
3. 验证 stderr 日志

**断言**: stderr 日志含 tc_piecewise 编译记录，证明分段编译后端生效；MMLU 精度 ≥ 基准值

---

#### 2.4 `--cuda-graph-max-bs-decode`

##### 业务理解

- **定义**: `server_args.py:2000-2003`，decode 阶段 CUDA Graph 最大 batch size
- **取值**: `Optional[int]`，默认 `None`（系统根据显存自动计算）
- **折叠**: 合并入 `cuda_graph_config.decode.max_bs`

##### 通俗理解

手动限制 decode graph 捕获的最大 batch size。默认自动算，传了就用你指定的值。

##### GPU 社区用例分析

8 个文件引用此参数，均为启动参数传递，**无独立 E2E 测试**。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `cuda_graph_max_bs_decode` | `None` (默认) | 系统根据显存自动计算 max_bs，所有不传此参数的测试已隐式覆盖 |
| | 显式值（如 `8`） | 覆盖自动计算，合并入 `cuda_graph_config.decode.max_bs`，限制 decode graph 最大 batch size |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 手动限制 batch size | 用户需限制 decode graph 最大 batch size 以控制显存占用 | `8` | server 正常启动 + 推理成功（显式 max_bs 生效） |

##### 测试点设计

本参数与 `--cuda-graph-max-bs-prefill`、`--cuda-graph-bs-decode`、`--cuda-graph-bs-prefill` 合并为一个组合测试用例，详见 [§组合测试点设计](#组合测试点设计)。

---

#### 2.5 `--cuda-graph-max-bs-prefill`

##### 业务理解

- **定义**: `server_args.py:2004-2007`，prefill 阶段 CUDA Graph 最大 batch size
- **取值**: `Optional[int]`，默认 `None`（系统根据显存自动计算）

##### 通俗理解

手动限制 prefill graph 捕获的最大 batch size。默认自动算。

##### GPU 社区用例分析

0 个文件引用。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `cuda_graph_max_bs_prefill` | `None` (默认) | 系统自动计算，所有不传此参数的测试已隐式覆盖 |
| | 显式值（如 `8`） | 覆盖自动计算，限制 prefill graph 最大 batch size |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 手动限制 prefill batch | 用户需限制 prefill graph 最大 batch size | `8` | server 正常启动 + 推理成功 |

##### 测试点设计

本参数与 `--cuda-graph-max-bs-decode`、`--cuda-graph-bs-decode`、`--cuda-graph-bs-prefill` 合并为一个组合测试用例，详见 [§组合测试点设计](#组合测试点设计)。

---

#### 2.6 `--cuda-graph-bs-decode`

##### 业务理解

- **定义**: `server_args.py:2008-2011`，decode 阶段 graph 显式 batch size 列表
- **取值**: `Optional[List[int]]`，默认 `None`（系统自动生成 bs 列表）
- **折叠**: 合并入 `cuda_graph_config.decode.bs`

##### 通俗理解

手动指定 decode graph 要捕获哪些 batch size。传了就用你的，不传就自动算。

##### GPU 社区用例分析

0 个文件引用。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `cuda_graph_bs_decode` | `None` (默认) | 系统自动生成 bs 列表，所有不传此参数的测试已隐式覆盖 |
| | 显式值（如 `1 2 4 8`） | 覆盖自动生成，合并入 `cuda_graph_config.decode.bs` |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 手动指定 bs | 用户需精确控制 decode graph 捕获的 batch size | `1 2 4 8` | server 正常启动 + 推理成功 |

##### 测试点设计

本参数与 `--cuda-graph-max-bs-decode`、`--cuda-graph-max-bs-prefill`、`--cuda-graph-bs-prefill` 合并为一个组合测试用例，详见 [§组合测试点设计](#组合测试点设计)。

---

#### 2.7 `--cuda-graph-bs-prefill`

##### 业务理解

- **定义**: `server_args.py:2012-2015`，prefill 阶段 graph 显式 batch size 列表
- **取值**: `Optional[List[int]]`，默认 `None`

##### GPU 社区用例分析

1 个文件引用，非独立测试。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `cuda_graph_bs_prefill` | `None` (默认) | 系统自动生成 bs 列表，所有不传此参数的测试已隐式覆盖 |
| | 显式值（如 `1 2 4`） | 覆盖自动生成，合并入 `cuda_graph_config.prefill.bs` |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 手动指定 prefill bs | 用户需精确控制 prefill graph 捕获的 batch size | `1 2 4` | server 正常启动 + 推理成功 |

##### 测试点设计

本参数与 `--cuda-graph-max-bs-decode`、`--cuda-graph-max-bs-prefill`、`--cuda-graph-bs-decode` 合并为一个组合测试用例，详见 [§组合测试点设计](#组合测试点设计)。

---

##### 组合测试点设计

§2.4–2.7 四个参数功能相近，合并为一个组合用例，通过两次启动对比显存（`max_bs=1` vs `max_bs=8`）验证参数价值，同时日志确认全部参数被解析并生效。

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_cuda_graph_bs_override` | 四参数同时传值，两次启动对比显存（max_bs=1 vs 8）+ 日志验证 + 推理成功 | E2E | 新增 | P1 |

**T001: `test_cuda_graph_bs_override`** (新增)

**验证目标**: 四参数同时显式传值，全部被解析并生效；且 `max_bs` 设小值确实降低了 CUDA Graph 显存占用。

**测试步骤**:
1. 传 `--cuda-graph-max-bs-decode 1 --cuda-graph-max-bs-prefill 1 --cuda-graph-bs-decode 1 --cuda-graph-bs-prefill 1` 启动 Llama 3.2 1B server，捕获 stderr，记录 CUDA Graph 显存占用
2. 重启，传 `--cuda-graph-max-bs-decode 8 --cuda-graph-max-bs-prefill 8 --cuda-graph-bs-decode 1 2 4 8 --cuda-graph-bs-prefill 1 2 4`，捕获 stderr，记录显存占用
3. 两次均发送 `/generate` 请求

**断言**: 两次 `status_code == 200` 且 `response.text` 含 `"Paris"`；stderr 日志含 `max_bs` 和 `bs` 记录；`max_bs=1` 时 CUDA Graph 显存 < `max_bs=8` 时显存，证明 `max_bs` 确实控制显存

---

#### 2.8 `--disable-prefill-cuda-graph`（等价于 `--cuda-graph-backend-prefill=disabled`）

##### 业务理解

- **定义**: `server_args.py:2022`，boolean，Convenience 别名，禁用 prefill CUDA graph
- **取值**: `False` (默认) | `True`
- **等价于**: `--cuda-graph-backend-prefill=disabled`（合并为 `cuda_graph_config.prefill.backend = disabled`）

##### 通俗理解

快捷开关，传了就等于 `--cuda-graph-backend-prefill=disabled`，prefill 不走 graph 加速。

##### GPU 社区用例分析

0 个文件直接引用此参数名。但等价的 `--cuda-graph-backend-prefill=disabled` 有 GPU E2E（`test_no_extra_forked_cuda_context.py`，副作用验证模式）。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `disable_prefill_cuda_graph` | `False` (默认) | 不生效，所有不传此参数的测试已隐式覆盖 |
| | `True` | 合并为 `cuda_graph_config.prefill.backend = disabled`，与 `--cuda-graph-backend-prefill=disabled` 走相同代码路径 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 快捷禁用 | 用户用开关代替传值，期望行为与 `--cuda-graph-backend-prefill=disabled` 一致 | `True` | 参数被解析并合并为 disabled，server 正常启动 + 推理成功 |

##### 测试点设计

本参数与 `--disable-decode-cuda-graph`、`--disable-piecewise-cuda-graph` 合并为一个组合测试用例，详见 [§组合测试点设计](#组合测试点设计-2)。

---

#### 2.9 `--disable-decode-cuda-graph`（等价于 `--cuda-graph-backend-decode=disabled`）

##### 业务理解

- **定义**: `server_args.py:2026`，boolean，Convenience 别名，禁用 decode CUDA graph
- **取值**: `False` (默认) | `True`
- **等价于**: `--cuda-graph-backend-decode=disabled`（合并为 `cuda_graph_config.decode.backend = disabled`）

##### 通俗理解

快捷开关，传了就等于 `--cuda-graph-backend-decode=disabled`，decode 不走 graph 加速。

##### GPU 社区用例分析

0 个文件直接引用此参数名。但等价的 `--cuda-graph-backend-decode=disabled` 有 GPU E2E（`test_no_extra_forked_cuda_context.py`，副作用验证模式）。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `disable_decode_cuda_graph` | `False` (默认) | 所有不传此参数的测试已隐式覆盖 |
| | `True` | 合并为 `cuda_graph_config.decode.backend = disabled`，与 `--cuda-graph-backend-decode=disabled` 走相同代码路径 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 快捷禁用 | 用户用开关代替传值 | `True` | 参数被解析并合并为 disabled，server 正常启动 + 推理成功 |

##### 测试点设计

本参数与 `--disable-prefill-cuda-graph`、`--disable-piecewise-cuda-graph` 合并为一个组合测试用例，详见 [§组合测试点设计](#组合测试点设计-2)。

---

#### 2.10 `--disable-piecewise-cuda-graph`（已弃用，别名：`--cuda-graph-backend-prefill=disabled`）

##### 业务理解

- **定义**: `server_args.py:6492-6497`，Deprecated 别名，指向 `--cuda-graph-backend-prefill=disabled`
- **取值**: 传参即生效，等价于 `disabled`

##### 通俗理解

旧版参数名，框架自动转发到 `--cuda-graph-backend-prefill=disabled`，行为完全一致。

##### GPU 社区用例分析

0 个文件引用。

##### E2E 测试分析

Deprecated 别名，与 `--cuda-graph-backend-prefill=disabled` 走相同代码路径。

##### 测试点设计

本参数与 `--disable-prefill-cuda-graph`、`--disable-decode-cuda-graph` 合并为一个组合测试用例，详见 [§组合测试点设计](#组合测试点设计-2)。

---

##### 组合测试点设计

§2.8–2.10 三个 disable 别名合并为一个组合用例，一次启动传全部。

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_disable_cuda_graph` | 三个 disable 别名同时传值，全部生效 + 推理成功 | E2E | 新增 | P1 |

**T001: `test_disable_cuda_graph`** (新增)

**验证目标**: `--disable-prefill-cuda-graph`、`--disable-decode-cuda-graph`、`--disable-piecewise-cuda-graph` 同时传值，全部解析为对应的 disabled backend。

**测试步骤**:
1. 同时传三个参数启动 Llama 3.2 1B server，捕获 stderr
2. `/generate` 请求

**断言**: `status_code == 200`；`response.text` 含 `"Paris"`；stderr 日志含 `disabled` 记录，证明全部别名被解析并生效

---

#### 2.11 `--pre-warm-nccl`

##### 业务理解

- **定义**: `server_args.py:2105`，boolean，启动时预热 NCCL 通信
- **取值**:

| 值 | 含义 |
|---|---|
| `False` (默认) | 不执行预热，NCCL 通信首次调用时初始化 |
| `True` | 启动时执行 NCCL all-reduce 预热，避免首次通信耗时抖动 |

##### 通俗理解

启动时先跑一轮 NCCL 通信让网络链路预热，避免首次通信耗时抖动。

##### GPU 社区用例分析

0 个文件引用。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `pre_warm_nccl` | `False` (默认) | 不执行预热，NCCL 通信在首次推理时初始化。所有不传此 flag 的测试已隐式覆盖 |
| | `True` | 启动阶段执行 NCCL all-reduce 预热，验证预热后 server 正常进入就绪状态 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 启用预热 | 用户部署多卡推理，需预热 NCCL 避免首次通信延迟抖动 | `True` | TP=2 下预热降低首次请求延迟 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_pre_warm_nccl` | TP=2 对比验证：启用预热后首次请求延迟低于未启用 | E2E | 新增 | P2 |

**T001: `test_pre_warm_nccl`** (新增)

**验证目标**: TP=2 多卡场景下，`--pre-warm-nccl` 启用 NCCL 预热后首次请求延迟低于未启用，证明预热的核心价值。

**测试步骤**:
1. TP=2 不传 `--pre-warm-nccl` 启动 Llama 3.2 1B server，捕获 stderr，发送首次 `/generate` 请求并记录延迟
2. 重启，TP=2 传 `--pre-warm-nccl`，捕获 stderr，发送首次请求并记录延迟

**断言**: 两次 `status_code == 200` 且 `response.text` 含 `"Paris"`；True 时 stderr 含 NCCL 预热记录，False 时无；True 首次请求延迟 < False 首次请求延迟

---

#### 2.12 `--enable-dp-attention-local-control-broadcast`

##### 业务理解

- **定义**: `server_args.py:934`，boolean，启用 DP attention 本地控制广播
- **取值**:

| 值 | 含义 |
|---|---|
| `False` (默认) | 不启用，DP attention 控制信息跨机传输 |
| `True` | 启用本地广播，控制信息在本地广播，减少跨机通信开销 |

##### 通俗理解

DP 模式下 attention 计算的控制信息在本地广播而非跨机传输，减少通信开销。

##### GPU 社区用例分析

0 个文件引用。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| `enable_dp_attention_local_control_broadcast` | `False` (默认) | 不启用本地广播，所有不传此 flag 的测试已隐式覆盖 |
| | `True` | 启用后 DP attention 控制信息走本地广播路径，减少一次跨机通信 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 启用本地广播 | 用户部署 DP 多卡推理，启用本地控制广播减少通信开销 | `True` | DP=2 下对比验证：True 启用本地广播 vs False 跨机传输 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_dp_attention_local_control_broadcast` | DP=2 对比验证：启用本地广播后 server 正常启动 + 日志验证生效 | E2E | 新增 | P2 |

**T001: `test_dp_attention_local_control_broadcast`** (新增)

**验证目标**: DP=2 多卡场景下，`--enable-dp-attention-local-control-broadcast` 启用本地广播后 server 正常启动，对比 False 验证参数生效。

**测试步骤**:
1. DP=2 不传参数启动 Llama 3.2 1B server（默认 False），捕获 stderr，验证无本地广播日志
2. 重启，DP=2 传 `--enable-dp-attention-local-control-broadcast`，捕获 stderr，验证有本地广播日志
3. 两次均发送 `/generate` 请求

**断言**: 两次 `status_code == 200` 且 `response.text` 含 `"Paris"`；True 时 stderr 含本地广播记录，False 时无

---

#### 2.13 `--enable-torch-compile-debug-mode`

##### 业务理解

- **定义**: `server_args.py:739`，boolean，在 piecewise CUDA graph 的 NPU graph 回放时校验 tensor 地址一致性
- **取值**:

| 值 | 含义 |
|---|---|
| `False` (默认) | 不启用 debug 校验，NPU graph 回放时跳过地址一致性检查 |
| `True` | 在 NPU graph 捕获时记录所有输入 tensor 的 `data_ptr()` 地址，回放时 assert 地址与捕获时完全一致 |

- **作用链路**:
  ```
  server_args.enable_torch_compile_debug_mode
    → TcPiecewiseCudaGraphBackend.build_compilation_config() (tc_piecewise_cuda_graph_backend.py:119)
      → CompilationConfig(enable_debug_mode=...) (compilation_config.py:25)
        → NPUPiecewiseBackend.__call__() (npu_piecewise_backend.py:41-109)
          → L57-61: debug=True 时，NPU graph 捕获阶段记录 input tensor data_ptr()
          → L99-107: debug=True 时，NPU graph 回放阶段 assert 地址与捕获时一致
  ```

- **依赖**:
  - 前置条件: 必须启用 piecewise CUDA graph（走 `NPUPiecewiseBackend` 路径）
  - **NPU 已知限制**: `server_args.py:1321` 的 `is_npu()` 规则会自动禁用 piecewise cuda graph
  - **绕过方式**: 传 `--enforce-piecewise-cuda-graph`（`server_args.py:1303-1305` 跳过所有 auto-disable 规则）
  - 关联参数: `--enforce-piecewise-cuda-graph`（绕过 is_npu 排除）、`--piecewise-cuda-graph-max-tokens`（配置 prefill batch size）

##### 通俗理解

NPU graph 类似"录像回放"：第一次运行时录制 NPU 算子的执行过程，之后直接回放录像跳过重复计算。
debug 模式在回放前加了一道安检——验证每次回放时 tensor 在内存中的位置和录制时一样。如果地址不一致（说明内存管理有 bug），server 直接 crash 报错。正常推理不需要开，仅用于排查 piecewise cuda graph 的内存问题。

##### GPU 社区用例分析

0 个文件引用。

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响（代码行号） |
|---|---|---|
| `enable_torch_compile_debug_mode` | `False` (默认) | `npu_piecewise_backend.py:57,99`：跳过地址记录/校验，NPU graph 正常回放 |
| | `True` | `npu_piecewise_backend.py:57-61`：捕获时记录 `data_ptr()`；`npu_piecewise_backend.py:99-107`：回放时 assert 地址一致 |
| `enforce_piecewise_cuda_graph` | 不传 | `server_args.py:1321`：`is_npu()` → `disable_piecewise_cuda_graph=True`，NPUPiecewiseBackend 不会被创建 |
| | `True` | `server_args.py:1303-1305`：跳过所有 auto-disable 规则，NPUPiecewiseBackend 正常初始化 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 排查 piecewise 内存问题 | 用户在 NPU 上使用 piecewise cuda graph 遇到内存相关 crash，开启 debug 模式验证 tensor 地址一致性。debug 模式下地址校验会引入额外开销，推理耗时增加 | `True` + `--enforce-piecewise-cuda-graph` | debug 模式推理耗时 > 非 debug 模式 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 来源 | 优先级 |
|---|---|---|---|---|---|
| T1 | `test_enable_torch_compile_debug_mode` | 对比验证：debug ON vs OFF 的 GSM8K benchmark 耗时，debug 模式应更慢（地址校验开销） | E2E | 新增 | P1 |

**T001: `test_enable_torch_compile_debug_mode`** (新增)

**验证目标**: `--enable-torch-compile-debug-mode` 开启后 NPU graph 回放时执行 tensor 地址校验，引入额外耗时，推理总时间应大于关闭时。

**测试步骤**:
1. 使用 Qwen3 14B 模型，传 `--enforce-piecewise-cuda-graph --piecewise-cuda-graph-max-tokens 64 --enable-torch-compile-debug-mode` 启动 server
2. 运行 GSM8K 200 题 benchmark，重复 5 次取平均耗时
3. 关闭 server，重启，不传 `--enable-torch-compile-debug-mode`，其余参数相同
4. 再次运行 GSM8K 200 题 benchmark，重复 5 次取平均耗时

**断言**: `avg_time(debug=True) > avg_time(debug=False)`

---

#### 2.14 `--disable-attn-tp-gather`

##### 业务理解

- **定义**: `server_args.py:946-956`，boolean，禁用 scheduler 侧 attention TP gather 的 padding 和 buffer 预分配
- **取值**:

| 值 | 含义 |
|---|---|
| `False` (默认) | 由 `require_attn_tp_gather()` 自动判定：非 MOE 模型→不 gather；MOE 模型（非 dp_attention）→gather；MOE + dp_attention → dp_size < tp_size 时 gather |
| `True` | 强制跳过 gather padding，`require_attn_tp_gather()` 直接返回 False。供模型层自行管理 SP scatter/gather 的模型使用 |

- **作用链路** (`common.py:3219-3238` → `base_runner.py:587-590` → decode/prefill cuda_graph_runner):

```
disable_attn_tp_gather=True
  → require_attn_tp_gather() → return False (短路，line 3227-3228)

disable_attn_tp_gather=False (默认)
  → require_attn_tp_gather()
    → get_moe_a2a_backend().is_none() 且 moe_dense_tp_size=None → return False (line 3238)
    → MOE 模型 + enable_dp_attention → dp_size < tp_size ? True : False
    → MOE 模型 + 非 dp_attention → return True (line 3236)

When require_attn_tp_gather()=True:
  → global_num_tokens_cpu = [num_tokens] → GPU buffer 预分配 gather 空间
When False:
  → global_num_tokens_cpu = None → 无预分配
```

- **依赖**:
  - 下游: CUDA graph runner (decode/prefill)、speculative decoding runners（eagle/eagle3/mtp/frozen_kv_mtp）
  - 关联参数: `--enable-dp-attention`、MOE 相关参数（`--moe-dense-tp-size`、moe_a2a_backend）
  - 前置条件: 仅 MOE 模型或自定义 attention 实现时 `True`/`False` 触发不同代码路径

- **⚠️ 路径收敛分析** (`common.py:3219-3238`):

| 模型类型 | `disable_attn_tp_gather=True` | `disable_attn_tp_gather=False` | 路径不同？ |
|---|---|---|---|
| Llama (非 MOE) | `→ line 3227-3228 → return False` | `→ line 3238 → return False` | ❌ 相同，`global_num_tokens_cpu = None` |
| MOE (非 dp_attention) | `→ line 3227-3228 → return False` | `→ line 3236 → return True` | ✅ 不同，buffer 分配不同 |

> **结论**: 对标准 Llama 模型，两个取值代码路径完全收敛，与 TP 组网规模无关。仅 MOE 模型能触发差异化行为。

##### 通俗理解

Attention TP 模式下，多张卡各自算一部分 attention 结果，通常需要一个 "gather" 操作把结果拼回完整。
`--disable-attn-tp-gather` 告诉系统："别帮我 gather padding，我的 attention 自己会处理。"

- `False` (默认): 系统自动判断是否需要 gather padding——MOE 模型需要，普通 Llama 不需要。所有不传此 flag 的测试已隐式覆盖
- `True`: 强制跳过，由模型层自己负责 gather/scatter
- 用户什么时候需要改: 部署自己管理 SP scatter/gather 的自定义模型（如某些 DeepSeek 变体），避免框架层 padding 干扰模型内部 gather 逻辑

##### GPU 社区用例分析

| 文件 | 类型 | 测试了什么 | 参数角色 | 覆盖值 | 可移植 |
|---|---|---|---|---|---|
| — | — | GPU 社区无引用 | — | — | — |

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 (引用代码行号) |
|---|---|---|
| `disable_attn_tp_gather` | `False` (默认) | `common.py:3238` → `require_attn_tp_gather()` 走自动判定；非 MOE 模型返回 False；所有不传此 flag 的测试已隐式覆盖 |
| | `True` | `common.py:3227-3228` → `require_attn_tp_gather()` 短路返回 False；对非 MOE 模型与 False 路径收敛，对 MOE 模型行为不同 |
| 模型类型 | Llama (非 MOE) | 两个取值均 → `global_num_tokens_cpu = None`，路径收敛 |
| | MOE (非 dp_attention) | True→None vs False→gather buffer，路径分化 |

**用户场景分析** |

| 场景 | 用户场景 | `--moe-dense-tp-size` 取值 | `--disable-attn-tp-gather` 取值 | 验证点 |
|---|---|---|---|---|
| MOE 模型禁用 gather | 用户部署 MOE 模型（如 OLMoE），需验证 `--disable-attn-tp-gather` 能正确禁用 scheduler 侧的 attention TP gather padding，且不影响推理 | `1` | `False` (默认) | `require_attn_tp_gather()=True` → gather buffer 分配；server 正常启动 + 推理正确 |
| MOE 模型启用 flag | 用户部署自定义 attention 的 MOE 模型，需跳过框架层 gather padding 让模型自行管理 | `1` | `True` | `require_attn_tp_gather()=False` (短路) → 无 gather buffer；server 正常启动 + 推理正确且与 False 一致 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_disable_attn_tp_gather_contrastive` | 对比验证：OLMoE + `--moe-dense-tp-size 1`，传 flag vs 不传 flag — 触发不同 `require_attn_tp_gather()` 路径但推理结果一致且正确 | E2E | Parameter | 新增 | P1 |

**T001: `test_disable_attn_tp_gather_contrastive`** (新增)

**验证目标**: `--disable-attn-tp-gather` 在 MOE 模型 (`--moe-dense-tp-size 1`) 下触发 `require_attn_tp_gather()` 的 True→False 路径切换，验证两种路径下推理均正确一致。

**测试步骤**:
1. 启动 OLMoE-1B-7B server，args: `--moe-dense-tp-size 1` + `--attention-backend ascend`，发送 `/generate` 请求，记录响应 R1
2. 重启，args: `--moe-dense-tp-size 1 --disable-attn-tp-gather --attention-backend ascend`，发送相同 `/generate` 请求，记录响应 R2
3. 对比两次响应

**断言**: 两次 `status_code == 200`；R1 和 R2 推理输出一致（temperature=0 确保确定性）；均为有效英文文本

##### 结论

| 值 | 测试策略 | 说明 |
|---|---|---|
| `False` (默认) | T001 对比验证 + 隐式覆盖 | T001 Phase 1 (不传 flag + `--moe-dense-tp-size 1`) 覆盖 `require_attn_tp_gather()=True` 路径；所有不传此 flag 的测试隐式覆盖非 MOE 下的 False 路径 |
| `True` | T001 对比验证 | T001 Phase 2 (传 flag + `--moe-dense-tp-size 1`) 覆盖 `require_attn_tp_gather()=False` 短路路径 |

---

#### 2.15 `--enable-dsa-prefill-context-parallel`（已弃用，别名：`--enable-prefill-cp`）

##### 业务理解

- **定义**: `server_args.py:6542-6548`，Deprecated 别名
- **等价于**: `--enable-prefill-cp`
- **用途**: 为 DeepSeek V3.2 (DSA) 模型启用 prefill context parallel

##### 通俗理解

为 DeepSeek V3.2 (DSA) 模型启用 prefill context parallel 的旧参数名，框架自动转发到 `--enable-prefill-cp`。

##### GPU 社区用例分析

| 参数名 | 覆盖文件 | 类型 | 测试内容 |
|---|---|---|---|
| 旧参数 `--enable-dsa-prefill-context-parallel` | 0 个文件 | — | — |
| 别名目标 `--enable-prefill-cp` | `test_gqa_prefill_cp.py` | E2E | GQA 模型 prefill CP |
| | `test_cp_prefix_len_fa3_parity.py` | E2E | CP 与 FA3 精度一致性 |
| | `test_deepseek_v4_pro_fp4_cp.py` | E2E | DeepSeek V4 + CP |
| | `test_server_args.py` | UT | 参数解析 |

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| 是否传参 | 不传 (默认) | prefill 不做 context parallel，所有不传此参数的测试已隐式覆盖 |
| | 传参 (TP=2) | 启用 prefill CP——prefill 阶段序列在 TP 设备间分片并行计算，长上下文时加速 prefill |
| 组网方式 | TP=2 | CP 需多卡分片，TP=2 即可验证分片生效 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| DSA 模型长上下文加速 | 用户部署 DeepSeek V3.2 (DSA) 模型，处理超长文档时 prefill 耗时过大，启用 CP 将 prompt 分片到多卡并行计算 | 传参 (TP=2) | 参数被解析并生效，server 正常启动 + 推理正确 |

##### 测试点设计

与 `--enable-prefill-cp`、`--cp-strategy` 合并为一个组合用例文件，验证 DSA 场景下 CP 功能。详见 [§CP 组合测试点设计](#cp-组合测试点设计)。

---

#### 2.16 `--enable-prefill-context-parallel`（已弃用，别名：`--enable-prefill-cp`）

##### 业务理解

- **定义**: `server_args.py:6549-6555`，Deprecated 别名
- **等价于**: `--enable-prefill-cp`
- **用途**: 为 MLA/MHA/GQA 模型启用 prefill context parallel

##### 通俗理解

为 MLA/MHA/GQA 模型启用 prefill context parallel 的旧参数名，框架自动转发到 `--enable-prefill-cp`。与 §2.15 互斥。

##### GPU 社区用例分析

| 参数名 | 覆盖文件 | 类型 | 测试内容 |
|---|---|---|---|
| 旧参数 `--enable-prefill-context-parallel` | 0 个文件 | — | — |
| 别名目标 `--enable-prefill-cp` | `test_gqa_prefill_cp.py` | E2E | GQA 模型 prefill CP |
| | `test_cp_prefix_len_fa3_parity.py` | E2E | CP 与 FA3 精度一致性 |
| | `test_deepseek_v4_pro_fp4_cp.py` | E2E | DeepSeek V4 + CP |
| | `test_server_args.py` | UT | 参数解析 |

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| 是否传参 | 不传 (默认) | prefill 不做 context parallel，所有不传此参数的测试已隐式覆盖 |
| | 传参 (TP=2) | 启用 prefill CP——prefill 阶段序列在 TP 设备间分片并行计算 |
| 组网方式 | TP=2 | CP 需多卡分片，TP=2 即可验证分片生效 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| MLA/MHA/GQA 模型长上下文加速 | 用户部署 GQA/MLA 架构模型，处理长 prompt 需降低 prefill 延迟，启用 CP 将序列分片到多卡并行 prefill | 传参 (TP=2) | 参数被解析并生效，server 正常启动 + 推理正确 |

##### 测试点设计

与 `--enable-prefill-cp`、`--cp-strategy` 合并为一个组合用例文件，验证 MLA/MHA/GQA 场景下 CP 功能。详见 [§CP 组合测试点设计](#cp-组合测试点设计)。

---

#### 2.17 `--dsa-prefill-cp-mode`（已弃用，别名：`--cp-strategy`）

##### 业务理解

- **定义**: `server_args.py:6556-6568`，Deprecated 别名
- **等价于**: `--cp-strategy`
- **取值映射**: `in-seq-split` → `zigzag`；`round-robin-split` → `interleave`

##### 通俗理解

控制 prefill context parallel 分片策略的旧参数名，框架自动转发到新参数 `--cp-strategy`。

##### GPU 社区用例分析

| 参数名 | 覆盖文件 | 类型 | 测试内容 |
|---|---|---|---|
| 旧参数 `--dsa-prefill-cp-mode` | 0 个文件 | — | — |
| 别名目标 `--cp-strategy` | `test_cp_strategy_unit.py` | UT | CP strategy 映射、绑定、分片/汇聚、attention dispatch |
| | `test_gqa_prefill_cp.py` | E2E | GQA 模型 CP |
| | `test_deepseek_v4_pro_fp4_cp.py` | E2E | DeepSeek V4 + CP |

##### E2E 测试分析

**测试因子分析** |

| 因子 | 取值 | 具体影响 |
|---|---|---|
| 分片策略 | `in-seq-split` | 等价于 `zigzag`：序列 token 按顺序交错分配到各设备，相邻 token 在不同卡上 |
| | `round-robin-split` | 等价于 `interleave`：序列均匀分配，每卡拿到连续的一段 token |
| 组网方式 | TP=2 | 多卡 CP 环境，两种策略产生不同的 token 分布模式 |

**用户场景分析** |

| 场景 | 用户场景 | 取值 | 验证点 |
|---|---|---|---|
| 序列交错分片 | 用户需相邻 token 分散在不同卡上，适合需要跨卡 attention 的场景 | `in-seq-split` (TP=2) | 参数被解析并映射为 zigzag，server 正常启动 + 推理正确 |

##### 测试点设计

与 `--enable-prefill-cp`、`--cp-strategy` 合并为一个组合用例文件，验证旧参数名的值映射和功能等价性。详见 [§CP 组合测试点设计](#cp-组合测试点设计)。

---

#### 2.18 `--prefill-cp-mode`（已弃用，别名：`--cp-strategy`）

##### 业务理解

- **定义**: `server_args.py:928 + 6570+`，`no_cli=True` + Deprecated 别名
- **等价于**: `--cp-strategy`

##### 通俗理解

内部使用的 prefill context parallel 模式参数，不对外暴露，框架根据模型类型自动设置。

##### GPU 社区用例分析

| 参数名 | 覆盖文件 | 类型 | 测试内容 |
|---|---|---|---|
| 旧参数 `--prefill-cp-mode` | 3 个文件 | 内部引用 | `no_cli=True`，均为内部代码参数传递 |
| 别名目标 `--cp-strategy` | `test_cp_strategy_unit.py` | UT | CP strategy 映射、绑定、分片/汇聚、attention dispatch |
| | `test_gqa_prefill_cp.py` | E2E | GQA 模型 CP |
| | `test_deepseek_v4_pro_fp4_cp.py` | E2E | DeepSeek V4 + CP |

##### E2E 测试分析

`no_cli=True`，不对外暴露 CLI 参数，无法通过命令行传参测试。其等价目标 `--cp-strategy` 已有 E2E 测试点（TP=2），详见 §2.20。

---

#### 2.19 `--enable-prefill-cp`（`--enable-dsa-prefill-context-parallel`、`--enable-nsa-prefill-context-parallel`、`--enable-prefill-context-parallel` 的别名目标）

##### 2.19.1 业务理解

- **定义**: `server_args.py:914-917`，boolean flag，启用 prefill 阶段的 context parallelism（上下文并行）。将 prefill 输入序列在多个设备间分片并行计算 attention，降低长上下文的 prefill 延迟。
- **取值**:

  | 值 | 含义 |
  |---|---|
  | `False` (默认) | 不启用 CP。prefill attention 在单设备上完整计算，所有不传此 flag 的测试均走此路径 |
  | `True` | 启用 CP。需配合 `--cp-strategy`（选择分片策略）和 `--attn-cp-size > 1`（设置 CP 并行度）。NPU 上 `--attn-cp-size` 必须等于 `--tp-size` |

- **作用链路**:

  ```
  CLI: --enable-prefill-cp → server_args.enable_prefill_cp = True
    │
    ├─► _handle_legacy_cp_arguments() [server_args.py:5034-5074]
    │     弃用别名 (--enable-dsa-prefill-context-parallel / --enable-nsa-prefill-context-parallel
    │     / --enable-prefill-context-parallel) 通过 DeprecatedStoreTrueAction 设置对应的
    │     dest 字段, 此函数统一将 enable_prefill_cp 设为 True
    │     └─► enable_dsa/nsa 别名 → enable_dsa_prefill_context_parallel=True → DSA CP 路径
    │     └─► enable-prefill-context-parallel → enable_prefill_context_parallel=True → GQA/MLA CP 路径
    │
    ├─► _handle_context_parallelism() [server_args.py:5076-5104]
    │     ├─ enable_prefill_cp=True & cp_strategy=None → ValueError (必须设置策略)
    │     ├─ model_arch in CP_V2_DEFAULT_MODEL_CLASSES → SGLANG_ENABLE_CP_V2=True
    │     └─ DSA model arch: attn_cp_size = tp_size // dp_size (非 NPU 自动设置)
    │
    ├─► init_cp_strategy(server_args) [layers/cp/base.py:210-236]
    │     ├─ enable_prefill_cp=False → _STRATEGY = None (CP 不初始化)
    │     ├─ attn_cp_size ≤ 1 → _STRATEGY = None (CP 不初始化)
    │     ├─ cp_strategy="zigzag" → ZigzagCPStrategy(cp_size) [layers/cp/zigzag.py]
    │     └─ cp_strategy="interleave" → InterleaveCPStrategy(cp_size) [layers/cp/interleave.py]
    │
    ├─► ModelRunner forward [model_executor/model_runner.py]
    │     DSA/MLA 模型: attn_cp_size 自动配置 = tp_size // dp_size
    │     GQA 模型: attn_cp_size 需显式传入
    │
    └─► NPU Attention Backend [hardware_backend/npu/attention/ascend_backend.py]
          ├─ DSA CP 路径 (line 1036-1050):
          │   is_dsa_enable_prefill_cp() & attn_cp_metadata is not None
          │   → do_cp_balance_attn() — DSA 专用 CP attention 分发
          │
          └─ 通用 CP 路径 (line 1134-1148):
              is_context_parallel_extend() & attn_cp_metadata is not None & attn_cp_size > 1
              → _cp_allgather_and_save_kv_npu() — 汇聚 K/V 到完整序列后写入 KV pool
  ```

- **依赖**:

  | 类型 | 内容 |
  |---|---|
  | 下游 | `layers/cp/base.py` — `init_cp_strategy`、`get_cp_strategy`、`is_cp_enabled`；`ascend_backend.py` — CP attention 分发 |
  | 关联参数 | `--cp-strategy`（**必需**）、`--attn-cp-size`（必须 > 1，NPU 上 = `--tp-size`） |
  | 前置条件 | TP ≥ 2，`--cp-strategy` 已设置，`--attn-cp-size > 1` |
  | 冲突 | `--prefill-only-disable-kv-cache` (server_args.py:5581-5586)；PD disaggregation decode 模式 (server_args.py:3715-3718) |
  | NPU 特殊 | `is_npu()` 为 True 时跳过 Hopper 平台实验性警告 (server_args.py:3645)，但 `attn_cp_size` 不会自动设置，需显式传入 |

##### 2.19.2 通俗理解

Context Parallelism (CP) 就像把一篇长文章复印后分给 4 个人同时阅读，每人负责其中一部分做笔记，最后把笔记汇总起来。没有 CP 时，一个人要读完整个文章才能做完所有笔记。

- **`False` (默认)**: 不拆分。prefill 阶段所有 token 在单个设备上顺序计算 attention。适合短 prompt（几百 token 以内），单卡就能快速完成。
- **`True`**: 启用拆分。prefill 序列被切成多段，分发到不同 NPU 上并行计算 attention，最后汇总结果。适合超长 prompt（数万 token），prefill 延迟可从数秒降低到亚秒级。
- **用户什么时候需要改**: 处理超长文档（如法律合同、学术论文、代码库全量分析）时，prefill 阶段耗时过大成为瓶颈，启用 CP 将 prefill 延迟线降低到可接受范围。需要至少 2 张 NPU。

##### 2.19.3 GPU 社区用例分析

| 文件 | 类型 | 测试了什么 | 参数角色 | 覆盖值 | 可移植 |
|---|---|---|---|---|---|
| `test/registered/cp/test_gqa_preill_cp.py` | E2E | Qwen3-30B-A3B-FP8, 4 GPU (TP=4, ATTN_CP=2, MOE_DP=2/EP=4), GSM8K accuracy ≥ 0.93。3 个 class 覆盖不同 (TP,EP,CP) 组合，均使用 `--enable-prefill-cp --cp-strategy zigzag` | 被测试对象（作为 CP 功能的开关） | `True` (配合 `--cp-strategy zigzag`) | ✅ 可移植：改 GPU→NPU，`attention-backend fa3`→`ascend`，Qwen3-30B-A3B-FP8→Qwen3-30B-A3B NPU 权重，`register_cuda_ci`→`register_npu_ci` |
| `test/registered/cp/test_cp_strategy_unit.py` | UT (CPU) | ZigzagCPStrategy：metadata 计算、hidden states 分片/汇聚、KV cache 分片/汇聚、position IDs 分片、attention dispatch。**不涉及 server 启动** | 被测试对象（传入 `SimpleNamespace(enable_prefill_cp=True)` 初始化 strategy） | `True` (配合 zigzag) | ✅ 可移植为 NPU UT |
| `test/registered/cp/test_deepseek_v32_cp_single_node.py` | E2E | DeepSeek V3.2, 8 GPU, `--enable-dsa-prefill-context-parallel` + EAGLE, GSM8K | 被测试对象 | `True` (DSA deprecated alias) | ❌ 模型过大（DeepSeek V3.2），NPU 移植需另行评估 |
| `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` | E2E | DeepSeek-V4-Pro FP4, 8 MI35x, `--enable-prefill-cp --cp-strategy interleave`, unified_kv, GSM8K | 被测试对象 | `True` (配合 `--cp-strategy interleave`) | ❌ AMD 专用，模型过大 |
| `test/registered/unit/server_args/test_server_args.py` | UT | CLI 解析：canonical flag 解析、弃用别名转发映射、`--cp-strategy` 缺失报错 | 被测试对象 | `True`（多种组合） | ✅ 已有 NPU 移植 `test/registered/unit/server_args/test_server_args.py` |

##### 2.19.4 E2E 测试分析

###### 2.19.4.1 测试因子分析

| 因子 | 取值 | 具体影响（代码行号） |
|---|---|---|
| `enable_prefill_cp` | `False` (默认) | `init_cp_strategy()` [base.py:214] → `_STRATEGY = None` → CP 路径不激活。`forward_batch.attn_cp_metadata` 为 None → `ascend_backend.py:1138` 条件不满足 → 走常规 attention 路径。所有不传此 flag 的 NPU 测试均隐式覆盖 |
| | `True` | `init_cp_strategy()` → 根据 `cp_strategy` 创建 ZigzagCPStrategy 或 InterleaveCPStrategy。prefill 请求进入 EXTEND/MIXED forward mode → `is_context_parallel_extend()=True` [forward_batch_info.py:116] → `ascend_backend.py:1137` CP 路径激活 → KV 通过 `_cp_allgather_and_save_kv_npu` 汇聚写入 pool |
| `cp_strategy` | `"zigzag"` | `init_cp_strategy()` [base.py:224] → `ZigzagCPStrategy`。序列 token 按 zigzag 模式交替分配到各 rank（如 rank0: token 0,3,4,7；rank1: token 1,2,5,6），相邻 token 分布在不同 rank |
| | `"interleave"` | `init_cp_strategy()` [base.py:229] → `InterleaveCPStrategy`。序列按连续块分配（如 rank0: token 0-3；rank1: token 4-7），每 rank 拿到连续的一段 |
| `attn_cp_size` | `1` (默认) | `init_cp_strategy()` [base.py:219] → `cp_size <= 1` → `_STRATEGY = None` → CP 不激活。**NPU 上 DSA 模型不自动设置此值**（`is_npu()` 分支跳过 [server_args.py:3645]），必须显式传入 |
| | `2` (= `--tp-size`) | CP 初始化成功，2 个设备间分片 prefill。NPU 上这是最小合法值 |
| `--attention-backend` | `"ascend"` (NPU 默认) | CP attention 走 `ascend_backend.py` 的 NPU 实现：DSA CP 用 `do_cp_balance_attn()`，通用 CP 用 `_cp_allgather_and_save_kv_npu()` |
| | `"fa3"` / `"flashinfer"` (GPU) | GPU CP attention 路径，NPU 不适用 |

> **关键结论**: NPU 上 `--enable-prefill-cp` 要真正激活 CP，须同时满足 3 个条件：① `enable_prefill_cp=True` ② `--cp-strategy` 已设置（zigzag 或 interleave）③ `--attn-cp-size > 1`（显式传入，NPU 不会自动设置）。三者缺一不可。

###### 2.19.4.2 用户场景分析

| 场景 | 用户场景（真实使用描述） | `--enable-prefill-cp` 取值 | 验证点 |
|---|---|---|---|
| 长上下文 CP 加速 | 用户部署 Qwen3/GQA 模型处理超长 prompt（如 32K token），单卡 prefill 延迟过高。启用 CP 将 prefill 分片到 2 张 NPU 上并行计算，期望推理结果正确 | `True` (配合 `--cp-strategy zigzag --attn-cp-size 2 --tp-size 2`) | ① server 正常启动（CP strategy 初始化成功）；② 推理结果正确（MMLU accuracy 不低于基准）；③ CP 路径确实被激活（可通过日志或显存模式间接验证） |
| 默认不启用 CP | 用户部署常规推理服务，prompt 较短（几百 token），无需 prefill 并行加速。不传 `--enable-prefill-cp` | `False` (默认，不传) | 所有不传此 flag 的 NPU 测试均隐式覆盖。显式传 `False` 无意义（与默认同路径），不做显式测试 |
| interleave 策略 | 用户遇到特定模型/场景下 zigzag 策略表现不佳，尝试 interleave 分片策略（连续块分配） | `True` (配合 `--cp-strategy interleave --attn-cp-size 2 --tp-size 2`) | server 正常启动 + 推理正确，证明 interleave 策略在 NPU 上可用 |
| 弃用别名兼容 | 老用户迁移脚本仍使用旧参数名 `--enable-prefill-context-parallel`，期望行为与 `--enable-prefill-cp --cp-strategy zigzag` 一致 | `True` (通过弃用别名 `--enable-prefill-context-parallel`，配合 `--attn-cp-size 2 --tp-size 2`) | 框架接受旧参数名，内部转发到 `--enable-prefill-cp`，server 正常启动 + 推理正确 |

##### 2.19.5 测试点设计

`--enable-prefill-cp` 自身为 bool flag，其"启用 CP"功能与 `--cp-strategy` 强绑定（无策略无法启用）。测试点与 `--cp-strategy`（§2.20）及弃用别名（§2.15–§2.18）合并设计，详见 [§CP 组合测试点设计](#cp-组合测试点设计)。

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_cp_strategy_zigzag_mmlu` | `--enable-prefill-cp --cp-strategy zigzag --attn-cp-size 2 --tp-size 2`，**对比验证**：① MMLU accuracy ≥ 基线；② 与 interleave 固定 prompt 输出一致，证明两策略均正确 | E2E | 对比验证 | 新增 | P0 |
| T2 | `test_enable_prefill_cp_alias` | `--enable-prefill-context-parallel --attn-cp-size 2 --tp-size 2`（弃用别名），验证框架转发→zigzag + 固定 prompt 输出与 T1 zigzag 一致 | E2E | 别名兼容 | 新增 | P1 |
| T3 | `test_enable_dsa_prefill_cp_alias` | `--enable-dsa-prefill-context-parallel --attn-cp-size 2 --tp-size 2`（DSA 弃用别名），验证框架转发→interleave + 固定 prompt 输出与 T1 interleave 一致 | E2E | 别名兼容 | 新增 | P1 |
| T4 | `test_dsa_cp_mode_alias` | `--enable-prefill-cp --dsa-prefill-cp-mode in-seq-split --attn-cp-size 2 --tp-size 2`（mode 弃用别名），验证 `in-seq-split`→`zigzag` 映射 + 固定 prompt 输出与 T1 zigzag 一致 | E2E | 别名兼容 | 新增 | P1 |

---

#### 2.20 `--cp-strategy`（`--dsa-prefill-cp-mode`、`--nsa-prefill-cp-mode`、`--prefill-cp-mode` 的别名目标）

##### 2.20.1 业务理解

- **定义**: `server_args.py:918-924`，控制 prefill context parallel 的序列分片策略。与 `--enable-prefill-cp` 配合使用，选择 token 在各设备间的分配模式。
- **取值**:

  | 值 | 含义 |
  |---|---|
  | `None` (默认) | 不设置策略。若 `--enable-prefill-cp=True` 但 `--cp-strategy=None` → `ValueError`（必须设置） |
  | `"zigzag"` | 前身为 `in-seq-split` 模式。序列 token 按 zigzag 模式交替分配到各 rank。例如 CP=2：rank0 拿 token [0,1, 6,7]，rank1 拿 token [2,3,4,5]。相邻 attention 计算的 token 分布在多 rank 上，适合需要跨 rank 通信的 attention 模式 |
  | `"interleave"` | 前身为 `round-robin-split` 模式。序列 token 按连续块均匀分配到各 rank。例如 CP=2：rank0 拿 token [0,1,2,3]，rank1 拿 token [4,5,6,7]。每 rank 拿到连续 token 段，局部 attention 计算更连续 |

- **作用链路**:

  ```
  CLI: --cp-strategy zigzag|interleave → server_args.cp_strategy
    │
    ├─► _handle_legacy_cp_arguments() [server_args.py:5034-5074]
    │     弃用别名 --dsa-prefill-cp-mode / --nsa-prefill-cp-mode / --prefill-cp-mode
    │     → 映射 legacy_mode_to_strategy:
    │        "in-seq-split" → "zigzag"
    │        "round-robin-split" → "interleave"
    │     → server_args.cp_strategy = 映射后的值
    │
    ├─► _handle_context_parallelism() [server_args.py:5088-5091]
    │     enable_prefill_cp=True & cp_strategy=None → ValueError
    │
    ├─► init_cp_strategy() [layers/cp/base.py:223-236]
    │     cp_strategy="zigzag" → ZigzagCPStrategy(cp_size=attn_cp_size)
    │     cp_strategy="interleave" → InterleaveCPStrategy(cp_size=attn_cp_size)
    │
    └─► 运行时使用 [layers/cp/zigzag.py]、[layers/cp/interleave.py]
          ├─ build_metadata() — 计算各 rank 的 token 分片元数据 (split_list, zigzag_index 等)
          ├─ shard_hidden_states() — 按策略将 hidden states 分片到各 rank
          ├─ gather_hidden_states() — 汇聚各 rank 的 hidden states 回完整序列
          ├─ gather_kv_cache() — 汇聚各 rank 的 KV cache
          └─ run_attention() — 按策略分发 attention 计算 (prev/next 两次调用)
  ```

- **依赖**:

  | 类型 | 内容 |
  |---|---|
  | 下游 | `layers/cp/zigzag.py` — ZigzagCPStrategy；`layers/cp/interleave.py` — InterleaveCPStrategy；`ascend_backend.py` — CP attention 后端 |
  | 关联参数 | `--enable-prefill-cp`（**必需前置**）、`--attn-cp-size`（必需 > 1） |
  | 前置条件 | `--enable-prefill-cp=True`，`--attn-cp-size > 1` |
  | 弃用别名 | `--dsa-prefill-cp-mode` (DSA), `--nsa-prefill-cp-mode` (DSA), `--prefill-cp-mode` (内部, `no_cli=True`) |

##### 2.20.2 通俗理解

`--cp-strategy` 决定了"如何把一篇文章分给多个人读"：

- **`zigzag`** — 把文章切成小段，交替分给不同的人。比如第 1 段给 A，第 2 段给 B，第 3 段又给 A…就像拉链一样交错排列。好处是每个人的工作量均匀，相邻内容被不同人处理，适合需要相互参照的场景。
- **`interleave`** — 把文章切成大块，每人拿连续的一大段。比如前一半给 A，后一半给 B。每个人手里的内容都是连续的，局部计算效率更高。
- **用户什么时候需要改**: 默认 zigzag 适合大多数场景（attention 需要跨 token 交互）。某些模型/硬件组合下 interleave 可能有更好的 cache 局部性。两个策略都正确，差异在于性能而非精度。

##### 2.20.3 GPU 社区用例分析

| 文件 | 类型 | 测试了什么 | 参数角色 | 覆盖值 | 可移植 |
|---|---|---|---|---|---|
| `test/registered/cp/test_cp_strategy_unit.py` | UT (CPU) | ZigzagCPStrategy 的 metadata 计算、hidden states 分片/汇聚、KV cache 分片/汇聚、position IDs 分片、attention dispatch（prev/next 两次调用）、cp_v2 路径。覆盖 cp_size=2,4 和多种 bs×seq_len 组合。**不涉及 server** | 被测试对象（`SimpleNamespace(cp_strategy="zigzag" / "interleave")`） | `"zigzag"`, `"interleave"` | ✅ 可移植为 NPU UT |
| `test/registered/cp/test_gqa_preill_cp.py` | E2E | 3 个 class，Qwen3-30B-A3B-FP8, GSM8K ≥ 0.93。全部使用 `--cp-strategy zigzag`，不同 class 覆盖不同 (TP,CP,EP) 组合：TP2CP2EP2、TP2CP2EP4、CP4EP4 | 启动参数（作为 `--enable-prefill-cp` 的必需配对参数） | `"zigzag"` (全部 3 个 class) | ✅ 可移植（zigzag 路径） |
| `test/registered/amd/test_deepseek_v4_pro_fp4_cp.py` | E2E | DeepSeek-V4-Pro FP4, 8 MI35x, `--cp-strategy interleave`, GSM8K | 启动参数 | `"interleave"` | ❌ AMD 专用，模型过大 |

> GPU 社区中 `interleave` 策略的 E2E 覆盖仅 AMD DeepSeek-V4 测试。GPU CUDA 侧 `test_gqa_prefill_cp.py` 只用 zigzag。NPU 需同时覆盖两个策略。

##### 2.20.4 E2E 测试分析

###### 2.20.4.1 测试因子分析

| 因子 | 取值 | 具体影响（代码行号） |
|---|---|---|
| `cp_strategy` | `"zigzag"` | `init_cp_strategy()` [base.py:224] → `ZigzagCPStrategy`。`build_metadata()` [zigzag.py] 计算 zigzag 分片索引，相邻 token 在 `zigzag_index` 中交错排列。`run_attention()` 分 prev/next 两次调用 attention kernel |
| | `"interleave"` | `init_cp_strategy()` [base.py:229] → `InterleaveCPStrategy`。`build_metadata()` [interleave.py] 按连续块计算分片索引。与 zigzag 使用**完全不同的 metadata 计算逻辑**和**不同的 shard/gather 索引模式** |
| `attn_cp_size` | `2` | 2 个设备分片。ZigzagCPStrategy(2) → 4 段（2 段 prev + 2 段 next）。InterleaveCPStrategy(2) → 2 个连续块 |
| `--attention-backend` | `"ascend"` | NPU 后端。CP 的 gather 操作通过 HCCL all-gather 实现 |
| 模型类型 | GQA (如 Qwen3, Llama) | 通用 CP 路径（ascend_backend.py:1134-1148），走 `_cp_allgather_and_save_kv_npu` |
| | DSA (如 DeepSeek) | DSA CP 路径（ascend_backend.py:1036-1050），走 `do_cp_balance_attn` |

> **关键结论**: `zigzag` 和 `interleave` 触发**完全不同的 strategy 类**（不同的 metadata 计算、不同的 shard/gather 索引、不同的 attention dispatch 逻辑）。两条路径必须**分别测试**。不存在"两个值同路径只测一个"的情况。

###### 2.20.4.2 用户场景分析

| 场景 | 用户场景（真实使用描述） | `--cp-strategy` 取值 | 验证点 |
|---|---|---|---|
| 默认 zigzag 分片 | 用户启用 CP 后未指定策略（或明确选 zigzag），期望序列 token 交错分配到设备上，attention 计算正确 | `"zigzag"` (配合 `--enable-prefill-cp --attn-cp-size 2 --tp-size 2`) | server 正常启动 + MMLU accuracy ≥ 基准。证明 ZigzagCPStrategy 的 metadata、分片、汇聚、attention dispatch 在 NPU 上全部正确 |
| interleave 分片 | 用户根据模型特性选择 interleave 分片策略，期望连续块分配方式的 attention 计算正确 | `"interleave"` (配合 `--enable-prefill-cp --attn-cp-size 2 --tp-size 2`) | server 正常启动 + MMLU accuracy ≥ 基准。证明 InterleaveCPStrategy 在 NPU 上工作正常 |
| 旧 mode 参数兼容 | 老用户迁移脚本使用 `--dsa-prefill-cp-mode in-seq-split`，期望自动映射为 `--cp-strategy zigzag` | `"zigzag"` (通过弃用别名 `--dsa-prefill-cp-mode in-seq-split`, 配合 `--enable-prefill-cp --attn-cp-size 2 --tp-size 2`) | 框架接受旧 mode 名，内部映射为 zigzag，server 正常启动 + 推理正确 |

##### 2.20.5 测试点设计

与 §2.19 合并。zigzag 和 interleave 各一个 P0 E2E 测试点，弃用 mode 别名合并为 P1 测试点。详见 [§CP 组合测试点设计](#cp-组合测试点设计)。

---

##### CP 组合测试点设计

§2.15–§2.20 共 6 个 CP 相关参数（含 4 个弃用别名 + 2 个规范参数），合并为一个组合用例文件 `test_npu_prefill_cp.py`，TP=2、ATTN_CP=2 下验证全部取值和别名转发。

> **NPU CP 最小配置**: `--tp-size 2 --attn-cp-size 2 --enable-prefill-cp --cp-strategy <zigzag|interleave>`。NPU 上 `attn_cp_size` 不会自动设置（`is_npu()` 分支跳过 server_args.py:3645 的 auto-set），必须显式传入。`--cp-strategy` 是 `--enable-prefill-cp` 的必需配对参数（server_args.py:5088-5091）。

> **模型选择 — Llama 3.2 1B（主选）与 Qwen3-30B-A3B（备选）**:
>
> **主选 Llama 3.2 1B**，理由：
> - TP=2 时模型在两卡间均分，内存充足（<2GB/卡），启动快（~30s）
> - MMLU eval (num_examples=64) 的 prompt 长度（100–500 tokens）足以使 `can_apply()` 返回 True（`num_tokens ≥ 2×cp_size=4`），CP 路径被真正触发
> - `ascend_backend.py:1134-1148` 通用 CP 路径（非 DSA）处理 `attn_cp_metadata` → `_cp_allgather_and_save_kv_npu`，与模型架构无关
>
> **风险**: Llama 3.2 1B + CP 在 NPU 上未经 CI 验证。若 Llama CP 在 NPU 上不可用（如 Ascend flash attention 对 CP-sharded 输入不支持），**切换为 Qwen3-30B-A3B**（已知 NPU CP 可用，test_npu_qwen3_30b_attn_cp.py 已通过 CI）。
>
> 两个模型测试逻辑完全相同（相同的参数组合、相同的断言模式），仅替换 `model_path` 和 MMLU 基线值。

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_cp_strategy_zigzag_mmlu` | `--enable-prefill-cp --cp-strategy zigzag --attn-cp-size 2 --tp-size 2`，**对比验证**：先后启动 zigzag / interleave server，各发 1 个 short factual prompt（正确性）+ 1 个 ~150 token long prompt（保证 CP split 触发），temperature=0 下两策略输出必须完全一致 | E2E | 对比验证 | 新增 | P0 |
| T2 | `test_enable_prefill_cp_alias` | `--enable-prefill-context-parallel --attn-cp-size 2 --tp-size 2`（弃用别名），验证框架转发→zigzag，short + long prompt 均推理正确 | E2E | 别名兼容 | 新增 | P1 |
| T3 | `test_enable_dsa_prefill_cp_alias` | `--enable-dsa-prefill-context-parallel --attn-cp-size 2 --tp-size 2`（DSA 弃用别名），验证框架转发→interleave，short + long prompt 均推理正确 | E2E | 别名兼容 | 新增 | P1 |
| T4 | `test_dsa_cp_mode_alias` | `--enable-prefill-cp --dsa-prefill-cp-mode in-seq-split --attn-cp-size 2 --tp-size 2`（mode 弃用别名），验证 `in-seq-split`→`zigzag` 映射，short + long prompt 均推理正确 | E2E | 别名兼容 | 新增 | P1 |

---

**T001: `test_cp_strategy_zigzag_mmlu`** (新增，对比验证)

**验证目标**: `--cp-strategy zigzag` 和 `interleave` 在 NPU TP=2 下均正确工作，两个不同代码路径（ZigzagCPStrategy / InterleaveCPStrategy）产生相同结果。

**测试步骤**:
1. 以 `--tp-size 2 --attn-cp-size 2 --enable-prefill-cp --cp-strategy zigzag --attention-backend ascend --disable-cuda-graph` 启动 Llama 3.2 1B Instruct server
2. `/generate` 发 short prompt（"The capital of France is", `temperature=0`），记录 `zigzag_short`
3. `/generate` 发 long prompt（~150 tokens water cycle 描述，保证 `num_tokens ≥ 2*cp_size=4` 触发 CP split），记录 `zigzag_long`
4. 停 server，以 `--cp-strategy interleave` 重启，同 2 条 prompt 记录 `interleave_short`、`interleave_long`

**断言**:
- `zigzag_short` 含 `"Paris"` AND `interleave_short` 含 `"Paris"`
- `zigzag_short == interleave_short`（temperature=0，短 prompt 下两策略输出一致）
- `zigzag_long == interleave_long`（temperature=0，长 prompt 下 CP split 必然触发，两策略输出仍一致 → 证明两条代码路径均正确）
- 两次 server 正常启动且无 crash

---

**T002–T004** (新增): 弃用别名转发 + 正确性验证。三个各自独立启动 server，每个发 short + long prompt，验证别名被 CLI 接受且推理正确。不依赖 T001 的输出（每个 test method 自包含）。

**T002: `test_enable_prefill_cp_alias`** — 参数: `--enable-prefill-context-parallel --attn-cp-size 2 --tp-size 2 --attention-backend ascend --disable-cuda-graph`
- `_handle_legacy_cp_arguments` → `enable_prefill_cp=True, cp_strategy="zigzag"`（来自 `prefill_cp_mode=in-seq-split`）
- **断言**: short prompt 含 `"Paris"`；long prompt 返回长度 > 10

**T003: `test_enable_dsa_prefill_cp_alias`** — 参数: `--enable-dsa-prefill-context-parallel --attn-cp-size 2 --tp-size 2 --attention-backend ascend --disable-cuda-graph`
- `_handle_legacy_cp_arguments` → `enable_dsa_prefill_context_parallel=True, cp_strategy="interleave"`（来自 `dsa_prefill_cp_mode=round-robin-split`）
- 对 Llama 非 DSA 模型 fell back 到通用 CP 路径
- **断言**: short prompt 含 `"Paris"`；long prompt 返回长度 > 10

**T004: `test_dsa_cp_mode_alias`** — 参数: `--enable-prefill-cp --dsa-prefill-cp-mode in-seq-split --attn-cp-size 2 --tp-size 2 --attention-backend ascend --disable-cuda-graph`
- legacy 映射 `in-seq-split` → `cp_strategy="zigzag"`
- **断言**: short prompt 含 `"Paris"`；long prompt 返回长度 > 10

---

#### 2.21 `--enable-precise-embedding-interpolation`

> **范围说明**: 该参数仅对 Qwen3-VL 模型生效（`Qwen3VLMoeVisionModel`），需 VLM 图像输入触发 vision encoder 路径。已存在的测试文件 `test_npu_embedding_interpolation.py` 使用 Llama 3.2 1B（纯文本模型），参数在文本模型上无效果，需改为 Qwen3-VL 模型重写。

##### 2.21.1 业务理解

- **定义**: `server_args.py:2081-2084`，boolean flag，控制 Qwen3-VL 视觉编码器中 ViT 位置嵌入网格插值时是否使用 corner alignment（角对齐），以获得更精确（但更慢）的 interpolated embedding 值
- **取值**:

  | 值 | 含义 |
  |---|---|
  | `False` (默认) | 半像素偏移插值：`(arange(n)+0.5)*(side/n)-0.5`，再 `clip(0, side-1)`，工业标准做法 |
  | `True` | Corner-aligned 插值：`np.linspace(0, side-1, dim_size)`，角点精确对齐原始网格，更慢但更准确 |

- **作用链路**: `qwen3_vl.py:327-329 → 464-473 → 528-542 → 1083-1091`

  ```
  server_args.enable_precise_embedding_interpolation
    → Qwen3VLMoeVisionModel.__init__(): self.align_corners = get_global_server_args().enable_precise_embedding_interpolation
      → _get_interpolation_indices(dim_size) [line 464]:
          align_corners=True  → np.linspace(0, side-1, dim_size)        ← 角对齐 (linspace)
          align_corners=False → (arange(n)+0.5)*(side/n)-0.5, clip     ← 半像素偏移
      → _torch_interp_indices(dim_size, device) [line 528]:  同上逻辑，PyTorch tensor 版本
      → fast_pos_embed_interpolate(grid_thw) [line 759]: 调用 _get_interpolation_indices → 逐图计算 bilinear 插值
      → _prepare_graph_inputs() [line 1067] (graph 路径):
          align_corners=True + vectorized_available → fast_pos_embed_interpolate_vectorized() [始终 linspace]
          否则 → fast_pos_embed_interpolate() [参数控制]
  ```

- **关键发现 — 参数仅在 Graph 路径生效**:

  | 代码路径 | 函数 | interpolation 方式 | 参数是否生效 |
  |---|---|---|---|
  | Eager (forward, line 879) | `fast_pos_embed_interpolate_from_list` (line 544) | 硬编码 `torch.linspace()` → 始终角对齐 | ❌ 不生效 |
  | Graph (`_prepare_graph_inputs`, line 1067) | `fast_pos_embed_interpolate` (line 759) | 调用 `_get_interpolation_indices(self.align_corners)` → 参数控制 | ✅ 生效 |
  | Graph + vectorized | `fast_pos_embed_interpolate_vectorized` (line 615) | 始终 linspace → 始终角对齐 | ❌ 不生效（但需 `align_corners=True` 才走此路径） |

  > NPU 默认启用 graph → `--enable-precise-embedding-interpolation` 在 **NPU graph 路径（vectorized 不可用时）** 生效。

- **依赖**:

  | 类型 | 内容 |
  |---|---|
  | 下游 | `Qwen3VLMoeVisionModel` 的 `_get_interpolation_indices`、`_torch_interp_indices`、`fast_pos_embed_interpolate` |
  | 关联参数 | `--enable-cuda-graph`（需 graph 启用；eager 路径中 `fast_pos_embed_interpolate_from_list` 硬编码 linspace，参数无效果） |
  | 前置条件 | ① 必须使用 Qwen3-VL 模型（**仅 `Qwen3VLMoeVisionModel` 读取此参数**）；② 需传入图像触发 vision encoder；③ 需 graph 模式启用（NPU 默认启用） |
  | 影响范围 | 仅 `qwen3_vl.py` 中 `Qwen3VLMoeVisionModel`。其他 VL 模型（deepseek_ocr、glm4v、internvl、paddleocr_vl、siglip2、radio、step3_vl 等）均硬编码 `align_corners`，不受此参数影响 |

##### 2.21.2 通俗理解

ViT 视觉编码器需要把固定大小的位置嵌入网格（如 48×48）"缩放"到不同尺寸的图像网格上。`--enable-precise-embedding-interpolation` 控制缩放时的采样起点。

- **默认 (`False`)**：从每个格子的**中心点**开始采样。就像在格子中间钉钉子——缩放后的采样点从格子中心偏移半像素。这是标准做法。
- **启用 (`True`)**：从网格的**角点**开始采样。每个角点精确对齐原始网格，缩放后的采样更精确但计算量稍大（`linspace(0, side-1, n)`）。
- **用户什么时候需要改**：使用 Qwen3-VL 处理高精度视觉任务（如 OCR、细粒度图像理解）时，启用此参数可提高 ViT 位置嵌入的插值精度，以微小性能代价换取更好的视觉特征对齐。

##### 2.21.3 GPU 社区用例分析

| 文件 | 类型 | 测试了什么 | 参数角色 | 覆盖值 | 可移植 |
|---|---|---|---|---|---|
| `test/registered/models/test_vit_pos_embed_interpolate.py` | UT (CPU+CUDA) | 验证 `fast_pos_embed_interpolate_vectorized` 与 `fast_pos_embed_interpolate_from_list` / `fast_pos_embed_interpolate` 的 **bit-exact 数值一致性**（5 种 grid 尺寸、bf16/fp32、Qwen3-VL + MossVL 两种模型）。测试的是插值函数的数值正确性，**不涉及 `--enable-precise-embedding-interpolation` server flag** | 非参数测试 — UT 内部直接调用插值函数，不经过 server args | 无（UT 内部直接调函数验证 bit-exact 一致性） | ✅ 可移植为 NPU UT，但**不覆盖参数 E2E 行为** |

> GPU **没有**该参数的 E2E 测试。UT 测试的是向量化插值加速的数值正确性（加速但不改变结果），与参数控制的 `align_corners` 插值模式选择是不同维度。

##### 2.21.4 E2E 测试分析

###### 2.21.4.1 测试因子分析

| 因子 | 取值 | 具体影响（代码行号） |
|---|---|---|
| `enable_precise_embedding_interpolation` | `False` (默认) | `qwen3_vl.py:464` → `_get_interpolation_indices`: `(arange(n)+0.5)*(side/n)-0.5`，clip 到 `[0, side-1]`。所有不传此 flag 的测试隐式覆盖了此路径。但隐式覆盖的前提是**使用了 Qwen3-VL 模型**——文本模型（Llama）不会触发此路径，不算覆盖 |
| | `True` | `qwen3_vl.py:465` → `np.linspace(0, side-1, dim_size)`，角对齐插值。与 `False` 走不同的 `_get_interpolation_indices` 分支（line 464 vs 465），产生的 interpolation indices 数值不同 → vision embeddings 不同 → 最终 logits 不同 |
| Graph 状态 | 启用（NPU 默认） | `forward()` line 884 → `forward_with_npu_graph()` → `_prepare_graph_inputs()` → `fast_pos_embed_interpolate()` 受参数控制 |
| | 禁用 | `forward()` line 899 → `fast_pos_embed_interpolate_from_list()` 硬编码 `torch.linspace()`，参数无效果 |
| 模型 | Qwen3-VL | 参数生效的唯一模型，读取 `get_global_server_args().enable_precise_embedding_interpolation` |
| | 其他 VL / 文本模型 | 参数无效果（不读取此 server arg） |

###### 2.21.4.2 用户场景分析

| 场景 | 用户场景（真实使用描述） | `--enable-precise-embedding-interpolation` 取值 | 验证点 |
|---|---|---|---|
| 默认插值 | 用户使用 Qwen3-VL 进行常规图像理解，不关心插值精度细节，使用默认的半像素偏移插值 | `False` (默认，不传) | VLM 图像推理成功，输出描述含有图像关键元素的文本 |
| 精确角对齐 | 用户使用 Qwen3-VL 处理高精度视觉任务（如 OCR 文字识别、细粒度物体检测），需要更精确的 ViT 位置嵌入插值，启用角对齐模式 | `True` (显式传参) | ① VLM 图像推理成功；② 与默认插值的输出存在差异（同一图像、temperature=0），证明参数改变了 vision embeddings 的计算路径 |
| 非 Qwen3-VL 模型 | 用户在 Llama/其他 VL 模型上误传此参数 | `True` | 参数被接受但不影响行为，推理正常（无 crash）——此场景通过文本模型的隐式覆盖验证，不单独新增测试 |

##### 2.21.5 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_precise_embedding_interpolation_enabled` | Qwen3-VL-4B + `--enable-precise-embedding-interpolation`，发送图像 QA，验证 VLM 输出含图像描述关键词 | E2E | 功能验证 | 新增（重写已有文件） | P0 |
| T2 | `test_precise_embedding_interpolation_contrastive` | 先后启动 server（带 flag / 不带 flag），同一图像 temperature=0 推理，验证两次输出不同，证明参数改变了 vision embeddings 计算路径 | E2E | 对比验证 | 新增 | P0 |

> ##### T001: `test_precise_embedding_interpolation_enabled`（新增）
> **验证目标**: `--enable-precise-embedding-interpolation` 启用后 Qwen3-VL 图像推理功能正常。
> **测试步骤**:
> 1. 以 `--enable-precise-embedding-interpolation --enable-multimodal --trust-remote-code --attention-backend ascend` 启动 Qwen3-VL-4B-Instruct server
> 2. 通过 `/v1/chat/completions` 发送 `IMAGE_MAN_IRONING_URL` 图像 + "Describe this image in a sentence."
> **断言**: `status_code == 200`；输出含 `man`/`person`/`driver` 中至少一个 AND `car`/`vehicle`/`SUV`/`cab`/`taxi` 中至少一个

> ##### T002: `test_precise_embedding_interpolation_contrastive`（新增，对比验证）
> **验证目标**: 同一 Qwen3-VL 模型、同一图像，带 flag 与不带 flag 的输出不同，证明参数确实改变了 ViT 插值行为。
> **测试步骤**:
> 1. 以 `--enable-precise-embedding-interpolation --enable-multimodal --trust-remote-code --attention-backend ascend` 启动 Qwen3-VL-4B-Instruct server
> 2. `/v1/chat/completions` 发送 `IMAGE_MAN_IRONING_URL` + "Describe this image in a sentence."，`temperature=0`，记录 `output_enabled`
> 3. 停掉 server
> 4. 以 `--enable-multimodal --trust-remote-code --attention-backend ascend`（不传 `--enable-precise-embedding-interpolation`）重启 server
> 5. 同样请求，`temperature=0`，记录 `output_default`
> **断言**: 两次 `status_code == 200`；`output_enabled` 和 `output_default` 均含图像描述关键词；`output_enabled != output_default`（证明参数改变了 vision embeddings 计算路径）

---

#### 2.22 `--gc-threshold`

##### 业务理解

- **定义**: `server_args.py:829-832`，控制 Python GC 三代分代回收的触发频率。接收 1~3 个整数，直接透传给 Python 内置 `gc.set_threshold()`。
- **取值**:

| 值 | 含义 |
|---|---|
| `None` (默认) | 不干预，使用 Python 默认阈值 `(700, 10, 10)` |
| `[t0]` (1 个整数) | 仅设 gen0 阈值：每分配 `t0` 个对象触发一次 gen0 GC |
| `[t0, t1]` (2 个整数) | gen0 每 `t0` 次分配触发；gen1 每 `t1` 次 gen0 收集后触发 |
| `[t0, t1, t2]` (3 个整数) | 同上；gen2 每 `t2` 次 gen1 收集后触发 |

- **作用链路**:

```
server_args.py:6909-6913   CLI 校验：1 ≤ len(gc_threshold) ≤ 3，且均为整数，否则 ValueError
       ↓
engine.py:786              _set_gc(server_args) — 引擎启动早期调用（在 server_args.check_server_args() 之后、端口分配之前）
       ↓
engine.py:1338-1342        _set_gc(): 如果 gc_threshold 非空，import gc 并调用 gc.set_threshold(*gc_threshold)
       ↓
Python gc 模块             gc.set_threshold(t0, t1, t2) 设置三代回收的分配阈值计数器
```

  关键细节：这是一个**纯副作用调用**（除了 `gc.set_threshold()` 什么也不做），参数的每个合法取值都汇聚到同一个 `gc.set_threshold()` 调用上，**不存在分叉代码路径**。

- **依赖**:
  - 下游: Python 解释器的内置 `gc` 模块（CPython 行为，无 sglang 自定义代码）
  - 关联参数: `--gc-warning-threshold-secs`（GC 耗时告警，已通过 `test_npu_gc_warning_threshold.py` 覆盖）、`SGLANG_LOG_GC` 环境变量（启用 GC 详细日志，在 `scheduler.py:1098-1099` 中调用 `configure_gc_logger()`）
  - 前置条件: 无

##### 通俗理解

Python 的垃圾回收（GC）采用**分代回收**机制，把对象分成三代：
- **gen0（年轻代）**：刚创建的对象，大多数很快就会被丢弃
- **gen1（中年代）**：活过了第一轮 GC 的对象
- **gen2（老年代）**：长期存活的对象

`--gc-threshold` 控制"攒够多少垃圾才打扫一次"：

| 示例 | GC 频率 | 典型场景 |
|---|---|---|
| `--gc-threshold 100`（低） | gen0 每 100 个新对象就回收一次 → 频繁 GC | 显存紧张、需要尽快释放废弃张量 |
| 不传参数（默认 ≈ `700,10,10`） | gen0 每 700 个对象回收一次 → 适中 | 通用场景，Python 官方默认值 |
| `--gc-threshold 5000,50,20`（高） | gen0 每 5000 个对象才回收 → 稀少 GC | GC 开销敏感、显存充裕的高吞吐场景 |

- **低阈值**: GC 频率高 → 更快释放不再用的张量/中间结果 → 显存占用更安全，但 GC 本身有 CPU 开销
- **高阈值**: GC 频率低 → 对象堆积更多 → GC 开销低，但显存峰值可能更高
- **用户什么时候需要改**: 显存紧张、模型大、batch 大或并发高时调低阈值；显存充裕、追求吞吐时可调高。可配合 `--gc-warning-threshold-secs` 观察 GC 耗时，配合 `--gc-warning-threshold-secs` 和 `/freeze_gc` API 在预热后冻结 GC 避免长期对象的扫描开销

##### GPU 社区用例分析

| 文件 | 类型 | 说明 |
|---|---|---|
| 无 | — | 0 个文件引用 `gc_threshold`。GPU 社区未为此参数设计用例 |

##### E2E 测试分析

###### 测试因子分析

| 因子 | 取值 | 影响 |
|---|---|---|
| `gc_threshold` 参数值 | 不传 (None) | 使用 Python 默认 `(700, 10, 10)`，`_set_gc()` 中 `gc_threshold` 为 None → 跳过 `gc.set_threshold()` |
| | 1 个整数 (如 `"100"`) | `gc.set_threshold(100)` → 仅设 gen0 = 100 |
| | 2 个整数 (如 `"100,5"`) | `gc.set_threshold(100, 5)` → 设 gen0=100, gen1=5 |
| | 3 个整数 (如 `"100,5,5"`) | `gc.set_threshold(100, 5, 5)` → 设全部三代 |
| GC 线程环境 | TokenizerManager 进程 | `gc_warning_threshold_secs > 0` 时注册回调 (`tokenizer_manager.py:559-560`)，由 `configure_gc_warning()` 监控 GC 耗时 |
| | Scheduler 进程 | `SGLANG_LOG_GC=1` 时启用 `configure_gc_logger()` (`scheduler.py:1098-1099`)，记录每次 GC 起止 |

> **关键结论**: 所有合法取值经 `gc.set_threshold(*values)` 处理，代码路径完全相同。不同取值仅改变 Python GC 内部计数器，无 sglang 级分支差异。按"举一反三"通用原则（两个值同路径只测一个），**不需要逐值覆盖**。测试重点应放在参数核心功能的副作用验证上——低阈值下 GC 更频繁但不破坏推理。

###### 用户场景分析

| 场景 | 用户场景 | `--gc-threshold` 取值 | 验证点 |
|---|---|---|---|
| 低阈值 GC 压力 | 用户为解决显存紧张问题，主动调低 GC 阈值，期望 GC 更频繁回收临时对象却不影响推理正确性 | `"50"`（极低阈值）vs 不传参数 | 对比验证：低阈值 GC 高频回收下推理仍正确完成、无 OOM，证明参数核心功能有效 |

##### 测试点设计

| ID | 方法名 | 测试内容 | 类型 | 分类 | 来源 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | `test_gc_threshold_low_vs_default` | 对比验证：`--gc-threshold 50` 极低阈值 vs 不传参数，发大量长 prompt 请求，验证推理正确且无 OOM | E2E | Parameter | 新增 | P0 |

**T001: `test_gc_threshold_low_vs_default`** (新增)

验证目标: 验证 `--gc-threshold` 的核心功能——极低阈值下 GC 高频回收不破坏推理正确性，且无 OOM。使用对比验证模式，低阈值 vs 默认值同条件对比，证明参数生效且安全。
测试步骤:
1. 启动 server，args `--gc-threshold 50 --disable-cuda-graph` + Llama 3.2 1B
2. 连续发送多次 `/generate` 请求（prompt 使用长文本如 "just return me a string with of 10000 characters: " + "A"*10000，产生大量临时对象触发 GC）
3. 重启 server，不传 `--gc-threshold` 作为对照组，发送相同请求
断言: 两组 `status_code == 200` 且输出内容一致；低阈值组不会因频繁 GC 导致推理中断或 OOM

---

### 最终结论

#### 文件清单

```
test/registered/ascend/basic_function/optimization_debug_options/
├── test_npu_cuda_graph_config.py                   (已有: --cuda-graph-config)
├── test_npu_cuda_graph_backend_decode.py           (已有: --cuda-graph-backend-decode)
├── test_npu_cuda_graph_backend_prefill.py          (已有: --cuda-graph-backend-prefill)
├── test_npu_cuda_graph_bs.py                       (已有: --cuda-graph-max-bs-decode、--cuda-graph-max-bs-prefill、--cuda-graph-bs-decode、--cuda-graph-bs-prefill)
├── test_npu_disable_cuda_graph.py                  (已有: --disable-prefill-cuda-graph、--disable-decode-cuda-graph、--disable-piecewise-cuda-graph)
├── test_npu_pre_warm_nccl.py                       (已有: --pre-warm-nccl)
├── test_npu_dp_attention.py                        (已有: --enable-dp-attention-local-control-broadcast)
├── test_npu_torch_compile_debug.py                 (已有: --enable-torch-compile-debug-mode)
├── test_npu_attn_tp_gather.py                      (已有: --disable-attn-tp-gather)
├── test_npu_embedding_interpolation.py             (重写: --enable-precise-embedding-interpolation，Qwen3-VL 模型 + 对比验证，已推送)
├── test_npu_prefill_cp.py                            (待创建: --cp-strategy zigzag/interleave 对比验证 + 弃用别名 --enable-prefill-context-parallel、--enable-dsa-prefill-context-parallel、--dsa-prefill-cp-mode 等价性验证)
└── test_npu_gc_threshold.py                        (待创建: --gc-threshold)
```

#### CI 注册汇总

| 参数 | 来源 | 文件 | CI Suite |
|---|---|---|---|
| `--cuda-graph-config` | 新增 E2E | `test_npu_cuda_graph_config.py` | `debug-full-1-npu-a3` |
| `--cuda-graph-backend-decode` | 移植 GPU E2E + 新增 E2E | `test_npu_cuda_graph_backend_decode.py` | `debug-full-1-npu-a3` |
| `--cuda-graph-backend-prefill` | 移植 GPU E2E + 新增 E2E | `test_npu_cuda_graph_backend_prefill.py` | `debug-full-1-npu-a3` |
| `--cuda-graph-max-bs-decode`、`--cuda-graph-max-bs-prefill`、`--cuda-graph-bs-decode`、`--cuda-graph-bs-prefill` | 新增 E2E 组合 | `test_npu_cuda_graph_bs.py` | `full-1-npu-a3` |
| `--disable-prefill-cuda-graph` | 新增 E2E | `test_npu_disable_cuda_graph.py` | `full-1-npu-a3` |
| `--disable-decode-cuda-graph` | 新增 E2E | `test_npu_disable_cuda_graph.py` | `full-1-npu-a3` |
| `--disable-piecewise-cuda-graph` | 新增 E2E | `test_npu_disable_cuda_graph.py` | `full-1-npu-a3` |
| `--pre-warm-nccl` | 新增 E2E | `test_npu_pre_warm_nccl.py` | `debug-full-2-npu-a3` |
| `--enable-dp-attention-local-control-broadcast` | 新增 E2E | `test_npu_dp_attention.py` | `debug-full-2-npu-a3` |
| `--enable-torch-compile-debug-mode` | 新增 E2E | `test_npu_torch_compile_debug.py` | `debug-full-1-npu-a3` |
| `--disable-attn-tp-gather` | 新增 E2E (OLMoE + `--moe-dense-tp-size 1` 对比验证) | `test_npu_attn_tp_gather.py` | `debug-full-1-npu-a3` |
| `--cp-strategy` | 新增 E2E 对比验证 (TP=2, ATTN_CP=2) — zigzag vs interleave short+long prompt 输出一致性 | `test_npu_prefill_cp.py` | `debug-full-2-npu-a3` |
| `--enable-dsa-prefill-context-parallel` | 弃用别名转发验证 (TP=2, ATTN_CP=2) | `test_npu_prefill_cp.py` | `debug-full-2-npu-a3` |
| `--enable-prefill-context-parallel` | 弃用别名转发验证 (TP=2, ATTN_CP=2) | `test_npu_prefill_cp.py` | `debug-full-2-npu-a3` |
| `--dsa-prefill-cp-mode` | 弃用别名转发验证 (TP=2, ATTN_CP=2) | `test_npu_prefill_cp.py` | `debug-full-2-npu-a3` |
| `--prefill-cp-mode` | 内部参数 (no_cli=True) | — | — |
| `--enable-precise-embedding-interpolation` | 重写 E2E（Qwen3-VL 模型 + SGLANG_VIT_ENABLE_CUDA_GRAPH + 对比验证） | `test_npu_embedding_interpolation.py` | `debug-full-1-npu-a3` |
| `--gc-threshold` | 新增 E2E | `test_npu_gc_threshold.py` | `debug-full-1-npu-a3` |

#### 统计

| 来源 | 数量 | 说明 |
|---|---|---|
| 新增 E2E | 18 | 已通过 CI (9): `--cuda-graph-config` + `--cuda-graph-backend-decode` full/disabled + `--cuda-graph-backend-prefill` disabled/breakable + cuda-graph-bs 四参数组合 + `--disable-prefill-cuda-graph`、`--disable-decode-cuda-graph`、`--disable-piecewise-cuda-graph` 别名验证 + `--pre-warm-nccl` + `--enable-dp-attention-local-control-broadcast` + `--enable-torch-compile-debug-mode` + `--disable-attn-tp-gather`；已推送 (1): `--enable-precise-embedding-interpolation`（Qwen3-VL + SGLANG_VIT_ENABLE_CUDA_GRAPH + 对比验证）；待创建 (9): `--gc-threshold` + `--cuda-graph-backend-decode` breakable/tc_piecewise + `--cuda-graph-backend-prefill` tc_piecewise + `test_npu_prefill_cp.py` (4 个 CP 测试点: zigzag+interleave 对比验证(1) + 3 弃用别名) |
| 移植 GPU E2E | 4 | `--cuda-graph-backend-decode` full、disabled + `--cuda-graph-backend-prefill` disabled、breakable（已通过 CI） |
| 无需测试 | 1 | `--prefill-cp-mode`（`no_cli=True`，不对外暴露，无法通过命令行传参） |

---

## 3. 设计自检

| # | 组 | 检查项 | ✅ |
|---|---|---|---|
| 1 | A-结构 | 两个特性齐全，每个参数含全量子章节 | ✅ |
| 2 | A-结构 | 所有取值列出 2 列含义表 | ✅ |
| 3 | A-结构 | E2E 场景表三列完整（场景/用户场景/验证点） | ✅ |
| 4 | A-结构 | 测试点表含"测试内容"列 | ✅ |
| 5 | B-内容 | GPU/NPU 用例逐个 Read 代码 | ✅ |
| 6 | B-内容 | 测试因子每个取值写明具体行为 | ✅ |
| 7 | B-内容 | 场景与验证点对应，不夸大 | ✅ |
| 8 | B-内容 | 来源列标注具体路径 | ✅ |
| 9 | B-内容 | 代码路径有语言描述，面向测试人员 | ✅ |
| 10 | C-覆盖 | 所有可选值有对应策略 | ✅ |
| 11 | C-覆盖 | 22 个参数全覆盖，顺序正确 | ✅ |
| 12 | C-覆盖 | 多值参数参考 GPU 设计模式 | ✅ |
| 13 | C-覆盖 | 默认值逐场景分析，不一刀切 | ✅ |
| 14 | C-覆盖 | Deprecated/别名已标注 | ✅ |
| 15 | C-覆盖 | 无"测试排除"独立章节 | ✅ |

---

## 4. CI 测试报告

### Run #28517138186 (2026-07-01)

| 文件 | 参数 | 注册器 | 结果 | 耗时 |
|---|---|---|---|---|
| `test_npu_cuda_graph_config.py` | `--cuda-graph-config` | full-1-npu-a3 | ✅ PASSED | 96s |
| `test_npu_cuda_graph_backend_prefill.py` | `--cuda-graph-backend-prefill` | full-1-npu-a3 | ✅ PASSED | 102s |
| `test_npu_cuda_graph_backend_decode.py` | `--cuda-graph-backend-decode` | full-1-npu-a3 | ✅ PASSED | 158s |
| `test_npu_model_config_parser.py` | `--model-config-parser` | full-1-npu-a3 | ✅ PASSED | 126s |
| `test_npu_tokenizer_backend.py` | `--tokenizer-backend` | full-1-npu-a3 | ✅ PASSED | 130s |

**5/5 全部通过**，无失败用例。注册器已从 `debug-full-1-npu-a3` 升级为 `full-1-npu-a3`。

---

### Run #28524502777 (2026-07-01) — PR #886

| 文件 | 参数 | Suite | 结果 | 耗时 |
|---|---|---|---|---|
| `test_npu_cuda_graph_bs.py` | `--cuda-graph-max-bs-decode`、`--cuda-graph-max-bs-prefill`、`--cuda-graph-bs-decode`、`--cuda-graph-bs-prefill` | debug-full-1-npu-a3 | ✅ PASSED | 160s |
| `test_npu_disable_cuda_graph.py` | `--disable-prefill-cuda-graph`、`--disable-decode-cuda-graph`、`--disable-piecewise-cuda-graph` | debug-full-1-npu-a3 | ✅ PASSED | 66s |

**2/2 全部通过**。注册器已从 `debug-full-1-npu-a3` 升级为 `full-1-npu-a3`。

---

## 5. 弃用参数汇总

| 旧参数 | 章节 | 别名目标 | 旧参数 GPU 覆盖 | 别名目标 GPU 覆盖 | 测试策略 |
|---|---|---|---|---|---|
| `--disable-piecewise-cuda-graph` | §2.10 | `--cuda-graph-backend-prefill=disabled` | 0 | 有（§2.3 已覆盖 disabled） | 组合测试已覆盖 |
| `--enable-dsa-prefill-context-parallel` | §2.15 | `--enable-prefill-cp` | 0 | `test_gqa_prefill_cp.py`(E2E)、`test_cp_prefix_len_fa3_parity.py`(E2E)、`test_deepseek_v4_pro_fp4_cp.py`(E2E)、`test_server_args.py`(UT) | TP=2 别名转发+等价性验证（合并入 `test_npu_prefill_cp.py`） |
| `--enable-prefill-context-parallel` | §2.16 | `--enable-prefill-cp` | 0 | 同上 | TP=2 别名转发+等价性验证（合并入 `test_npu_prefill_cp.py`） |
| `--dsa-prefill-cp-mode` | §2.17 | `--cp-strategy` | 0 | `test_cp_strategy_unit.py`(UT)、`test_gqa_prefill_cp.py`(E2E)、`test_deepseek_v4_pro_fp4_cp.py`(E2E) | TP=2 别名转发+等价性验证（合并入 `test_npu_prefill_cp.py`） |
| `--prefill-cp-mode` | §2.18 | `--cp-strategy` | 3 个文件（内部引用） | 同上 | 不对外暴露（`no_cli=True`），无法通过命令行传参 |
