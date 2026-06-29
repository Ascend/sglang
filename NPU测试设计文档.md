# Model and Tokenizer 特性 — NPU 测试设计文档

## 1. 特性大类业务理解

**Model and Tokenizer** 特性负责 SGLang 的两大基础能力：模型加载和 token 转换。

```
用户请求(text) → [Tokenizer] → token_ids → [Engine 推理] → token_ids → [Detokenizer] → text → 返回用户
```

涉及的参数分为三类：

| 类别 | 参数 | 说明 |
|---|---|---|
| Tokenizer 选型 | `--tokenizer-backend`、`--tokenizer-mode` | 控制 tokenizer 底层库和 fast/slow 模式 |
| 并行度控制 | `--tokenizer-worker-num`、`--detokenizer-worker-num` | 控制 tokenizer/detokenizer 并行 worker 数量 |
| 模型加载 | `--model-config-parser`、`--model-impl` | 控制模型配置解析策略和模型实现选型 |

本次覆盖 3 个尚无 NPU 测试的参数：`--tokenizer-backend`、`--detokenizer-worker-num`、`--model-config-parser`。

---

## 2. 逐参数分析

### 2.1 `--tokenizer-backend`

#### 业务理解

- **定义**: 控制 tokenizer 底层库选型 (`server_args.py:434-442`)
- **取值**: `"huggingface"` (默认) | `"fastokens"`
- **作用链路**:

  ```
  server_args.tokenizer_backend
    → get_tokenizer(tokenizer_backend=...)
      → "huggingface" → HuggingFace 原生 tokenizers 库（AutoTokenizer）
      → "fastokens"   → _ensure_fastokens_patched() 猴子补丁 transformers，
                        用 fastokens 库替换底层 tokenizer 实现
  ```

- **同时传给** `get_processor()` 影响多模态 Processor 的 tokenizer 加载
- **fastokens 前置条件**: `pip install fastokens`

#### 通俗理解

> **Tokenizer 就像翻译官，把人类语言翻译成模型能理解的数字（token ids）。`--tokenizer-backend` 选择的是"哪个翻译官"来干活。**
>
> - **`huggingface`（默认）**: HuggingFace 官方翻译官。最稳定、兼容性最好，所有模型都能用。
> - **`fastokens`**: 快手翻译官。用 `fastokens` 库替换 HuggingFace 底层实现，高并发场景下提速明显，但需额外安装 `pip install fastokens`。
>
> **对用户的影响**: 高并发推理时，`fastokens` 可降低 tokenization 延迟。普通单用户场景无需关心。

#### GPU 社区用例分析

| 文件 | 位置 | 测试了什么 |
|---|---|---|
| `test_hf_transformers_fastokens.py` | `test/registered/unit/utils/` | **T1** `test_shim_is_applied`: 验证 `get_tokenizer(model, tokenizer_backend="fastokens")` 后，tokenizer 内部 `._tokenizer` 被替换为 fastokens 的 `_TokenizerShim` 实例，确认注入成功。**T2** `test_encode_decode_roundtrip`: 验证 encode → decode 回环正确。**注**: GPU 版有 `@unittest.skipUnless(HAS_FASTOKENS)` 跳过未安装 fastokens 的场景。 |

#### 测试点设计

| ID | 方法名 | 分类 | 输入 | 预期 | 来源 |
|---|---|---|---|---|---|
| T1 | `test_fastokens_shim_is_applied_npu` | 注入验证 | `get_tokenizer(TOKENIZER_MODEL, tokenizer_backend="fastokens")` | `._tokenizer` 是 `_TokenizerShim` 实例 | **移植**（去掉 skip） |
| T2 | `test_fastokens_encode_decode_roundtrip_npu` | 回环验证 | fastokens tokenizer 编码 → 解码 | 解码 == 原文 | **移植**（去掉 skip） |
| T3 | `test_tokenizer_backend_fastokens` | 边界 | `--tokenizer-backend fastokens` 启动 server + generate | 200，含 "Paris" | **新增** |

> 默认值 `huggingface` 不单独测试：所有不显式指定 `--tokenizer-backend` 的 NPU 用例均已隐式覆盖。

**结论**: 移植 GPU 用例 (T1,T2) + 新增 E2E (T3)，合并为一个文件。

---

### 2.2 `--detokenizer-worker-num`

#### 业务理解

- **定义**: 控制 detokenizer 进程并行度 (`server_args.py:444`)
- **取值**: `int`，默认 `1`，必须 `>= 1`（`server_args.py:6822` assert 校验）
- **作用链路** (`engine.py:705-753`):

  ```
  detokenizer_worker_num <= 1  → 单进程 DetokenizerManager
  detokenizer_worker_num > 1   → N 个 DetokenizerManager worker + 1 个 MultiDetokenizerRouter
                                  Router 持有原始 IPC，按请求 key hash 分发到不同 worker
                                  worker 通过 MultiHttpWorkerDetokenizerMixin 回传结果
  ```

- **特殊规则**: `skip_tokenizer_init=True` 时强制 `detokenizer_worker_num=1`
- **关联参数**: `--tokenizer-worker-num`（配合使用，已有 NPU 测试覆盖 tokenizer 端）

#### 通俗理解

> **Detokenizer 是"解码员"，把推理产出的 token ids 翻译回人类语言。`--detokenizer-worker-num` 控制有多少个解码员同时工作。**
>
> - **`= 1`（默认）**: 就一个解码员，所有请求排队等他一个人处理。类似快递驿站只有 1 个取件员。
> - **`> 1`（如 4）**: 雇佣 4 个解码员，Router 把请求按 hash 分发到不同解码员并行处理。类似驿站开 4 个取件窗口。
>
> | | 1 Worker | 多 Worker |
> |---|---|---|
> | 拓扑 | DetokenizerManager 直接处理 | N×Worker + Router 分发 |
> | 负载 | 单点串行 | hash 分发，并行处理 |
> | 适用 | 低并发 | 高并发 |
>
> **对用户的影响**: 高并发场景下，增大该值可减少 detokenizer 排队等待时间。

#### GPU 社区用例分析

| 文件 | 位置 | 测试了什么 |
|---|---|---|
| `test_multi_detokenizer.py` | `test/registered/tokenizer/` | **T1** `test_multi_detokenizer_ttft`: 启动 server (`--detokenizer-worker-num=4 --tokenizer-worker-num=8`)，用 `DEFAULT_MODEL_NAME_FOR_TEST` (8B)，100 prompts benchmark，验证 E2E 延迟 <11000ms、TTFT <86ms、ITL <10ms。继承 MMLUMixin 确保精度≥0.65。CUDA+AMD 双平台。 |

#### 测试点设计

| ID | 方法名 | 分类 | 输入 | 预期 | 来源 |
|---|---|---|---|---|---|
| T1 | `test_multi_detokenizer_ttft_npu` | 边界/性能 | `--detokenizer-worker-num=4 --tokenizer-worker-num=4` + Llama 1B + benchmark | 推理成功，TTFT 在合理范围 | **移植适配** |

**结论**: 移植适配 GPU 用例，默认值 1 在其他用例中已有覆盖，无需重复测试。

---

### 2.3 `--model-config-parser`

#### 业务理解

- **定义**: 控制模型配置加载策略 (`server_args.py:510-520`)
- **取值**: `"auto"` (默认) | `"hf"` | 插件注册的自定义名
- **解析链路** (`config.py:215-248`, `model_config_parser_registry.py`):

  ```
  server_args.model_config_parser
    → get_config(model_config_parser=...)
      → "auto"  → is_mistral_model() 启发式判断 → "mistral" 或 "hf"
      → "hf"    → AutoConfig.from_pretrained(config.json)
      → 其他     → get_model_config_parser(name) 从注册表取自定义解析器
  ```

- **插件体系**: `@register_model_config_parser("my_parser")` → `ModelConfigParserBase` 子类 → `parse()` 方法
- **GGUF 模型**: 强制 `"hf"`，无视 auto 解析

#### 通俗理解

> **每个模型有份"使用说明书" (`config.json`)，记录了模型结构（层数、隐藏维度等关键参数）。`--model-config-parser` 选择"谁来读这份说明书"。**
>
> - **`auto`（默认）**: 自动选择阅读器。模型名含 "Mistral" → 派 Mistral 专用阅读器；其他 → 派 HuggingFace 通用阅读器。
> - **`hf`**: 强制用 HuggingFace 通用阅读器（`AutoConfig.from_pretrained`）。
> - **插件自定义**: 通过 `@register_model_config_parser("xxx")` 注册的特殊阅读器，适配非标模型格式。
>
> **对用户的影响**: 默认 `auto` 对绝大多数模型都正确。只在模型加载报 config 相关错误时需要手动指定。比如某模型 config 格式特殊，`auto` 选错了解析器导致加载失败，就需显式指定 `hf` 或插件名。

#### GPU 社区用例分析

| 文件 | 位置 | 测试了什么 |
|---|---|---|
| `test_model_config_parser_registry.py` | `test/registered/unit/configs/` | **T1** `test_register_then_get_roundtrip`: 验证注册 FakeParser → `get_model_config_parser("fake")` 返回正确实例。**T2** `test_register_rejects_non_subclass`: 验证非 ModelConfigParserBase 子类注册报 ValueError。**T3** `test_unknown_name_raises_with_registered_list`: 验证查不存在的注册名报错 + 错误信息含已注册列表。纯单元测试，验证注册表 API。 |

#### 测试点设计

| ID | 方法名 | 分类 | 输入 | 预期 | 来源 |
|---|---|---|---|---|---|
| T1 | `test_register_then_get_roundtrip_npu` | 注册验证 | `register("fake")(FakeParser)` → `get("fake")` | 返回 FakeParser 实例 | **移植** |
| T2 | `test_register_rejects_non_subclass_npu` | 异常验证 | 注册非子类 | 抛 ValueError | **移植** |
| T3 | `test_unknown_name_raises_with_registered_list_npu` | 异常验证 | `get_model_config_parser("does-not-exist")` | 抛 ValueError，含 "fake" | **移植** |
| T4 | `test_model_config_parser_auto` | 正常路径 | `--model-config-parser auto` 启动 + generate | 200，推理成功 | **新增** |
| T5 | `test_model_config_parser_hf` | 边界 | `--model-config-parser hf` 启动 + generate | 200，推理成功 | **新增** |

**结论**: 移植 GPU 注册表单元测试 (T1,T2,T3) + 新增 E2E 覆盖 `auto`/`hf` (T4,T5)，合并为一个文件。

---

## 3. 最终结论

| 参数 | 输出文件 | 移植 | 新增 | 说明 |
|---|---|---|---|---|
| `--tokenizer-backend` | `test_npu_tokenizer_backend.py` | 2 | 1 | GPU unit + fastokens E2E (默认值已由其他用例覆盖) |
| `--detokenizer-worker-num` | `test_npu_detokenizer_worker_num.py` | 1 | 0 | GPU 移植适配 |
| `--model-config-parser` | `test_npu_model_config_parser.py` | 3 | 2 | GPU unit + server E2E |

### 文件清单

```
test/registered/ascend/basic_function/model_tokenizer/
├── NPU测试设计文档.md                      (本文档)
├── test_npu_model_tokenizer.py             (已有)
├── test_npu_model_tokenizer_multimodal.py  (已有)
├── test_npu_openai_embedding.py            (已有)
├── test_npu_tokenizer_backend.py           (新建)
├── test_npu_detokenizer_worker_num.py      (新建)
└── test_npu_model_config_parser.py         (新建)
```

### 移植 vs 新增汇总

| 来源 | 数量 | 说明 |
|---|---|---|
| 移植 GPU（去掉 skip） | 6 | fastokens shim×2、detokenizer MMLU×1、model_config_parser 注册表×3 |
| 新增 NPU E2E | 3 | tokenizer_backend (fastokens)、model_config_parser (auto + hf) |
| 无需测试 | 5 | 无效值×3 + 默认值(detokenizer=1 已有覆盖) + 默认值(tokenizer_backend=huggingface 已有覆盖) |

---

## 4. 测试排除说明

以下场景经分析后**不纳入测试范围**：

| 问题 | 结论 | 理由 |
|---|---|---|
| `tokenizer_backend` 同时传给 `get_processor()`，是否需要测试多模态模型？ | 不需要 | `get_processor()` 底层调用同一 `get_tokenizer()`，不触发新代码分支，且 `test_npu_model_tokenizer_multimodal.py` 已覆盖多模态 tokenizer 场景 |
| fastokens 降低高并发 tokenization 延迟，是否需要测试高并发？ | 不需要 | fastokens 功能正确性已由 T1(shim 注入) + T2(encode/decode 回环) 验证。高并发下 fastokens 差异是性能量变，非功能质变。并发路径已由 `test_npu_detokenizer_worker_num.py` (100 prompts benchmark) 覆盖 |

## 5. 设计自检

```
[√] 每个参数的所有可选值都有测试覆盖
[√] GPT 社区用例已分析并移植
[√] 移植策略明确（来源标注）
[√] 端到端测试优先
[√] 无冗余测试（透传参数不重复测、纯性能差异不测）
[√] 多模态不需要单独测（同一调用链）
[√] 并发不需要单独测（已有 benchmark 覆盖）
```
