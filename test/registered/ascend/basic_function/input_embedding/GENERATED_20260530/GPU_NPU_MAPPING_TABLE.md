# GPU与NPU Input Embedding测试覆盖分析（完整报告）

**分析日期**: 2026-05-30

**分析范围**: test/registered/input_embedding/ (仅此目录)

**生成器**: npu-test-gap-v9.1 skill

---

## 1. 排除测试（仅单元测试）

本目录无单元测试排除。所有测试文件均为集成测试。

---

## 2. GPU集成测试摘要

| GPU测试文件 | 测试类 | 测试类型 | 模型 | 配置 | 测试场景 |
|------------|--------|---------|------|------|---------|
| test_input_embeddings.py | TestInputEmbeds | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | --disable-radix, --cuda-graph-max-bs 4 | 基本embedding功能测试 |
| test_input_embeds_chunked.py | TestInputEmbedsChunkedAndRetract | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | --disable-radix-cache, --chunked-prefill-size 256, --cuda-graph-max-bs 4 | 分块预填充与回退测试 |

**测试详情**：

### test_input_embeddings.py
- **测试方法**：
  - `test_text_based_response()`: 测试文本输入API响应
  - `test_embedding_based_response()`: 测试embedding输入API响应
  - `test_compare_text_vs_embedding()`: 对比文本和embedding输入结果
  - `test_generate_from_file()`: 测试文件输入API响应
- **功能**：验证API支持多种输入格式（文本、embedding、文件）并成功推理

### test_input_embeds_chunked.py
- **测试方法**：
  - `test_chunked_prefill_truncation_and_continuation()`: 回归测试#20376，单请求分块截断和延续
  - `test_chunked_prefill_batch_truncation()`: 回归测试#20376，批处理分块截断
  - `test_retraction_with_output_ids()`: 回归测试#14110，带output_ids的回退
- **功能**：验证input_embeds在分块预填充和回退场景下的shape匹配问题
- **特殊配置**：使用SGLANG_TEST_RETRACT环境变量强制周期性回退

---

## 3. NPU现有测试摘要

| NPU测试文件 | 测试类 | 模型 | 配置 | 测试场景 | 状态 |
|------------|--------|------|------|---------|------|
| test_npu_input_embeddings.py | TestInputEmbeds | LLAMA_3_2_1B_INSTRUCT | --disable-radix, --cuda-graph-max-bs 4, --attention-backend ascend, --disable-cuda-graph | 基本embedding功能测试 | 已存在（位置需调整） |

**位置问题**：
- 现有NPU测试位于 `ascend/basic_function/parameter/` 目录
- 建议迁移至 `ascend/basic_function/input_embedding/` 目录以保持一致性

---

## 4. GPU-NPU测试映射表（完整）

| 序号 | GPU测试文件 | GPU测试类 | 测试类型 | 模型 | NPU测试文件 | NPU测试类 | 映射状态 | NPU状态 | 关键适配说明 |
|-----|------------|----------|---------|------|------------|----------|---------|---------|-------------|
| 1 | test_input_embeddings.py | TestInputEmbeds | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | test_npu_input_embeddings.py | TestInputEmbeds | ✅ 已适配 | 已存在 | 模型：DEFAULT_SMALL_MODEL_NAME_FOR_TEST → LLAMA_3_2_1B_INSTRUCT；后端：添加ascend后端和禁用cuda graph |
| 2 | test_input_embeds_chunked.py | TestInputEmbedsChunkedAndRetract | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | test_npu_input_embeds_chunked.py | TestNPUInputEmbedsChunkedAndRetract | ✅ 已适配 | 已生成 | 模型：DEFAULT_SMALL_MODEL_NAME_FOR_TEST → LLAMA_3_2_1B_INSTRUCT；后端：添加ascend后端和禁用cuda graph |

**适配说明**：
- "模型：DEFAULT_SMALL_MODEL_NAME_FOR_TEST → LLAMA_3_2_1B_INSTRUCT（NPU标准小模型）"
- "后端适配：添加 --attention-backend ascend --disable-cuda-graph（NPU必需参数）"
- "测试逻辑保持一致：保持所有测试方法和断言不变"

---

## 5. 覆盖率统计

**生成前**：
- GPU测试数量：2
- NPU测试数量：1（位置需调整）
- 覆盖率：**50%** (1/2)

**生成后**：
- GPU测试数量：2
- NPU测试数量：2（已生成1个新测试）
- 支持的测试：2
- 不支持的测试：0
- **有效覆盖率**：**100%** (2/2)

---

## 6. 差距分析矩阵

| GPU测试 | NPU支持原因 | 状态 | 所需操作 |
|--------|------------|------|---------|
| test_input_embeddings.py | ✅ 模型可用，功能支持 | 已存在 | 建议迁移至正确目录 |
| test_input_embeds_chunked.py | ✅ 模型可用，功能支持 | 已生成 | 运行测试验证 |

---

## 7. NPU测试增强机会

### 7.1 模型可用性问题
无问题。LLAMA_3_2_1B_INSTRUCT模型在NPU上可用。

### 7.2 算法支持
Input Embedding功能在NPU上完全支持，无算法差异。

### 7.3 后端考虑
- 必须使用 `--attention-backend ascend`
- 必须使用 `--disable-cuda-graph`
- 分块预填充大小可调整以适应NPU内存

### 7.4 量化方案
测试使用FP16/FP32，无需量化适配。

---

## 8. 推荐测试生成优先级

### 阶段1（已完成）
| 优先级 | GPU测试 | 功能 | NPU适配 | 模型路径 | 配置 | 状态 |
|-------|--------|------|--------|---------|------|------|
| 高 | test_input_embeds_chunked.py | 分块预填充与回退 | 模型+后端适配 | LLAMA_3_2_1B_INSTRUCT | --disable-radix-cache, --chunked-prefill-size 256 | ✅ 已生成 |

### 阶段2（建议迁移）
| 优先级 | NPU测试 | 当前位置 | 建议位置 | 状态 |
|-------|--------|---------|---------|------|
| 中 | test_npu_input_embeddings.py | parameter/ | input_embedding/ | 建议迁移 |

---

## 9. NPU关键适配说明

### 9.1 算法适配
无需特殊算法适配。Input Embedding功能在NPU上原生支持。

### 9.2 后端适配
```bash
# NPU必需参数
--attention-backend ascend    # 使用Ascend后端
--disable-cuda-graph          # 禁用CUDA图（NPU不支持）
```

### 9.3 模型适配
- GPU测试使用 `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`
- NPU测试使用 `LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH`（来自test_ascend_utils.py）

### 9.4 并行策略
测试无需特殊并行策略，单卡即可运行。

### 9.5 评估阈值
测试主要验证功能正确性，无精度阈值要求。

### 9.6 环境变量
- `SGLANG_TEST_RETRACT=True`: 用于test_retraction_with_output_ids测试，强制周期性回退

### 9.7 超时调整
- est_time=400秒: NPU测试预估时间
- 超时时间DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH由测试框架自动管理

---

## 10. 生成的NPU测试场景

### 10.1 test_npu_input_embeds_chunked.py
**TestNPUInputEmbedsChunkedAndRetract**

**测试类别**: 集成测试（回归测试）

**测试目标**:
- 分块预填充截断和延续（#20376）
- 批处理分块截断（#20376）
- 带output_ids的回退（#14110）

**测试方法**:
1. `test_chunked_prefill_truncation_and_continuation()`: 验证单请求超过chunked_prefill_size时的截断和延续
2. `test_chunked_prefill_batch_truncation()`: 验证批处理请求的分块截断
3. `test_retraction_with_output_ids()`: 验证回退后input_embeds与output_ids的shape匹配

**服务配置**:
```
--disable-radix-cache
--chunked-prefill-size 256
--cuda-graph-max-bs 4
--attention-backend ascend
--disable-cuda-graph
```

**关键测试逻辑**:
- 使用SGLANG_TEST_RETRACT=True强制周期性回退
- 生成超过CHUNKED_PREFILL_SIZE的embedding序列
- 验证API返回状态码200和响应文本

---

## 11. 运行测试

### 11.1 运行所有生成测试
```bash
python -m unittest sglang518.test.registered.ascend.basic_function.input_embedding.GENERATED_20260530.test_npu_input_embeds_chunked.TestNPUInputEmbedsChunkedAndRetract
```

### 11.2 运行特定测试方法
```bash
# 运行分块预填充截断测试
python -m unittest sglang518.test.registered.ascend.basic_function.input_embedding.GENERATED_20260530.test_npu_input_embeds_chunked.TestNPUInputEmbedsChunkedAndRetract.test_chunked_prefill_truncation_and_continuation

# 运行批处理截断测试
python -m unittest sglang518.test.registered.ascend.basic_function.input_embedding.GENERATED_20260530.test_npu_input_embeds_chunked.TestNPUInputEmbedsChunkedAndRetract.test_chunked_prefill_batch_truncation

# 运行回退测试
python -m unittest sglang518.test.registered.ascend.basic_function.input_embedding.GENERATED_20260530.test_npu_input_embeds_chunked.TestNPUInputEmbedsChunkedAndRetract.test_retraction_with_output_ids
```

### 11.3 语法验证
```bash
python -m py_compile sglang518/test/registered/ascend/basic_function/input_embedding/GENERATED_20260530/test_npu_input_embeds_chunked.py
```

---

## 12. 后续工作

### 12.1 受阻任务（模型权重）
无受阻任务。所有模型权重可用。

### 12.2 阈值验证
- 验证分块预填充功能在NPU上的正确性
- 验证回退机制在NPU上的正确性
- 验证embedding输入格式在NPU上的兼容性

### 12.3 扩展测试
- 考虑添加更多边界条件的embedding测试
- 考虑添加不同embedding维度的测试
- 考虑添加embedding与文本混合输入的测试

### 12.4 集成工作
- 将现有test_npu_input_embeddings.py从parameter/目录迁移至input_embedding/目录
- 将生成的测试集成到CI流水线
- 配置nightly测试套件

### 12.5 目录结构优化
建议调整目录结构以保持一致性：
```
test/registered/ascend/basic_function/input_embedding/
├── GENERATED_20260530/
│   ├── test_npu_input_embeds_chunked.py
│   └── GPU_NPU_MAPPING_TABLE.md
└── test_npu_input_embeddings.py  # 从parameter/迁移
```

---

## 13. 总结

当前NPU Input Embedding测试覆盖率已从 **50%** 提升至 **100%**，共生成1个新测试。

**主要成果**：
1. ✅ 生成test_npu_input_embeds_chunked.py，覆盖分块预填充和回退场景
2. ✅ 所有GPU测试均已适配为NPU测试
3. ✅ 使用标准NPU模型路径和后端配置

**建议改进**：
1. 将test_npu_input_embeddings.py迁移至正确目录
2. 在CI中配置nightly测试
3. 定期验证测试通过率

---

## 14. 生成的文件

| 文件 | 目录 | 描述 |
|-----|------|------|
| test_npu_input_embeds_chunked.py | GENERATED_20260530/ | 分块预填充与回退测试 |
| GPU_NPU_MAPPING_TABLE.md | GENERATED_20260530/ | 完整分析报告（本文件） |

**完整路径**:
```
sglang518/test/registered/ascend/basic_function/input_embedding/GENERATED_20260530/
├── test_npu_input_embeds_chunked.py
└── GPU_NPU_MAPPING_TABLE.md
```

**现有文件（建议迁移）**:
```
sglang518/test/registered/ascend/basic_function/parameter/test_npu_input_embeddings.py
```
建议迁移至：
```
sglang518/test/registered/ascend/basic_function/input_embedding/test_npu_input_embeddings.py
```
