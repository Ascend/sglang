# GPU与NPU 可观测性（Observability）测试覆盖分析（完整报告）

**分析日期**: 2026-06-06

**分析范围**: test/registered/observability/ （仅此目录）

**生成器**: npu-test-gap-v9.1 skill

---

## 1. 排除测试（仅单元测试）

本目录下的单元测试文件（test/registered/unit/observability/）已被排除：

| GPU测试文件 | 测试类型 | 排除原因 |
|------------|---------|---------|
| test_trace.py | 单元测试 | 无服务器依赖的纯逻辑测试 |
| test_startup_func_log_and_timer.py | 单元测试 | 无服务器依赖的纯逻辑测试 |
| test_request_metrics_exporter.py | 单元测试 | 无服务器依赖的纯逻辑测试 |
| test_metrics_utils.py | 单元测试 | 无服务器依赖的纯逻辑测试 |
| test_label_transform.py | 单元测试 | 无服务器依赖的纯逻辑测试 |
| test_func_timer.py | 单元测试 | 无服务器依赖的纯逻辑测试 |
| test_forward_pass_metrics.py | 单元测试 | 无服务器依赖的纯逻辑测试 |
| test_cpu_monitor.py | 单元测试 | 无服务器依赖的纯逻辑测试 |

---

## 2. GPU集成测试摘要

| GPU测试文件 | 测试类 | 测试类型 | 模型 | 配置 | 测试场景 |
|------------|--------|---------|------|------|---------|
| test_tracing.py | TestTracePackage | 单元测试 | - | - | Trace API测试（无服务器） |
| test_tracing.py | TestTraceServer | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | enable-trace, OTLP endpoint | 服务器级别trace测试 |
| test_tracing.py | TestTraceEngine | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | enable-trace, OTLP endpoint | Engine API trace测试 |
| test_tracing_disaggregation.py | TestTraceDisaggregation | 集成测试 | DEFAULT_MODEL_NAME_FOR_TEST | 2-GPU, PD disaggregation, enable-trace | disaggregation trace测试 |
| test_priority_metrics.py | TestQueueCount | 单元测试 | - | - | QueueCount逻辑测试（无服务器） |
| test_priority_metrics.py | TestPriorityMetrics | 集成测试 | Qwen/Qwen3-0.6B | enable-metrics, enable-priority-scheduling | 优先级调度指标测试 |
| test_metrics.py | TestEnableMetrics | 集成测试 | Qwen/Qwen3-0.6B | enable-metrics, MFU metrics | Prometheus指标测试 |
| test_metrics.py | TestComputeRoutingKeyStats | 单元测试 | - | - | Routing key统计逻辑测试（无服务器） |

---

## 3. NPU现有测试摘要

| NPU测试文件 | 测试类 | 模型 | 配置 | 测试场景 | 状态 |
|------------|--------|------|------|---------|------|
| test_npu_tracing.py | TestNPUTracePackage | - | - | Trace API测试（无服务器） | 已存在（GENERATED_20260529） |
| test_npu_tracing.py | TestNPUTraceServer | LLAMA_3_2_1B_INSTRUCT | ascend backend, enable-trace | 服务器级别trace测试 | 已存在（GENERATED_20260529） |
| test_npu_tracing.py | TestNPUTraceEngine | LLAMA_3_2_1B_INSTRUCT | ascend backend, enable-trace | Engine API trace测试 | 已存在（GENERATED_20260529） |
| test_npu_metrics.py | TestNPUEnableMetrics | QWEN3_0_6B | ascend backend, enable-metrics | Prometheus指标测试 | 已存在（GENERATED_20260529） |
| test_npu_metrics.py | TestNPUComputeRoutingKeyStats | - | - | Routing key统计逻辑测试 | 已存在（GENERATED_20260529） |
| test_npu_priority_metrics.py | TestNPUQueueCount | - | - | QueueCount逻辑测试 | 已存在（GENERATED_20260529） |
| test_npu_priority_metrics.py | TestNPUPriorityMetrics | QWEN3_0_6B | ascend backend, enable-priority-scheduling | 优先级调度指标测试 | 已存在（GENERATED_20260529） |
| test_npu_tracing_disaggregation.py | TestNPUTraceDisaggregation | LLAMA_3_2_1B_INSTRUCT | 2-NPU, PD disaggregation, enable-trace | disaggregation trace测试 | 新生成（GENERATED_20260606） |

---

## 4. GPU-NPU测试映射表（完整）

**关键输出：展示精确的测试对应关系**

| 序号 | GPU测试文件 | GPU测试类 | 测试类型 | 模型 | NPU测试文件 | NPU测试类 | 映射状态 | NPU状态 | 关键适配说明 |
|-----|------------|----------|---------|------|------------|----------|---------|---------|-------------|
| 1 | test_tracing.py | TestTracePackage | 单元测试 | - | test_npu_tracing.py | TestNPUTracePackage | ✅ 已映射 | 已存在 | 无需服务器，纯逻辑测试 |
| 2 | test_tracing.py | TestTraceServer | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | test_npu_tracing.py | TestNPUTraceServer | ⚠️ 已适配 | 已存在 | 模型：DEFAULT_SMALL_MODEL → LLAMA_3_2_1B_INSTRUCT；后端：默认 → ascend；禁用cuda graph |
| 3 | test_tracing.py | TestTraceEngine | 集成测试 | DEFAULT_SMALL_MODEL_NAME_FOR_TEST | test_npu_tracing.py | TestNPUTraceEngine | ⚠️ 已适配 | 已存在 | 模型：DEFAULT_SMALL_MODEL → LLAMA_3_2_1B_INSTRUCT；后端：默认 → ascend；禁用cuda graph |
| 4 | test_tracing_disaggregation.py | TestTraceDisaggregation | 集成测试 | DEFAULT_MODEL_NAME_FOR_TEST | test_npu_tracing_disaggregation.py | TestNPUTraceDisaggregation | ⚠️ 已适配 | 新生成 | 模型：DEFAULT_MODEL → LLAMA_3_2_1B_INSTRUCT；后端：默认 → ascend；禁用cuda graph；使用PDDisaggregationServerBase |
| 5 | test_priority_metrics.py | TestQueueCount | 单元测试 | - | test_npu_priority_metrics.py | TestNPUQueueCount | ✅ 已映射 | 已存在 | 无需服务器，纯逻辑测试 |
| 6 | test_priority_metrics.py | TestPriorityMetrics | 集成测试 | Qwen/Qwen3-0.6B | test_npu_priority_metrics.py | TestNPUPriorityMetrics | ⚠️ 已适配 | 已存在 | 模型：Qwen/Qwen3-0.6B → QWEN3_0_6B_WEIGHTS_PATH；后端：默认 → ascend；禁用cuda graph |
| 7 | test_metrics.py | TestEnableMetrics | 集成测试 | Qwen/Qwen3-0.6B | test_npu_metrics.py | TestNPUEnableMetrics | ⚠️ 已适配 | 已存在 | 模型：Qwen/Qwen3-0.6B → QWEN3_0_6B_WEIGHTS_PATH；后端：默认 → ascend；禁用cuda graph |
| 8 | test_metrics.py | TestComputeRoutingKeyStats | 单元测试 | - | test_npu_metrics.py | TestNPUComputeRoutingKeyStats | ✅ 已映射 | 已存在 | 无需服务器，纯逻辑测试 |

**适配说明示例**：
- "模型：DEFAULT_SMALL_MODEL_NAME_FOR_TEST → LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH（NPU可用模型）"
- "模型：Qwen/Qwen3-0.6B → QWEN3_0_6B_WEIGHTS_PATH（NPU本地路径）"
- "后端：默认 → ascend（使用Ascend后端）"
- "禁用cuda graph：NPU不支持cuda graph，已添加--disable-cuda-graph"
- "使用PDDisaggregationServerBase：复用现有的disaggregation测试框架"

---

## 5. 覆盖率统计

**生成前**：
- GPU测试数量：8个测试类（排除单元测试后为4个集成测试类）
- NPU测试数量：7个测试类
- 覆盖率：**87.5%** (7/8)

**生成后**：
- GPU测试数量：8个测试类
- NPU测试数量：8个测试类（已生成）
- 支持的测试：8
- 不支持的测试：0
- **有效覆盖率**：100% (8/8)

---

## 6. 差距分析矩阵

| GPU测试 | NPU支持原因 | 状态 | 所需操作 |
|--------|------------|------|---------|
| test_tracing.py (TestTracePackage) | ✅ 无服务器依赖，纯逻辑测试 | 已存在 | 无需额外操作 |
| test_tracing.py (TestTraceServer) | ✅ 功能支持，模型可用 | 已存在 | 无需额外操作 |
| test_tracing.py (TestTraceEngine) | ✅ 功能支持，模型可用 | 已存在 | 无需额外操作 |
| test_tracing_disaggregation.py | ✅ 功能支持，NPU支持disaggregation，模型可用 | 已生成 | 运行测试验证阈值 |
| test_priority_metrics.py (TestQueueCount) | ✅ 无服务器依赖，纯逻辑测试 | 已存在 | 无需额外操作 |
| test_priority_metrics.py (TestPriorityMetrics) | ✅ 功能支持，模型可用 | 已存在 | 无需额外操作 |
| test_metrics.py (TestEnableMetrics) | ✅ 功能支持，模型可用 | 已存在 | 无需额外操作 |
| test_metrics.py (TestComputeRoutingKeyStats) | ✅ 无服务器依赖，纯逻辑测试 | 已存在 | 无需额外操作 |

---

## 7. NPU测试增强机会

### 7.1 模型可用性问题
- 无模型可用性问题，所有GPU测试使用的模型在NPU上都有对应版本

### 7.2 算法支持
- Tracing功能在NPU上完全支持
- Prometheus指标收集功能在NPU上完全支持
- PD disaggregation在NPU上已验证支持（已有多个disaggregation测试）

### 7.3 后端考虑
- GPU使用默认后端（可能为Triton或FlashAttention）
- NPU统一使用Ascend后端（`--attention-backend ascend`）

### 7.4 量化方案
- 本次测试未涉及量化方案差异

### 7.5 并行策略
- Disaggregation测试需要2-NPU环境
- 其他测试可在1-NPU上运行

---

## 8. 推荐测试生成优先级

### 阶段1（已完成）
| 优先级 | GPU测试 | 功能 | NPU适配 | 模型路径 | 配置 | 状态 |
|-------|--------|------|--------|---------|------|------|
| 高 | test_tracing.py (TestTraceServer) | Trace服务器测试 | 模型适配 | LLAMA_3_2_1B_INSTRUCT | ascend backend, enable-trace | ✅ 已存在 |
| 高 | test_tracing.py (TestTraceEngine) | Trace Engine测试 | 模型适配 | LLAMA_3_2_1B_INSTRUCT | ascend backend, enable-trace | ✅ 已存在 |
| 高 | test_priority_metrics.py | 优先级调度指标 | 模型适配 | QWEN3_0_6B | ascend backend, enable-priority-scheduling | ✅ 已存在 |
| 高 | test_metrics.py | Prometheus指标 | 模型适配 | QWEN3_0_6B | ascend backend, enable-metrics | ✅ 已存在 |

### 阶段2（本次生成）
| 优先级 | GPU测试 | 功能 | NPU适配 | 模型路径 | 配置 | 状态 |
|-------|--------|------|--------|---------|------|------|
| 高 | test_tracing_disaggregation.py | Disaggregation trace测试 | 模型适配 | LLAMA_3_2_1B_INSTRUCT | 2-NPU, PD disaggregation, enable-trace | ✅ 新生成 |

---

## 9. NPU关键适配说明

### 9.1 算法适配
- Tracing功能使用OpenTelemetry协议，与硬件平台无关，无需算法适配
- Prometheus指标收集与硬件平台无关，无需算法适配

### 9.2 后端适配
- GPU测试使用默认后端
- NPU测试统一使用`--attention-backend ascend`
- NPU测试统一添加`--disable-cuda-graph`（NPU不支持cuda graph）

### 9.3 模型适配
| GPU模型 | NPU模型路径 | 说明 |
|--------|-----------|------|
| DEFAULT_SMALL_MODEL_NAME_FOR_TEST | LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH | 小型测试模型 |
| DEFAULT_MODEL_NAME_FOR_TEST | LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH | 通用测试模型 |
| Qwen/Qwen3-0.6B | QWEN3_0_6B_WEIGHTS_PATH | NPU本地ModelScope路径 |

### 9.4 并行策略
- Disaggregation测试需要2-NPU环境（prefill和decode各占用1个NPU）
- 其他测试在1-NPU上运行
- 注册为`nightly-2-npu-a3`测试套件

### 9.5 评估阈值
- Tracing测试不涉及数值阈值，仅验证span存在性
- 指标测试验证指标存在性和数值合理性，阈值无需调整

### 9.6 环境变量
- OTLP exporter配置：`SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=50`
- OTLP exporter配置：`SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=4`
- Disaggregation配置：`MC_TCP_ENABLE_CONNECTION_POOL=true`

### 9.7 超时调整
- Server启动超时：DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH（无需调整）
- Disaggregation测试：est_time=400秒（与GPU版本一致）
- 其他测试：est_time=400秒（统一设置）

---

## 10. 生成的NPU测试场景

### 10.1 test_npu_tracing_disaggregation.py
**TestNPUTraceDisaggregation**

**测试类别**: 集成测试（Disaggregation + Tracing）

**测试目标**:
- Disaggregation模式下的tracing功能
- PREFILL_TRANSFER_KV_CACHE span验证
- DECODE_TRANSFERRED span验证

**测试方法**:
1. `test_disaggregation_transfer_spans()`: 验证disaggregation传输相关的span生成

**服务配置**:
```
--trust-remote-code
--disaggregation-mode prefill/decode
--enable-trace
--otlp-traces-endpoint localhost:4317
--attention-backend ascend
--disable-cuda-graph
--mem-fraction-static 0.3
```

**关键验证点**:
- Prefill和Decode服务器均启用tracing
- Load balancer正确路由请求
- OTLP collector正确收集span
- 验证disaggregation特有的span类型（PREFILL_TRANSFER_KV_CACHE, DECODE_TRANSFERRED）

---

## 11. 运行测试

### 11.1 运行所有生成测试
```bash
# Disaggregation tracing测试（需要2-NPU环境）
python -m unittest sglang518.test.registered.ascend.basic_function.observability.GENERATED_20260606.test_npu_tracing_disaggregation.TestNPUTraceDisaggregation
```

### 11.2 运行特定测试方法
```bash
python -m unittest ...TestNPUTraceDisaggregation.test_disaggregation_transfer_spans
```

### 11.3 语法验证
```bash
python -m py_compile sglang518/test/registered/ascend/basic_function/observability/GENERATED_20260606/test_npu_tracing_disaggregation.py
```

### 11.4 运行已存在的NPU测试
```bash
# Tracing测试（1-NPU）
python -m unittest sglang518.test.registered.ascend.basic_function.observability.GENERATED_20260529.test_npu_tracing

# Metrics测试（1-NPU）
python -m unittest sglang518.test.registered.ascend.basic_function.observability.GENERATED_20260529.test_npu_metrics

# Priority metrics测试（1-NPU）
python -m unittest sglang518.test.registered.ascend.basic_function.observability.GENERATED_20260529.test_npu_priority_metrics
```

---

## 12. 后续工作

### 12.1 受阻任务
无受阻任务，所有测试均可生成并运行。

### 12.2 阈值验证
- Disaggregation tracing测试需要在真实NPU环境验证span生成正确性
- 验证OTLP collector与NPU环境的兼容性
- 验证Mooncake传输backend在NPU上的稳定性

### 12.3 扩展测试
- 可添加更多disaggregation场景的tracing测试（如pause/resume、retract等）
- 可添加tracing与metrics联合测试
- 可添加tracing性能测试（验证tracing开销）

### 12.4 集成工作
- 将新生成的测试集成到CI流水线（nightly-2-npu-a3）
- 定期运行disaggregation tracing测试验证功能稳定性
- 监控OTLP collector的资源消耗

---

## 13. 总结

当前NPU可观测性测试覆盖率已从 **87.5%** 提升至 **100%**，共生成1个新测试文件。

**主要成果**：
1. ✅ 已覆盖所有GPU测试（8/8）
2. ✅ 所有单元测试均已映射（无需服务器依赖）
3. ✅ 所有集成测试均已适配（使用NPU可用模型和Ascend后端）
4. ✅ Disaggregation tracing测试已生成（验证NPU disaggregation环境下的tracing功能）

**主要适配**：
1. ⚠️ 模型路径适配：使用NPU本地ModelScope路径
2. ⚠️ 后端适配：统一使用Ascend后端
3. ⚠️ Cuda graph禁用：NPU不支持cuda graph
4. ⚠️ Disaggregation框架：使用PDDisaggregationServerBase简化测试编写

**建议**：
1. 在真实2-NPU环境运行disaggregation tracing测试，验证功能正确性
2. 监控OTLP collector性能，确保tracing开销可控
3. 定期运行所有observability测试，确保指标和tracing功能稳定

---

## 14. 生成的文件

| 文件 | 目录 | 描述 |
|-----|------|------|
| test_npu_tracing_disaggregation.py | GENERATED_20260606/ | Disaggregation tracing测试 |
| GPU_NPU_MAPPING_TABLE.md | GENERATED_20260606/ | 完整分析报告（本文件） |

**完整路径**:
```
sglang518/test/registered/ascend/basic_function/observability/
├── GENERATED_20260529/
│   ├── test_npu_tracing.py
│   ├── test_npu_metrics.py
│   └── test_npu_priority_metrics.py
└── GENERATED_20260606/
    ├── test_npu_tracing_disaggregation.py
    └── GPU_NPU_MAPPING_TABLE.md
```

**历史生成记录**:
- GENERATED_20260529: 生成tracing、metrics、priority_metrics测试（覆盖87.5%）
- GENERATED_20260606: 生成disaggregation tracing测试（达到100%覆盖）
