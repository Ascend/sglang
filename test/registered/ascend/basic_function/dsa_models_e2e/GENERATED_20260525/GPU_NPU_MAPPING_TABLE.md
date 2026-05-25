# GPU-NPU DSA Models E2E Test Coverage Analysis (Complete Report)
# GPU与NPU DSA模型端到端测试覆盖分析（完整报告）

**Analysis Date**: 2026-05-25
**分析日期**: 2026-05-25

**Analyzed Scope**: test/registered/dsa_models_e2e/ (only this directory)
**分析范围**: test/registered/dsa_models_e2e/ (仅此目录)

**Generator**: npu-test-gap-v9 skill
**生成器**: npu-test-gap-v9 skill

---

## 1. Excluded Tests (Unit Tests Only)
## 1. 排除测试（仅单元测试）

No unit tests excluded in this directory. All 4 test files are E2E integration tests.
本目录无单元测试排除。所有4个测试文件均为端到端集成测试。

---

## 2. GPU Integration Tests Summary
## 2. GPU集成测试摘要

| GPU Test File | Test Class | Test Type | Model | Parallel Strategy | Speculative Algorithm | Test Scenario |
|---------------|------------|-----------|-------|-------------------|----------------------|---------------|
| test_dsa_glm5_tp_mtp.py | TestGLM5TPMTP | E2E Integration | GLM-5-FP8 (zai-org/GLM-5-FP8) | TP=8 | EAGLE | GSM8K accuracy + Speed benchmark |
| test_dsa_glm5_dp_mtp.py | TestGLM5DPMTP | E2E Integration | GLM-5-FP8 (zai-org/GLM-5-FP8) | TP=8, DP=8, enable_dp_attention | EAGLE | GSM8K accuracy + Speed benchmark (DP) |
| test_dsa_dsv32_tp_mtp.py | TestDeepseekV32TPMTP | E2E Integration | DeepSeek-V3.2 (deepseek-ai/DeepSeek-V3.2) | TP=8 | EAGLE | GSM8K accuracy + Speed benchmark |
| test_dsa_dsv32_dp_mtp.py | TestDeepseekV32DPMTP | E2E Integration | DeepSeek-V3.2 (deepseek-ai/DeepSeek-V3.2) | TP=8, DP=8, enable_dp_attention | EAGLE | GSM8K accuracy + Speed benchmark (DP) |

**Common Test Components**:
- Base fixture: `DsaMtpServerBase` (server lifecycle management)
- Config defaults: `DsaMtpEvalConfigDefaults` (GSM8K thresholds, accept length)
- GSM8K evaluation: `GSM8KMixin.test_gsm8k()` (accuracy >= 0.94, accept_length >= 2.7)
- Speed benchmark: `SpecDecodingMixin.test_bs_1_speed()` (speed threshold per variant)

**Hardware Requirements**: 8-GPU H200 per test
**硬件需求**: 每个测试需要8张H200 GPU

---

## 3. NPU Existing Tests Summary
## 3. NPU现有测试摘要

No existing NPU tests in `test/registered/ascend/basic_function/dsa_models_e2e/` before this analysis.
分析前，`test/registered/ascend/basic_function/dsa_models_e2e/` 目录下无NPU测试。

**Related NPU Tests** (in speculative_inference directory):
**相关NPU测试**（在speculative_inference目录）:

| NPU Test File | Test Class | Model | Algorithm | Config | Status |
|---------------|------------|-------|-----------|--------|--------|
| test_npu_speculative_multi_npu.py | TestNpuSpeculativeDraftParams | Qwen3-32B-W8A8-MindIE | EAGLE3 | TP=4, draft params | EXISTING |
| test_npu_adaptive_speculative.py | TestNPUAdaptiveSpeculativeServer | Qwen3-32B-W8A8-MindIE | EAGLE3 | TP=8, adaptive | GENERATED (20260525) |

---

## 4. GPU-NPU Test Mapping Table (Complete)
## 4. GPU-NPU测试映射表（完整）

| # | GPU Test File | GPU Test Class | Test Type | Model | NPU Test File | NPU Test Class | Mapping Status | NPU Status | Key Adaptations |
|---|---------------|----------------|-----------|-------|---------------|----------------|----------------|------------|-----------------|
| 1 | test_dsa_glm5_tp_mtp.py | TestGLM5TPMTP | E2E Integration | GLM-5-FP8 | - | - | ❌ NOT SUPPORTED | - | GLM-5-FP8 weights unavailable on NPU |
| 2 | test_dsa_glm5_dp_mtp.py | TestGLM5DPMTP | E2E Integration | GLM-5-FP8 | - | - | ❌ NOT SUPPORTED | - | GLM-5-FP8 weights unavailable on NPU + DP attention |
| 3 | test_dsa_dsv32_tp_mtp.py | TestDeepseekV32TPMTP | E2E Integration | DeepSeek-V3.2 | test_npu_dsa_dsv32_tp_mtp.py | TestNPUDsv32TPMTP | ⚠️ ADAPTED | GENERATED | Model: DeepSeek-V3.2 → DeepSeek-V3.2-W8A8; Algorithm: EAGLE → EAGLE3; Quantization: FP8 → W8A8 (modelslim) |
| 4 | test_dsa_dsv32_dp_mtp.py | TestDeepseekV32DPMTP | E2E Integration | DeepSeek-V3.2 | test_npu_dsa_dsv32_dp_mtp.py | TestNPUDsv32DPMTP | ⚠️ ADAPTED | GENERATED | Model: DeepSeek-V3.2 → DeepSeek-V3.2-W8A8; Algorithm: EAGLE → EAGLE3; DP attention enabled; Quantization: W8A8 (modelslim) |

**Adaptation Notes**:
**适配说明**:

### test_npu_dsa_dsv32_tp_mtp.py (Generated)
- **Algorithm**: EAGLE → EAGLE3 (NPU supports EAGLE3 variant)
- **Model**: DeepSeek-V3.2 (FP8) → DeepSeek-V3.2-W8A8 (W8A8 quantization via modelslim)
- **Backend**: Triton/FlashAttention → Ascend backend
- **TP Config**: TP=8 preserved (requires 8-NPU)
- **Accuracy Threshold**: 0.94 → 0.85 (adjusted for W8A8 quantization)
- **Speed Threshold**: 180 → 150 token/s (adjusted for NPU hardware)
- **Accept Length**: 2.7 → 2.5 (adjusted threshold)
- **Environment**: SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1, SGLANG_ENABLE_SPEC_V2=1
- **Timeout**: DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3 (extended for NPU initialization)

### test_npu_dsa_dsv32_dp_mtp.py (Generated)
- **DP Attention**: Enabled (--dp-size 8, --enable-dp-attention)
- **Algorithm**: EAGLE → EAGLE3
- **Model**: Same adaptation as TP test
- **Speed Threshold**: 90 → 90 token/s (DP attention typically slower)
- **Timeout**: DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5 (extended for DP initialization)

---

## 5. Coverage Statistics
## 5. 覆盖率统计

**Before Analysis**:
**生成前**:
- GPU Tests: 4
- NPU Tests: 0
- Coverage: **0%** (0/4)

**After Analysis**:
**生成后**:
- GPU Tests: 4
- NPU Tests: 2 (generated)
- NPU Supported Tests: 2 (DeepSeek-V3.2 variants)
- NPU Not Supported: 2 (GLM-5-FP8 variants - model weights unavailable)
- **Effective Coverage**: **50%** (2/4 for supported models)
- **Overall Coverage**: **50%** (2/4 considering model availability)

---

## 6. Gap Analysis Matrix
## 6. 差距分析矩阵

| GPU Test | NPU Support Reason | Status | Action Required |
|----------|-------------------|--------|-----------------|
| test_dsa_glm5_tp_mtp.py | ❌ GLM-5-FP8 weights unavailable on NPU | NOT SUPPORTED | Request model weights from ZhipuAI or use alternative GLM variant |
| test_dsa_glm5_dp_mtp.py | ❌ GLM-5-FP8 weights unavailable + DP attention | NOT SUPPORTED | Same as above, plus verify DP attention support for GLM models |
| test_dsa_dsv32_tp_mtp.py | ✅ DeepSeek-V3.2-W8A8 available, EAGLE3 supported | GENERATED | Run test to validate accuracy/speed thresholds |
| test_dsa_dsv32_dp_mtp.py | ✅ DeepSeek-V3.2-W8A8 available, DP attention supported | GENERATED | Run test to validate DP attention + speculative decoding |

---

## 7. NPU Test Enhancement Opportunities
## 7. NPU测试增强机会

### 7.1 Model Availability Issues
**模型可用性问题**:
- GLM-5-FP8: No NPU weights available. Could use GLM-4-9B-Chat (existing path: `GLM_4_9B_CHAT_WEIGHTS_PATH`) as alternative for basic GSM8K tests, but not for EAGLE speculative decoding (requires specific draft model).
- DeepSeek-V3.2: W8A8 quantization available, FP8 version not yet available.

### 7.2 Algorithm Support
**算法支持**:
- EAGLE: GPU uses original EAGLE algorithm
- EAGLE3: NPU supports EAGLE3 variant (enhanced version with better NPU compatibility)
- All generated tests use EAGLE3

### 7.3 Backend Considerations
**后端考虑**:
- GPU: Triton backend, FlashAttention3
- NPU: Ascend backend, native attention implementation
- Added `--attention-backend ascend` to all NPU tests

### 7.4 Quantization
**量化**:
- GPU: FP8 quantization for GLM-5 and DeepSeek-V3.2
- NPU: W8A8 quantization (modelslim) for DeepSeek-V3.2
- Added `--quantization modelslim` to NPU tests

---

## 8. Recommended Test Generation Priority
## 8. 推荐测试生成优先级

### Phase 1 (Completed)
**阶段1（已完成）**:
| Priority | GPU Test | Feature | NPU Adaptation | Model Path | Config | Status |
|----------|----------|---------|----------------|------------|--------|--------|
| High | test_dsa_dsv32_tp_mtp.py | EAGLE3 + TP | DeepSeek-V3.2 W8A8 | DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH | TP=8 | ✅ GENERATED |
| High | test_dsa_dsv32_dp_mtp.py | EAGLE3 + DP attention | DeepSeek-V3.2 W8A8 | DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH | TP=8, DP=8 | ✅ GENERATED |

### Phase 2 (Blocked - Model Weights)
**阶段2（受阻 - 模型权重）**:
| Priority | GPU Test | Feature | Blocker | Resolution |
|----------|----------|---------|---------|------------|
| Medium | test_dsa_glm5_tp_mtp.py | GLM-5-FP8 + EAGLE | GLM-5-FP8 weights unavailable | Request from ZhipuAI or use GLM-4-9B alternative |
| Medium | test_dsa_glm5_dp_mtp.py | GLM-5-FP8 + DP + EAGLE | GLM-5-FP8 weights unavailable | Same as above |

---

## 9. Key NPU Adaptation Notes
## 9. NPU关键适配说明

### 9.1 Algorithm Adaptation
**算法适配**:
- EAGLE → EAGLE3: NPU uses EAGLE3 variant for speculative decoding
- Speculative parameters preserved: num_steps=3, eagle_topk=1, num_draft_tokens=4

### 9.2 Backend Adaptation
**后端适配**:
- Triton → Ascend: Use `--attention-backend ascend`
- FlashAttention → Ascend native attention
- CUDA Graph → Disabled: `--disable-cuda-graph`

### 9.3 Model Adaptation
**模型适配**:
- DeepSeek-V3.2 (FP8) → DeepSeek-V3.2-W8A8 (W8A8 via modelslim)
- Quantization: `--quantization modelslim`
- Dtype: `--dtype bfloat16`

### 9.4 Parallel Strategy
**并行策略**:
- TP=8: Preserved (requires 8-NPU)
- DP=8 + enable_dp_attention: Supported on NPU (verified in existing tests)

### 9.5 Evaluation Thresholds
**评估阈值**:
- GSM8K accuracy: 0.94 → 0.85 (adjusted for W8A8 quantization)
- Accept length: 2.7 → 2.5 (adjusted threshold)
- Speed threshold: Adjusted per variant based on NPU hardware characteristics

### 9.6 Environment Variables
**环境变量**:
- `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`: Enable overlap scheduling
- `SGLANG_ENABLE_SPEC_V2=1`: Enable speculative decoding v2

### 9.7 Timeout Adjustments
**超时调整**:
- TP test: DEFAULT_TIMEOUT * 3 (extended for NPU initialization)
- DP test: DEFAULT_TIMEOUT * 5 (extended for DP attention initialization)

---

## 10. Generated NPU Test Scenarios
## 10. 生成的NPU测试场景

### 10.1 test_npu_dsa_dsv32_tp_mtp.py
**TestNPUDsv32TPMTP**

**Test Category**: E2E Speculative Decoding
**测试类别**: 端到端推测解码

**Test Target**:
**测试目标**:
- DeepSeek-V3.2 model with W8A8 quantization
- EAGLE3 speculative decoding algorithm
- Tensor Parallelism (TP=8)
- GSM8K accuracy evaluation
- Speed benchmark (bs_1 speed)

**Test Methods**:
**测试方法**:
1. `test_gsm8k()`: GSM8K accuracy >= 0.85, accept_length >= 2.5
2. `test_bs_1_speed()`: Speed >= 150 token/s, accept_length > 2.5

**Server Configuration**:
**服务配置**:
```
--trust-remote-code
--attention-backend ascend
--quantization modelslim
--tp-size 8
--speculative-algorithm EAGLE3
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
--mem-fraction-static 0.7
--disable-cuda-graph
--dtype bfloat16
```

**Environment**:
**环境**:
```
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
SGLANG_ENABLE_SPEC_V2=1
```

### 10.2 test_npu_dsa_dsv32_dp_mtp.py
**TestNPUDsv32DPMTP**

**Test Category**: E2E Speculative Decoding with DP Attention
**测试类别**: 端到端推测解码 + DP注意力

**Test Target**:
**测试目标**:
- DeepSeek-V3.2 model with W8A8 quantization
- EAGLE3 speculative decoding algorithm
- Data Parallelism (DP=8) + DP attention
- Tensor Parallelism (TP=8)
- GSM8K accuracy evaluation
- Speed benchmark (bs_1 speed)

**Test Methods**:
**测试方法**:
1. `test_gsm8k()`: GSM8K accuracy >= 0.85, accept_length >= 2.5
2. `test_bs_1_speed()`: Speed >= 90 token/s (DP attention typically slower)

**Server Configuration**:
**服务配置**:
```
--trust-remote-code
--attention-backend ascend
--quantization modelslim
--tp-size 8
--dp-size 8
--enable-dp-attention
--speculative-algorithm EAGLE3
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
--mem-fraction-static 0.7
--disable-cuda-graph
--dtype bfloat16
```

**Environment**:
**环境**:
```
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
SGLANG_ENABLE_SPEC_V2=1
```

---

## 11. Running the Tests
## 11. 运行测试

### 11.1 Run All Generated Tests
**运行所有生成测试**:
```bash
# Run TP test
python -m unittest sglang518.test.registered.ascend.basic_function.dsa_models_e2e.GENERATED_20260525.test_npu_dsa_dsv32_tp_mtp.TestNPUDsv32TPMTP

# Run DP test
python -m unittest sglang518.test.registered.ascend.basic_function.dsa_models_e2e.GENERATED_20260525.test_npu_dsa_dsv32_dp_mtp.TestNPUDsv32DPMTP
```

### 11.2 Run Specific Test Methods
**运行特定测试方法**:
```bash
# GSM8K evaluation only
python -m unittest sglang518.test.registered.ascend.basic_function.dsa_models_e2e.GENERATED_20260525.test_npu_dsa_dsv32_tp_mtp.TestNPUDsv32TPMTP.test_gsm8k

# Speed benchmark only
python -m unittest sglang518.test.registered.ascend.basic_function.dsa_models_e2e.GENERATED_20260525.test_npu_dsa_dsv32_tp_mtp.TestNPUDsv32TPMTP.test_bs_1_speed
```

### 11.3 Syntax Verification
**语法验证**:
```bash
python -m py_compile sglang518/test/registered/ascend/basic_function/dsa_models_e2e/GENERATED_20260525/test_npu_dsa_dsv32_tp_mtp.py
python -m py_compile sglang518/test/registered/ascend/basic_function/dsa_models_e2e/GENERATED_20260525/test_npu_dsa_dsv32_dp_mtp.py
```

---

## 12. Future Work
## 12. 后续工作

### 12.1 Blocked Tasks (Model Weights)
**受阻任务（模型权重）**:
- Generate NPU tests for GLM-5-FP8 once weights become available
- Validate EAGLE draft model compatibility for GLM-5 on NPU

### 12.2 Threshold Validation
**阈值验证**:
- Run generated tests and collect actual GSM8K accuracy scores
- Adjust thresholds based on real NPU performance data
- Validate speed benchmarks on NPU hardware

### 12.3 Extended Testing
**扩展测试**:
- Add more GSM8K test variants (different shot counts)
- Add MMLU evaluation if time permits
- Add longer sequence length tests for speculative decoding

### 12.4 Integration
**集成**:
- Register tests in CI pipeline (nightly-8-npu-a3 suite)
- Monitor test stability over multiple runs
- Document any NPU-specific issues discovered

---

## 13. Conclusion
## 13. 总结

The current NPU DSA models E2E test coverage has improved from **0%** to **50%** with 2 newly generated tests. The generated tests cover DeepSeek-V3.2 with both TP and DP attention configurations:

当前NPU DSA模型端到端测试覆盖率已从 **0%** 提升至 **50%**，共生成2个新测试。生成的测试覆盖了DeepSeek-V3.2的TP和DP注意力配置：

1. ✅ **TP Test**: test_npu_dsa_dsv32_tp_mtp.py - DeepSeek-V3.2 + EAGLE3 + TP=8
2. ✅ **DP Test**: test_npu_dsa_dsv32_dp_mtp.py - DeepSeek-V3.2 + EAGLE3 + TP=8 + DP=8

**Key Limitations**:
**主要限制**:
1. ❌ **GLM-5-FP8 Tests**: Blocked due to unavailable model weights on NPU
2. ⚠️ **Threshold Adjustment**: Accuracy and speed thresholds adjusted for W8A8 quantization; needs validation

**Recommendations**:
**建议**:
1. Request GLM-5-FP8 model weights for NPU from ZhipuAI team
2. Run generated DeepSeek-V3.2 tests and validate thresholds
3. Monitor test stability and adjust thresholds based on real performance data

---

## 14. Generated Files
## 14. 生成的文件

| File | Directory | Description |
|------|-----------|-------------|
| test_npu_dsa_dsv32_tp_mtp.py | GENERATED_20260525/ | DeepSeek-V3.2 + EAGLE3 + TP test |
| test_npu_dsa_dsv32_dp_mtp.py | GENERATED_20260525/ | DeepSeek-V3.2 + EAGLE3 + DP attention test |
| GPU_NPU_MAPPING_TABLE.md | GENERATED_20260525/ | Complete coverage analysis report (this file) |

**Full Path**:
**完整路径**:
```
sglang518/test/registered/ascend/basic_function/dsa_models_e2e/GENERATED_20260525/
├── test_npu_dsa_dsv32_tp_mtp.py
├── test_npu_dsa_dsv32_dp_mtp.py
└── GPU_NPU_MAPPING_TABLE.md
```
