---
name: npu-param-to-test
description: >
  End-to-end NPU test automation pipeline. Phase 0: sync NPU parameter baseline from
  ascend_npu_support_features.mdx. Phase 1: cross-graph analysis using business code
  knowledge graph (sglang) + test code knowledge graph (ascend_sglang). Phase 2: test
  case design with npu-test-design principles. Phase 3: executable script generation.
  Designed to work with /test-gen for script generation and /graphify for graph queries.
trigger: /npu-test
---

# /npu-test — NPU 参数 → 测试自动化 全流程管线

端到端：NPU 参数基线 → 双图谱分析 → 用例设计 → 自动化脚本。
核心原则：**先复用社区、再查图谱理解、经由人工确认、最后输出脚本**。

## Usage

```
/npu-test <param_name>                    # 单参数全流程
/npu-test <param_name> --step 0|1|2|3    # 单步执行
/npu-test --baseline                      # 仅刷新参数基线列表
/npu-test --batch                         # 批量处理所有未覆盖的 A2,A3 参数
```

## 知识图谱

| 图谱 | 路径 | 用途 |
|---|---|---|
| 业务代码图谱 | `D:\00_code\claude\sglang\graphify-out` | 理解参数所属特性的上下文、调用链、依赖关系 |
| 测试代码图谱 | `D:\00_code\claude\ascend_sglang\graphify-out` | 查找最接近的现有测试、复用模式、避免重复 |

## 搜索范围

| 搜索内容 | 仓库 | 路径 |
|---|---|---|
| NPU 测试用例 | `D:\00_code\claude\ascend_sglang` | `test/registered/ascend/` |
| 业务代码 & 社区用例 | `D:\00_code\claude\sglang` | `python/sglang/` + `test/registered/` |

## 配套 Skill

| Skill | 作用 | 本 Skill 如何调用 |
|---|---|---|
| `npu-test-design` | 测试设计原则（测什么） | Step 2 逐项过 7 条原则 |
| `test-generator` (`/test-gen`) | 三阶段闭环测试生成 | Step 3 委托脚本生成 |
| `write-sglang-test` | CI/UT 测试写法规范 | Step 3 格式校验 |

---

## Step 0 — 获取 NPU 参数基线

### 0a. 确认参数来源

参数基线文档：

```
https://github.com/sgl-project/sglang/blob/main/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx
```

**Server supported** 列标注为 A2, A3 或 A2,A3 的参数即为 NPU 已适配项，均需用例覆盖。

### 0b. 提取参数列表

1. 读取 `ascend_npu_support_features.mdx`，解析表格
2. 筛选 `Server supported` 列含 A2 或 A3 的行
3. 输出参数清单，标注每项的 Server supported 值和特性分类

### 0c. 复用检查（优先级最高）

**在分析和设计之前**，分两路搜索：

**搜索 1 — NPU 已有测试**（在 `D:\00_code\claude\ascend_sglang`）：
```bash
graphify query "test_npu_<param_name>" --graph D:\00_code\claude\ascend_sglang\graphify-out\graph.json
glob test/registered/ascend/**/test_npu_*<param_name>*.py
```

**搜索 2 — 社区 GPU 测试**（在 `D:\00_code\claude\sglang`）：
```bash
graphify query "test for <param_name>" --graph D:\00_code\claude\sglang\graphify-out\graph.json
grep -rn "<param_name>" test/registered/ --include="*.py" -l
```

发现已有 NPU 用例 → 跳过。
发现 GPU 社区用例 → 评估移植可行性。

---

## Step 1 — 双图谱分析（理解参数）

### 1a. 业务代码图谱分析

在 `D:\00_code\claude\sglang\graphify-out` 上执行：

```bash
graphify query "what feature does <param_name> belong to and what are its dependencies"
graphify explain "<param_name>"
graphify query "what other parameters are related to <param_name>"
```

输出：参数的**业务上下文** — 属于哪个特性、调用链是什么、与哪些参数联动。

### 1b. 测试代码图谱分析

在 `D:\00_code\claude\ascend_sglang\graphify-out` 上执行：

```bash
graphify query "test for <feature_name> or similar to <param_name>"
graphify path "<param_name>" "test_npu_<closest_feature>"
graphify explain "test_npu_<closest_feature>"
```

### 1c. 通俗理解（必须输出）

对每个参数，用通俗语言解释：
- **这个参数是干什么的**（一句话能让人听懂）
- **不同取值之间有什么区别**（对用户/推理有什么实际影响）
- **什么场景下用户会需要改这个参数**

### 1d. 人工确认

汇总结果，向测试人员确认后方可进入 Step 2。

---

## Step 2 — 测试用例设计

### 设计原则

1. **所有可配置 value 值必须覆盖** — 默认值 + 每个显式可选值都有对应测试点
2. **端到端视角** — 站在用户角度，验证"启动 server → 推理 → 结果正确"，不是内部状态校验
3. **参数可组合** — 同特性参数可以放在一个用例文件中，组合测试
4. **移植 GPU 用例时**：
   - GPU 中 `@unittest.skip` 的用例在 NPU 中**不跳过**，全部执行
   - 用例名称尽量保留原名，添加 `_npu` 或 `test_npu_` 前缀
   - 必须说明 GPU 原用例测试了什么
5. **移植 + 新增合并** — 同一参数既有移植又有新增测试点，放入同一文件

### 输出：用例设计文档

包含以下章节：

```
1. 特性大类业务理解
2. 逐参数分析
   2.1 参数业务理解（通俗语言）
   2.2 GPU 社区用例分析（测试了什么，哪些可移植）
   2.3 测试点设计（标注来源：移植 / 新增）
3. 最终结论（每个参数：移植 vs 新增，文件清单）
```

### 检查清单

```
[ ] 每个参数的所有可选值都有测试覆盖
[ ] 理解了 GPU 用例测试了什么
[ ] 移植策略明确（每个测试点标注来源）
[ ] 端到端测试优先于单元测试
[ ] 同特性参数考虑组合测试
```

---

## Step 3 — 生成测试脚本

### 文件组织

```
test/registered/ascend/basic_function/
└── <feature_name>/
    └── test_npu_<feature>.py    # 同特性参数合并到一个文件
```

### 格式参考

读取 `test/registered/ascend/basic_function/` 下现有用例文件，复用 import / base class / server fixture / CI 注册。

### CI 注册规范

```python
@register_npu_ci(est_time=400, suite="full-<N>-npu-a3", nightly=True)
```

`<N>` 计算规则：

| 组网方式 | N 值 |
|---|---|
| 单机 / 双机混部 | `tp_size × pp_size`（pp 默认 1） |
| PD 分离 | `P(tp×pp) + D(tp×pp)` |
| 同文件多测试点 | 取 **最大值** |

### 移植 GPU 用例规范

- GPU 中 `@unittest.skip*` 装饰器 → NPU 中移除
- GPU 测试方法名 → NPU 中保留原名或加 `_npu` 后缀
- `register_cuda_ci` / `register_cpu_ci` → `register_npu_ci`
- 模型路径 → `test_ascend_utils` 中的 NPU 模型
- server args → 添加 `--attention-backend ascend`

### 完成后自检

```
[ ] python -m py_compile <output> 通过
[ ] 文件位于 test/registered/ascend/basic_function/<feature>/
[ ] CI 注册使用 register_npu_ci，suite 的 N 值正确
[ ] 每个参数的所有可选值都覆盖了
[ ] GPU 移植的 skip 已移除
[ ] 交由测试人员最终审核
```
