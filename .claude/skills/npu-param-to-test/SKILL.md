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

### 1c. 参数分析输出（业务理解 + 通俗理解，两者缺一不可）

每个参数的分析必须包含两部分：**业务理解**（技术细节）和 **通俗理解**（用户视角）。

> 模板示例如下：

```
<param_name>

  业务理解:
    定义: <文件>:<行号>, <一句话描述>
    取值: <默认值> | <可选值列表>
    作用链路: <调用链路径>
      → <关键函数> → <行为1>
      → <关键函数> → <行为2>
    依赖:
      - 下游: <受影响的模块>
      - 关联参数: <有联动关系的参数>
      - 前置条件: <使用该参数需满足的条件>

  通俗理解:
    <用比喻或生活场景解释这个参数是干什么的>
    - 取值 "<A>": <一句话说清楚这个值的含义和影响>
    - 取值 "<B>": <一句话说清楚这个值的含义和影响>
    - 用户什么时候需要改这个参数: <具体场景>
```

> 真实案例 — `--tokenizer-backend`：

```
--tokenizer-backend

  业务理解:
    定义: server_args.py:434-442, 控制 tokenizer 底层库选型
    取值: "huggingface" (默认) | "fastokens"
    作用链路: tokenizer.py:459-476
      server_args.tokenizer_backend
        → get_tokenizer(tokenizer_backend=...)
          → "huggingface" → HuggingFace 原生 tokenizers 库
          → "fastokens"   → _ensure_fastokens_patched() 猴子补丁 transformers
        → 同时传给 get_processor() 影响多模态 tokenizer 加载
    依赖:
      - 下游: TokenizerManager、DetokenizerManager 的 tokenizer 初始化
      - 关联参数: --tokenizer-mode (fast/slow 控制模式，与 backend 正交)
      - 前置条件: fastokens 需 pip install fastokens

  通俗理解:
    Tokenizer 就像翻译官，把人类语言翻译成模型能理解的数字。
    --tokenizer-backend 选择"哪个翻译官"来干活。
    - huggingface: 官方翻译官，最稳定、所有模型都能用
    - fastokens: 快手翻译官，高并发时更快，需额外安装 fastokens 包
    - 用户什么时候需要改: 高并发推理时切换 fastokens 降低 tokenization 延迟
```

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
6. **不做冗余测试** — 满足以下任一条件不测：
   - 同一调用链已被其他参数用例覆盖
   - 参数默认值已被同类用例隐式验证（所有不指定该参数的用例都会走默认路径）
   - 性能差异属于量变而非质变
   - 修改参数值不会触发任何不同的代码分支

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
[ ] 没有冗余测试（透传/不触发新分支/纯性能差异）
```

### 设计完成后自检

设计完成后，逐参数过一遍：

```
[ ] 每个参数的依赖链路是否完整梳理？
[ ] 关联参数是否需要一起测试？（如 detokenizer_worker_num 配合 tokenizer_worker_num）
[ ] 是否检查了现有 NPU 用例避免重复？（graphify 查 + glob 搜）
[ ] 是否检查了 GPU 社区用例可移植？（grep 搜 sglang 仓库）
[ ] 多模态是否需要单独测试？（仅在参数触发不同多模态代码分支时才需要）
[ ] 并发是否需要单独测试？（仅在其他用例未覆盖并发路径时才需要）
[ ] 每个测试点来源标注是否正确？（移植/新增/复用）
```

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

---

## Step 4 — 推送前 Lint 检查

**每次推送代码前**，必须本地运行 pre-commit，避免 CI lint 失败：

```bash
pip3 install pre-commit
pre-commit install          # 安装 git hook，之后每次 git commit 自动触发
pre-commit run --all-files  # 手动全量运行，自动修复可修复的 lint 问题
```

> 如果第一次运行失败，**再运行一次**确保 lint 全部通过。部分工具需要第二次才能应用修复。

**禁止直接提交到 main 分支**，始终从新分支提 PR。
CI 中的链接检查 (lychee) 默认不阻塞本地 commit，如需手动检查：
```bash
pre-commit run --hook-stage manual lychee --all-files
```

---

## Step 5 — CI 调试

测试脚本写完后，需要在流水线上验证。调试流程如下：

### 5a. 注册独立 debug suite

将新增用例注册到**独立调试 suite**，避免干扰现有用例：

```python
# 调试阶段
register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)

# 调试通过后改为正式 suite
register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
```

### 5b. 修改 `.github/workflows/full-test-npu.yml`

参考 PR [#801](https://github.com/Ascend/sglang/pull/801)：

**启用 PR trigger**：
```yaml
on:
  pull_request:
    branches: [main, testcases]
    paths:
      - ".github/workflows/full-test-npu.yml"
```

**更新默认配置**：
```yaml
skip_install_flag: required: true, default: true   # 调试阶段跳过安装
test_scope: default: all
image_a2/a3: 更新为最新镜像日期
```

**注释掉非调试 suite**（`#` 前缀保留原代码，不删除）：
```yaml
#  nighly-test-npu:
#  full-1-npu-a3:
#  full-2-npu-a3:
#  ...
```

**新增 debug job**（完整复制 `full-1-npu-a3` 的 steps，只改两处）：
```yaml
  debug-full-1-npu-a3:
    needs: [ set-image-config ]
    runs-on: linux-aarch64-a3-2
    # ... 完整复制原 job 的所有 steps ...
    # 差异1: 额外安装 fastokens（如测试需要）
    # 差异2: suite 名改为 debug-full-1-npu-a3
    - name: Run debug test
      timeout-minutes: 360
      run: |
        ...
        cd test
        python3 run_suite.py --hw npu --suite debug-full-1-npu-a3 --nightly --continue-on-error --timeout-per-file 3600
```

**更新 `check-all-jobs`**：
```yaml
  check-all-jobs:
    needs:
      - set-image-config
      - debug-full-1-npu-a3
#      - nighly-test-npu          # 注释保留，调试完恢复
#      - full-1-npu-a3
#      ...
```

### 5c. 修改 `scripts/ci/npu/npu_ci_install_dependency.sh`

```diff
- && (cd ... && ln -s deep_ep/deep_ep_cpp*.so)
+ && (cd ... && ln -sf deep_ep/deep_ep_cpp*.so)
```

### 5d. 常见坑

| 问题 | 原因 | 修复 |
|---|---|---|
| pytest `--timeout 3600` 报错 | pytest 不识别空格分隔的 timeout | 用 `run_suite.py --timeout-per-file` 替代 |
| job 步骤行缺失 | 手写 job 漏了原 job 的注释/echo/空格行 | 复制原 job 后逐行对比补回 |
| 用例跑不到 | suite 名不匹配 | 测试文件 register 和 run_suite.py `--suite` 保持一致 |
| 被 CI 调度器 kill | 更高优先级任务抢占 runner | 正常，重跑即可 |

### 5e. 调试完成后恢复

```bash
# 1. full-test-npu.yml 取消注释，恢复完整 CI
# 2. 测试文件 suite 改回正式值
register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
# 3. submit clean PR
```

### 代码修改清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `test/.../test_npu_<param>.py` | 新建 | suite=`debug-full-1-npu-a3` |
| `.github/workflows/full-test-npu.yml` | 修改 | PR trigger、镜像、注释旧 job、新增 debug job |
| `scripts/ci/npu/npu_ci_install_dependency.sh` | 修改 | `ln -s` → `ln -sf` |
| Skill / 设计文档 | 暂存本地 | 不提交到 PR |
