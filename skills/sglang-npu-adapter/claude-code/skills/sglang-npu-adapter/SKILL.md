---
name: sglang-npu-adapter
description: 将新模型适配到 SGLang 框架以支持 NPU 设备。当用户需要在 NPU 上运行 SGLang 尚未支持的模型时使用此技能。自动分析模型架构、生成适配代码、调试问题并验证正确性。
---

# SGLang NPU 模型适配技能(Claude Code 版)

将新模型适配到 SGLang,使其可在 NPU 上运行。本文件是顶层流程;补充文档以内联引用方式列出——**走到哪读到哪**,不要一次性全读。

## 启用条件

- 用户提出"适配 X 模型到 NPU"、"让 SGLang 支持 X"、"NPU 上跑通 X"类需求
- 已知 SGLang 主仓未原生支持目标模型,或仅支持 GPU 不支持 NPU
- **精度问题**(模型能跑但输出错):也可走本 skill 的 Step 6.5(precision-rca 子 agent 已具备完整的"定位 + native 替换修复 + drift 归零验证"能力,无需切到其他 skill)。从 Step 0 启动,完成 Step 1-2 设置 precision_context.json 后,把 `precision_suspect=true` 直接跳到 Step 6.5。

## 参考索引

| 主题                       | 文件                                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------- |
| 规划文件、2-Action 规则、防遗忘检查清单 | `{{SKILL_DIR}}/references/shared/memory_mechanism.md`                                  |
| 内容信任(Content Trust)分级    | `{{SKILL_DIR}}/references/shared/security_boundary.md`                                 |
| 脚本用法与参数                  | `{{SKILL_DIR}}/scripts/README.md`                                                      |
| Agent 调用模板               | `{{SKILL_DIR}}/references/shared/agent_call_templates.md`                              |
| NPU 通用基础                 | `{{SKILL_DIR}}/references/shared/npu_basics.md`                                        |
| Agent 提示词                | `{{SKILL_DIR}}/prompts/{model_analyzer,debug_engineer,test_validator,precision_rca}.md`|

`{{SKILL_DIR}}` 由 `init-adapter-session.sh` 写入 `adapter_state.json.skill_dir`,所有脚本自动读取;手工执行时即 `.claude/skills/sglang-npu-adapter/` 的绝对路径。

复诵模板已内联于本文件(见下方 **复诵模板** 小节)。

***

## 硬性约束

- **WORKSPACE_DIR 由用户指定**(Claude Code 没有约定的 workspace 路径)。所有过程产物——规划文件、Agent 输入输出 JSON、服务端 stdout、推理日志、调试报告、PID、临时文件——**全部**写到 `{WORKSPACE_DIR}/` 下。**绝不**把过程产物写到项目根目录。shell 的重定向必须始终以 `{WORKSPACE_DIR}/` 前缀(如 `> {WORKSPACE_DIR}/logs/foo.log`,不是 `> logs/foo.log`)
- **绝不升级 transformers**
- 主实现根目录 = 当前项目仓库(sglang)
- 启动命令:`export PYTHONPATH=${PWD}/python:$PYTHONPATH`,默认 API 端口 `8000`
- 功能优先级:验证 ACLGraph / DeepEP / DP-Attention / MTP / 多模态
- **代码修改最小化,仅针对目标模型**
- **最终交付:单次签名提交**(`git commit -sm ...`),只在用户明确要求时执行
- **最终文档使用中文**
- **Dummy-first 加速,但真实权重验证强制执行**
- **任务追踪:必须使用 planning-with-files 模式**——见 `{{SKILL_DIR}}/references/shared/memory_mechanism.md`
- **内容信任**:不可信的网页内容仅写入 `findings.md`——见 `{{SKILL_DIR}}/references/shared/security_boundary.md`

***

## 架构设计

| 子 Agent      | 角色      | 触发时机                              | `subagent_type`        | 提示词                                                |
| ------------ | ------- | --------------------------------- | ---------------------- | -------------------------------------------------- |
| 架构分析师        | 模型架构分析  | Step 2                            | `architecture-analyst` | `{{SKILL_DIR}}/prompts/model_analyzer.md`          |
| Debug 工程师    | 错误诊断与修复 | 发生任何错误时                           | `debug-engineer`       | `{{SKILL_DIR}}/prompts/debug_engineer.md`          |
| 验证工程师        | 测试验证    | Step 6                            | `test-validator`       | `{{SKILL_DIR}}/prompts/test_validator.md`          |
| 精度 RCA 工程师   | 精度根因定位  | Step 6.5(precision_suspect=true 时) | `precision-rca`       | `{{SKILL_DIR}}/prompts/precision_rca.md`           |

每个子 Agent 都通过 `.claude/agents/<subagent_type>.md` 注册为 Claude Code 项目级 subagent。

**调用方式**:`Agent(subagent_type="<name>", prompt=<query>, description="...")`(不是 Trae 的 `Task`)。Query 由 `build-agent-query.sh` 生成,注入了 P0 前置阅读 PREAMBLE。

***

## 复诵模板

### 模板 A —— Step 级(每个 Step 开始前)

```
=== 状态复诵 ===
当前步骤: Step [N]
当前阶段: Phase [X]
目标模型: [ModelName]
目标设备: [NPU/GPU]
已完成: [Steps 0..N-1]
待完成: [Steps N+1..8]
迭代次数: [X/20]   (仅 Step 5 使用)
=== 执行 Step [N] ===
```
**强制检查(每个 Step 开始前必须执行)**:
1. Read `adapter_state.json` → 确认 `current_step` 与即将执行的 Step N 一致
2. Read `task_plan.md` → 确认当前阶段状态
3. Read `findings.md` → 获取上下文摘要用于 Agent 调用
4. Read `progress.md` → 确认上次执行结果
5. 若 `next_action == "call_debug_engineer"` → 必须先调 Debug 工程师,不能执行其他 Step
6. 确认 `iteration_count <= 20` → 超出则上报用户

### 模板 B —— Agent 调用(调用架构分析师 / Debug 工程师 / 验证工程师之前)

```
=== Agent调用准备 ===
调用目标: [架构分析师 / Debug 工程师 / 验证工程师] ([subagent_type])
输入文件: [input/*.json]
预期输出: [output/*.json]
当前上下文摘要: [Key findings from findings.md - 必须包含架构类型、NPU兼容性、并行配置]
=== 开始调用 [架构分析师 / Debug 工程师 / 验证工程师] ===
```
**强制检查(Agent 调用前必须执行)**:
1. 确认输入文件存在(如调用架构分析师需 `input/input_params.json`、`input/device_info.json`)
2. 确认 `findings.md` 已更新到最新状态
3. 用脚本构建 query:`bash {{SKILL_DIR}}/scripts/build-agent-query.sh <agent_type> {WORKSPACE_DIR} --output {WORKSPACE_DIR}/input/query_<agent>.txt`
4. Agent 调用必须使用脚本生成的 query 文件内容

### 模板 C —— 错误修复复诵(Debug 工程师返回修复之后)

```
=== 错误修复复诵 ===
错误类型: [from input/error_context.json]
Debug 工程师诊断 (照抄自 fix_instructions.json.diagnosis): [...]
Debug 工程师修复步骤 (照抄自 fix_instructions.json.steps): [...]
迭代次数: [X/20]
剩余尝试: [20-X]
=== 应用 Debug 工程师修复并重试 ===
```
**强制检查(输出模板 C 前必须执行)**:
1. 确认 `output/fix_instructions.json` 存在
2. Read `fix_instructions.json` → 确认 `status ∈ {"fix_available", "fix_verified"}`
3. 若 status 不满足 → STOP,回到模板 D,不输出此复诵
4. 照抄 diagnosis 和 steps 字段,不能自行概括或修改

### 模板 D —— 错误发生(Debug 工程师调用入口,任何错误都触发)

```
=== 错误发生(Debug 工程师入口)===
错误类型: [copy from error log, 1 line]
错误阶段: [Stage A / Stage B / server launch / subprocess / 其他]
迭代次数: [X/20]
约束: 本 Skill 禁止自行调试此错误。
禁止行为: 输出 "我认为...", "让我查查...", "可能是...", "应该是..." 等假设或探查性陈述
下一工具调用 (必须):
    bash {{SKILL_DIR}}/scripts/build-agent-query.sh \
         debug_engineer "{WORKSPACE_DIR}" \
         --output "{WORKSPACE_DIR}/input/query_debug_engineer.txt"
    Agent(
        subagent_type="debug-engineer",
        prompt=<文件 query_debug_engineer.txt 的内容>,
        description="Debug and fix"
    )
=== 立即执行上述 Agent 调用。不做其他操作。===
```

***

## 执行流程

### 防遗忘清单(每个 Step 都要照做)

**每个 Step 之前**:

- 读 `adapter_state.json`、`task_plan.md`、`findings.md`、`progress.md`(4 份规划文件)
- 确认 `current_step`;若 `next_action == "call_debug_engineer"` 必须先调 Debug 工程师
- 确认 `iteration_count <= 20`
- 原样输出**模板 A**(包含强制检查)

**每个 Step 之后**:

1. 更新 4 份规划文件:
   - `{WORKSPACE_DIR}/task_plan.md` —— 阶段状态推进(pending → in_progress → complete)、新决策
   - `{WORKSPACE_DIR}/findings.md` —— 本步的研究发现、技术洞见、问题根因
   - `{WORKSPACE_DIR}/progress.md` —— 本步的执行动作、被创建/修改的文件、时间戳
   - `{WORKSPACE_DIR}/adapter_state.json` —— 机器可读状态:`next_action`、错误信息、`iteration_count`、`last_update`
2. 运行 `bash {{SKILL_DIR}}/scripts/mark-step-complete.sh N {WORKSPACE_DIR}` —— 把 `adapter_state.json.last_completed_step` 设为 N、`current_step` 设为 N+1。**pre-step-check 依赖这个字段;漏掉会导致下一个 Step 的前置校验失败。**

**验证失败时**(Step 5 专用):

- 立即更新全部规划文件:完整错误日志写 `adapter_state.json`,原因分析写 `findings.md`,失败过程写 `progress.md`
- `iteration_count` 自增
- 调用 Debug 工程师(按模板 D 执行)

**2-Action 规则**:每做 2 次 {Read / Grep / Glob / Bash} 立即更新 `findings.md`。详细说明与背景见 `{{SKILL_DIR}}/references/shared/memory_mechanism.md`。

### Step 0:初始化

```bash
bash {{SKILL_DIR}}/scripts/init-adapter-session.sh {WORKSPACE_DIR} {ModelName} {ModelPath}
```

创建 `input/`、`output/`、`logs/` 以及 `task_plan.md`、`findings.md`、`progress.md`、`adapter_state.json`、`input/input_params.json`。幂等——已存在的文件会跳过,因此续跑安全。

### Step 1:收集上下文

**职责:收集,不决策。**

1. `AskUserQuestion`(若用户尚未在初始 prompt 提供):模型路径、目标设备(npu/gpu)、特殊需求
2. 运行环境审计:
   ```bash
   python {{SKILL_DIR}}/scripts/check_environment.py \
       --output {WORKSPACE_DIR}/input/environment.json --quiet
   npu-smi info 2>/dev/null | grep "Ascend" | wc -l
   ```
3. 产出 `input/device_info.json`(供架构分析师使用的精简子集 + 用户输入——target_device、device_count、device_model、memory_per_device)。`environment.json` 作为完整审计记录保留。

### Step 2:调用架构分析师

```bash
bash {{SKILL_DIR}}/scripts/pre-step-check.sh 2 {WORKSPACE_DIR}
```

原样输出**模板 B**(Agent 调用),然后:

1. 填写 `input/input_params.json`
2. 用脚本构建完整 query(自动注入前置阅读、替换全部变量、绝对化路径):
   ```bash
   bash {{SKILL_DIR}}/scripts/build-agent-query.sh \
       architecture_analyst {WORKSPACE_DIR} \
       --output {WORKSPACE_DIR}/input/query_architecture_analyst.txt
   ```
3. `Agent(subagent_type="architecture-analyst", prompt=<文件 query_architecture_analyst.txt 的内容>, description="Model architecture analysis")`
4. 解析 `output/output_summary.json`

### Step 3:选择适配策略

根据架构分析师的 `similarity` 字段:

- `high` → 直接复用参考模型
- `medium` → 复用并加条件分支
- `low` → 新建模型文件

**【快速跳过规则】**:若 Step 2 输出 `adapter_strategy=direct_use` 且 `requires_new_adapter=false`:
1. 更新 `task_plan.md` → 记录决策:`Step 3-4: direct_use 策略,无需代码修改`
2. 更新 `findings.md` → 记录跳过原因
3. 更新 `adapter_state.json` → `adapter_strategy: "direct_use"`, `requires_new_adapter: false`
4. 运行 `mark-step-complete.sh 3 {WORKSPACE_DIR}`
5. 运行 `mark-step-complete.sh 4 {WORKSPACE_DIR}`
6. 跳转到 **Step 5** 两阶段验证

### Step 4:实施代码修改

**【快速跳过检查】**:若 `adapter_state.json.adapter_strategy=direct_use`:
- 此 Step 已在 Step 3 快速跳过规则中完成,无需执行
- 直接进入 Step 5

**正常执行(仅当 `requires_new_adapter=true`)**:

原则:

1. 优先复用已有架构(`python/sglang/srt/models/` 下查找最接近的参考实现)
2. 修改隔离:使用条件分支(if device.is_npu 之类),不污染 GPU 路径
3. NPU 兼容:优先 torch-native 实现,避免 CUDA-only kernel
4. 最小化修改范围:仅改模型文件 + 必要的注册表项

### Step 5:两阶段验证

```bash
bash {{SKILL_DIR}}/scripts/pre-step-check.sh 5 {WORKSPACE_DIR}
```

**【硬约束】任何错误都触发——包括:server 启动失败 / subprocess 非零退出 / `run_tests.py status != "passed"` / log 含 `Traceback` / `RuntimeError` / `AttributeError` / `Error code` / env 或 config 错误:**

- **禁止行为**:输出 "我认为..."、"让我查查..."、"可能是..."、"应该是..." 等假设性陈述;自行尝试 workaround
- **唯一允许动作**:依序执行下方 Error Handling Flow 步骤 1–10。最终工具调用必须是 `Agent(subagent_type="debug-engineer", ...)`

**Error Handling Flow**(一旦发现错误——不要跳步、不要重排、不要自行调试):

1. **STOP**——不再读 log、不再 grep、不再产出分析文本
2. 原样输出**模板 D**(错误发生入口,包含强制检查)
3. 填写 `input/error_context.json`(error_log、error_type、iteration、max_iterations=20、previous_fixes)
4. 将 `adapter_state.json.next_action` 置为 `"call_debug_engineer"`
5. 用脚本构建完整 query,再调用 Agent:
   ```bash
   bash {{SKILL_DIR}}/scripts/build-agent-query.sh \
       debug_engineer {WORKSPACE_DIR} \
       --output {WORKSPACE_DIR}/input/query_debug_engineer.txt
   ```
6. `Agent(subagent_type="debug-engineer", prompt=<文件 query_debug_engineer.txt 的内容>, description="Debug and fix")`
7. 等待 Debug 工程师产出 `output/fix_instructions.json` 和 `output/debug_report.md`
8. 运行 `post-error-check.sh {WORKSPACE_DIR}`——要求 `fix_instructions.json.status ∈ {"fix_available","fix_verified"}`
9. 原样输出**模板 C**(错误修复复诵,包含强制检查)——所有字段此时均有来自 fix_instructions.json 的真实内容
10. 应用修复,`iteration_count` 自增,重新验证。**最多迭代 20 次。超出 → 上报用户寻求帮助。**

**Stage A:Dummy 验证**

1. 以 `--load-format dummy` 启动服务。stdout → `{WORKSPACE_DIR}/logs/dummy_run.log`。PID → `{WORKSPACE_DIR}/logs/server.pid`。
2. **服务健康检查**(运行测试之前——用于捕获 env/启动错误):
   - Server PID 仍存活(`kill -0 $(cat {WORKSPACE_DIR}/logs/server.pid)`)
   - 端口 8000 已开启(`curl -s localhost:8000/v1/models` 返回)
   - `{WORKSPACE_DIR}/logs/dummy_run.log` 不含 `Traceback` / `RuntimeError` / `AttributeError` / `Error code`
   - 任一不满足 → **STOP,进入 Error Handling Flow**
3. 跑推理冒烟测试(仅在健康检查通过时):
   ```bash
   python {{SKILL_DIR}}/scripts/run_tests.py \
       --port 8000 --wait 300 --mode quick \
       --output {WORKSPACE_DIR}/logs/dummy_inference.json
   ```
4. 读取 `{WORKSPACE_DIR}/logs/dummy_inference.json` 的 `status` 字段:
   - `"passed"` → 将 `adapter_state.json.validation.dummy_passed` 置为 true,关闭服务,进入 Stage B
   - 其他值 → **STOP,进入 Error Handling Flow**

**Stage B:真实权重验证**

1. 以真实权重启动服务。stdout → `{WORKSPACE_DIR}/logs/real_run.log`。PID → `{WORKSPACE_DIR}/logs/server.pid`。
2. **服务健康检查**(运行测试之前):
   - Server PID 仍存活
   - 端口 8000 已开启
   - `{WORKSPACE_DIR}/logs/real_run.log` 不含 `Traceback` / `RuntimeError` / `AttributeError` / `Error code`
   - 任一不满足 → **STOP,进入 Error Handling Flow**
3. 跑推理冒烟测试(仅在健康检查通过时):
   ```bash
   python {{SKILL_DIR}}/scripts/run_tests.py \
       --port 8000 --wait 300 --mode quick \
       --output {WORKSPACE_DIR}/logs/real_inference.json
   ```
4. 读取 `{WORKSPACE_DIR}/logs/real_inference.json` 的 `status` 字段:
   - `"passed"` → 将 `adapter_state.json.validation.real_weight_passed` 置为 true,关闭服务,进入 Step 5.5
   - 其他值 → **STOP,进入 Error Handling Flow**

### Step 5.5:Triton 算子替换(NPU 性能优化)

**【前置条件】**:Step 5 两阶段验证已通过(`adapter_state.json.validation.real_weight_passed == true`)。

**职责**:将整个适配过程中产生的 native PyTorch 算子实现,替换为 sglang 自带的 Triton kernel 版本,以提升 NPU 上的推理性能。

**背景**:在适配过程中,为了让模型在 NPU 上先跑通,可能使用了 native PyTorch 实现作为 fallback(例如从 sgl-kernel-npu site-packages 复制的纯 PyTorch 算子,或在 debug 时用 native torch 算子临时替代不兼容的 NPU kernel)。这些 native 版本功能正确但性能较差。sglang 的 `layers/` 目录下有对应的 Triton 版本,应优先使用以获得更好的推理性能。

**替换原则**:
- **扫描范围**:sgl-kernel-npu 仓中所有不含 `import triton` 的 .py 算子文件(不仅限于 mamba 模块)
- **匹配策略**:按文件名在 sglang 的 `srt/layers/` 目录下查找同名或语义等价的 Triton 版本
- **安全机制**:替换前备份原文件为 `.native_backup`;修改导入路径时备份为 `.pre_triton_backup`
- **回退策略**:若 Triton 版本在 NPU 上报错,立即回退为 native 版本并记录原因

**执行流程**:

1. 运行 Triton 算子替换脚本(dry-run 先预览):
   ```bash
   python {{SKILL_DIR}}/scripts/triton_op_replace.py \
       --sglang-dir {SGLANG_PYTHON_DIR} \
       --sgl-kernel-npu-dir {SGL_KERNEL_NPU_DIR} \
       --workspace-dir {WORKSPACE_DIR} \
       --dry-run
   ```
   - `{SGLANG_PYTHON_DIR}`:sglang 仓的 python 目录(如 `/path/to/sglang/python`)
   - `{SGL_KERNEL_NPU_DIR}`:sgl-kernel-npu 仓的包目录(如 `/path/to/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu`)

2. 确认替换计划无误后,实际执行替换:
   ```bash
   python {{SKILL_DIR}}/scripts/triton_op_replace.py \
       --sglang-dir {SGLANG_PYTHON_DIR} \
       --sgl-kernel-npu-dir {SGL_KERNEL_NPU_DIR} \
       --workspace-dir {WORKSPACE_DIR}
   ```

3. 脚本会自动:
   - 扫描 sgl-kernel-npu 包中所有子模块的 native 实现(不含 `import triton` 的 .py 文件)
   - 在 sglang `srt/layers/` 目录下查找对应的 Triton 版本
   - 复制 Triton 版本到 sgl-kernel-npu 对应位置,备份原 native 文件为 `.native_backup`
   - 修改 sglang 中所有引用被替换算子的导入路径(如将 `from sgl_kernel_npu.xxx.yyy import` 改为相对导入)

4. 读取 `{WORKSPACE_DIR}/output/triton_replace_result.json`,确认替换结果

5. 替换后需要重启服务验证 Triton 算子是否正常工作。如果 Triton 算子导致报错,回退为 native 版本:
   ```bash
   # 回退:恢复所有 .native_backup 文件
   find {SGL_KERNEL_NPU_DIR} -name "*.native_backup" | while read f; do
       mv "$f" "${f%.native_backup}"
   done
   # 回退所有被修改的导入路径文件
   find {SGLANG_PYTHON_DIR} -name "*.pre_triton_backup" | while read f; do
       mv "$f" "${f%.pre_triton_backup}"
   done
   ```

6. 更新 `findings.md` → 记录 Triton 替换结果
7. 运行 `mark-step-complete.sh 5.5 {WORKSPACE_DIR}`

### Step 5.6:GSM8K 精度测试

**【前置条件】**:Step 5.5 Triton 算子替换已完成(或确认无需替换)。

**职责**:使用 GSM8K 数据集测试模型精度。精度不通过时需重新适配算子,直到通过。

**判断标准**:
- accuracy >= threshold 且 invalid == 0 → **通过**
- accuracy < threshold 或 invalid > 0 → **不通过**,需要重新适配算子
- **threshold 规则**:大模型(>=5B参数) threshold=0.8;小模型(<5B参数) threshold=0.3
- **invalid == 0 是硬性条件**:任何 invalid 输出都表示算子适配有问题

**执行流程**:

1. 以真实权重启动服务(使用 Step 5 的启动命令,加上 Triton 算子替换后的配置)

2. 运行 GSM8K 精度测试:
   ```bash
   python {{SKILL_DIR}}/scripts/gsm8k_accuracy_test.py \
       --port 8000 \
       --data-path /home/w00937173/run_file/gsm8k.jsonl \
       --num-questions 10 \
       --parallel {PARALLEL} \
       --workspace-dir {WORKSPACE_DIR} \
       --min-accuracy 0.8 \
       --max-invalid 0.0
   ```
   - `--num-questions`:测试题数(默认 10,快速验证;完整验证用 1319)
   - `--parallel`:并行请求数(根据服务器配置调整)
   - `--min-accuracy`:最低精度阈值(默认 0.8)
   - `--max-invalid`:最大 invalid 比率阈值(默认 0.0,即不允许任何 invalid)

3. 读取 `{WORKSPACE_DIR}/output/gsm8k_accuracy_result.json`:
   - `"passed": true` → 精度通过,进入 Step 6
   - `"passed": false` → 精度不通过,进入精度修复循环:

4. **精度修复循环**(最多迭代 10 次):
   - 分析 `gsm8k_accuracy_result.json` 中的 `reason` 字段:
     - `accuracy_x.xx_below_threshold_0.8` → 精度不足,需要排查算子精度问题
     - `invalid_x.xx_above_threshold_0.0` → 有无效输出,需要排查算子稳定性问题
   - 调用 Debug 工程师定位精度问题根因(按模板 D 执行)
   - 修复后重新运行 Step 5.5 + 5.6
   - **最多迭代 10 次。超出 → 上报用户寻求帮助**

5. 更新 `findings.md` → 记录精度测试结果
6. 更新 `adapter_state.json` → `validation.gsm8k_passed: true/false`
7. 运行 `mark-step-complete.sh 5.6 {WORKSPACE_DIR}`

### Step 6:调用验证工程师

```bash
bash {{SKILL_DIR}}/scripts/pre-step-check.sh 6 {WORKSPACE_DIR}
```

原样输出**模板 B**(Agent 调用)。**职责:编排,不做测试决策。**

1. 填写 `input/test_config.json`(基于架构分析师输出 + Step 5 已验证的 launch_command)
2. 用脚本构建完整 query:
   ```bash
   bash {{SKILL_DIR}}/scripts/build-agent-query.sh \
       test_validator {WORKSPACE_DIR} \
       --output {WORKSPACE_DIR}/input/query_test_validator.txt
   ```
3. `Agent(subagent_type="test-validator", prompt=<文件 query_test_validator.txt 的内容>, description="Test verification")`
4. 解析 `output/test_result.json`

### Step 6.5:精度根因定位 + 修复(可选,触发式)

```bash
bash {{SKILL_DIR}}/scripts/pre-step-check.sh 6.5 {WORKSPACE_DIR}
```

**触发条件**:`adapter_state.precision_suspect == true`(由用户/同事在外部精度评测后置位)。
默认情况下(条件不满足,pre-step-check 返回 exit 2)直接跳过本步,进入 Step 7。

**precision-rca 子 agent 的能力**:不止"定位",而是**定位 + native 替换修复 + drift 归零验证**——
- P1 重现 + 错误分类(prefill_first_token / decode_after_first / random_undefined)
- P2 HF NPU eager 层级 dump(金标准)
- P3 SGLang 层级 dump + 二分首坏层
- P4 算子下钻 + native 替换循环(≤ 8 次,优先 native torch 替换让 drift 归零)
- P5 出 fix.patch + 中文报告

原样输出**模板 B**(Agent 调用),然后:

1. 校验 `input/precision_context.json` 存在且 schema 合法(由 pre-step-check 完成)。**新增字段**:
   - `failure_class`:可选,P1 会自动探测填回
   - `allow_code_fix`:默认 true;false 则只出 patch 草稿不动代码
2. 用脚本构建完整 query:
   ```bash
   bash {{SKILL_DIR}}/scripts/build-agent-query.sh \
       precision_rca {WORKSPACE_DIR} \
       --output {WORKSPACE_DIR}/input/query_precision_rca.txt
   ```
3. `Agent(subagent_type="precision-rca", prompt=<文件 query_precision_rca.txt 的内容>, description="Precision RCA + Fix")`
4. 解析 `output/root_cause.json`:
   - `status=fixed` → 检查 `output/fix.patch` 存在 + `drift_after_fix.max_abs < tolerance`;`next_action="proceed_to_step_7"`,patch 在 Step 7 写入正式 commit
   - `status=located_needs_human_fix` → `next_action="await_human_decision"`,展示 `precision_rca_report.md` + `replacements[]` 排序结果给用户
   - `status=located_inconclusive` → `next_action="await_human_decision"`,提示用户考虑放大 P3 二分粒度或上报算子团队
   - `status=cannot_reproduce` / `hf_load_failed` / `launch_failed_handoff` → 退化为人介入,`next_action="await_human_decision"`

### Step 7:生成产物并提交

1. 生成中文教程**到 workspace**(不是项目根目录):
   ```bash
   python {{SKILL_DIR}}/scripts/generate_report.py \
       --workspace {WORKSPACE_DIR} --model <ModelName> \
       --output {WORKSPACE_DIR}/output/<ModelName>.md
   ```
2. 复查生成的文件;必要时手工修订
3. **仅当**用户明确要求教程入库时:将其复制到规范的 docs 路径并 stage:
   ```bash
   mkdir -p docs/models && cp {WORKSPACE_DIR}/output/<ModelName>.md docs/models/<ModelName>.md
   git add docs/models/<ModelName>.md
   ```
   否则跳过本步——教程保留在 workspace 中作为交接产物。
4. **仅当用户明确要求时**才签名提交:`git commit -sm "feat: adapt <ModelName> for NPU support"`

### Step 8:交接产物

```bash
bash {{SKILL_DIR}}/scripts/check-step-complete.sh {WORKSPACE_DIR}
```

交付物:中文分析报告、运行手册、功能状态矩阵、修改文件清单、(可选)提交 hash。

***

## 文件结构

```
{WORKSPACE_DIR}/
├── adapter_state.json         # 机器可读状态(每个 Step 读写)
├── task_plan.md               # Planning-with-files
├── findings.md                # 研究发现
├── progress.md                # 会话日志
├── input/
│   ├── input_params.json      # 架构分析师输入
│   ├── environment.json       # 主机全量审计(check_environment.py)
│   ├── device_info.json       # 精简版;架构分析师读取此文件
│   ├── error_context.json     # Debug 工程师输入
│   ├── test_config.json       # 验证工程师输入
│   ├── precision_context.json # 精度 RCA 输入(可选)
│   └── query_<agent>.txt      # 由 build-agent-query.sh 生成的当次 query
├── output/
│   ├── output_summary.json    # 架构分析师输出
│   ├── analysis_report.md     # 架构分析师报告
│   ├── fix_instructions.json  # Debug 工程师输出
│   ├── debug_report.md        # Debug 工程师报告
│   ├── test_result.json       # 验证工程师输出
│   ├── test_report.md         # 验证工程师报告
│   ├── root_cause.json        # 精度 RCA + 修复输出(可选)
│   ├── precision_rca_report.md# 精度 RCA + 修复报告(可选)
│   ├── fix.patch              # status=fixed 时的 git diff(可选)
│   ├── native_impl.py         # fix_type=native_replace 时的等价实现(可选)
│   └── <ModelName>.md         # 最终中文教程(Step 7)
└── logs/
    ├── server.pid             # 服务进程 PID
    ├── dummy_run.log          # Dummy 权重服务 stdout
    ├── dummy_inference.json   # run_tests.py(Stage A)
    ├── real_run.log           # 真实权重服务 stdout
    ├── real_inference.json    # run_tests.py(Stage B)
    ├── precision_fix/         # 精度 RCA + 修复阶段的 stdout(P1-P5)
    │   ├── hf_dump.log
    │   ├── sgl_dump.log
    │   └── sgl_run.log
    └── agent_calls/           # 子 Agent 调用审计
        ├── index.jsonl
        └── <agent>_<ts>.txt

# precision/ 目录(Step 6.5 触发时由 precision-rca 创建)
{WORKSPACE_DIR}/precision/
├── env_fingerprint.json
├── hf_layer_outputs/
├── sgl_layer_outputs/
├── layer_diff.json
├── op_dumps/                  # P4 算子级 dump
└── replacements/              # P4 每次 native 替换的 verdict
```

***

## 脚本(索引——完整用法见 `{{SKILL_DIR}}/scripts/README.md`)

| 脚本                        | Step          | 作用                                       |
| ------------------------- | ------------- | ---------------------------------------- |
| `init-adapter-session.sh` | 0             | 创建所有状态/规划文件(含 `skill_dir`)               |
| `check_environment.py`    | 1             | 完整环境审计                                   |
| `pre-step-check.sh`       | Step 2/5/6 之前 | 校验前置条件                                   |
| `build-agent-query.sh`    | 子 Agent 调用之前  | 构建带前置阅读 PREAMBLE 的完整 query               |
| `mark-step-complete.sh`   | 每个 Step 之后    | 更新 last_completed_step / current_step    |
| `run_tests.py`            | 5(A/B)        | 推理冒烟测试                                   |
| `triton_op_replace.py`    | 5.5           | 将 sgl-kernel-npu native 算子替换为 sglang Triton 版本 |
| `gsm8k_accuracy_test.py`  | 5.6           | GSM8K 数据集精度测试                           |
| `post-error-check.sh`     | 5(出错时)        | 校验 Debug 工程师是否已被调用                       |
| `check-step-complete.sh`  | Step 8 之前     | 质量门禁                                     |
| `generate_report.py`      | 7             | 生成最终中文教程                                 |

***

## 质量门禁

- [ ] 服务成功启动
- [ ] 推理请求成功(不仅仅是启动)
- [ ] 功能集已汇报:ACLGraph / DeepEP / MTP / 多模态
- [ ] 容量基线(128k + bs16)已汇报
- [ ] Dummy + 真实权重证据齐全
- [ ] 教程文档存在
- [ ] (用户要求时)单次签名提交
- [ ] 最终响应包含 commit hash(若有)、文件路径、关键命令
- [ ] GSM8K 精度测试通过(accuracy >= 0.8, invalid == 0)
- [ ] Triton 算子替换已完成(或确认无需替换)
- [ ] 若有 Triton 算子回退,回退原因已记录
- [ ] 若 status=fixed:`output/fix.patch` 存在,`root_cause.json.drift_after_fix.max_abs < tolerance.atol`
