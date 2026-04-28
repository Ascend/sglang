---
name: sglang-npu-adapter
description: 将新模型适配到 SGLang 框架以支持 NPU 设备。当用户需要在 NPU上运行 SGLang 尚未支持的模型时使用此技能。自动分析模型架构、生成适配代码、调试问题并验证正确性。
user-invocable: true
---

# SGLang NPU 模型适配技能

将新模型适配到 SGLang，使其可在 NPU 上运行。本文件是顶层流程；补充文档以内联引用方式列出——**走到哪读到哪**，不要一次性全读。

## 参考索引

| 主题                       | 文件                                                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| 规划文件、2-Action 规则、防遗忘检查清单 | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/references/shared/memory_mechanism.md`                           |
| 内容信任（Content Trust）分级    | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/references/shared/security_boundary.md`                          |
| 脚本用法与参数                  | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/README.md`                                               |
| Agent 调用模板               | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/references/shared/agent_call_templates.md`                       |
| Agent 提示词                | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/prompts/{model_analyzer,debug_engineer,test_validator}.md`       |

复诵模板已内联于本文件（见下方 **复诵模板** 小节）。

***

## 硬性约束

- 每个任务在 `.trae/workspace/` 下使用独立文件夹存放 input/output
- **所有过程产物都写到** **`{WORKSPACE_DIR}/`** **下**——规划文件、Agent 输入输出 JSON、服务端 stdout、推理日志、调试报告、PID、临时文件均在此。**绝不**把过程产物写到项目根目录。shell 的重定向/输出必须始终以 `{WORKSPACE_DIR}/` 前缀（如 `> {WORKSPACE_DIR}/logs/foo.log`，不是 `> logs/foo.log`）
- **绝不升级 transformers**
- 主实现根目录 = 当前项目仓库
- 启动命令：`export PYTHONPATH=${PWD}/python:$PYTHONPATH`，默认 API 端口 `8000`
- 功能优先级：验证 ACLGraph / DeepEP / DP-Attention / MTP / 多模态
- **代码修改最小化，仅针对目标模型**
- **最终交付：单次签名提交**（`git commit -sm ...`）
- **最终文档使用中文**
- **Dummy-first 加速，但真实权重验证强制执行**
- **任务追踪：必须使用 planning-with-files 模式**——见 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/references/shared/memory_mechanism.md`
- **内容信任**：不可信的网页内容仅写入 `findings.md`——见 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/references/shared/security_boundary.md`

***

## 架构设计

| 子 Agent   | 角色      | 触发时机    | `subagent_type`        | 提示词                                                                              |
| --------- | ------- | ------- | ---------------------- | -------------------------------------------------------------------------------- |
| 架构分析师     | 模型架构分析  | Step 2  | `architecture-analyst` | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/prompts/model_analyzer.md`               |
| Debug 工程师 | 错误诊断与修复 | 发生任何错误时 | `debug-engineer`       | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/prompts/debug_engineer.md`               |
| 验证工程师     | 测试验证    | Step 6  | `test-validator`       | `${TRAE_SKILLS_PATH}/sglang-npu-adapter/prompts/test_validator.md`               |

每个子 Agent 在 Trae 中都有**专用**的 `subagent_type`。调用遵循 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/references/shared/agent_call_templates.md` 中的模板。

***

## 复诵模板

### 模板 A —— Step 级（每个 Step 开始前）

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
**强制检查（每个 Step 开始前必须执行）**：
1. Read `adapter_state.json` → 确认 `current_step` 与即将执行的 Step N 一致
2. Read `task_plan.md` → 确认当前阶段状态
3. Read `findings.md` → 获取上下文摘要用于 Agent 调用
4. Read `progress.md` → 确认上次执行结果
5. 若 `next_action == "call_debug_engineer"` → 必须先调 Debug 工程师，不能执行其他 Step
6. 确认 `iteration_count <= 20` → 超出则上报用户

### 模板 B —— Agent 调用（调用架构分析师 / Debug 工程师 / 验证工程师之前）

```
=== Agent调用准备 ===
调用目标: [架构分析师 / Debug 工程师 / 验证工程师] ([subagent_type])
输入文件: [input/*.json]
预期输出: [output/*.json]
当前上下文摘要: [Key findings from findings.md - 必须包含架构类型、NPU兼容性、并行配置]
=== 开始调用 [架构分析师 / Debug 工程师 / 验证工程师] ===
```
**强制检查（Agent 调用前必须执行）**：
1. 确认输入文件存在（如调用架构分析师需 `input/input_params.json`、`input/device_info.json`）
2. 确认 `findings.md` 已更新到最新状态
3. 用脚本构建 query：`bash build-agent-query.sh <agent_type> {WORKSPACE_DIR} --output {WORKSPACE_DIR}/input/query_<agent>.txt`
4. Task 调用必须使用脚本生成的 query 文件内容

### 模板 C —— 错误修复复诵（Debug 工程师返回修复之后）

```
=== 错误修复复诵 ===
错误类型: [from input/error_context.json]
Debug 工程师诊断 (照抄自 fix_instructions.json.diagnosis): [...]
Debug 工程师修复步骤 (照抄自 fix_instructions.json.steps): [...]
迭代次数: [X/20]
剩余尝试: [20-X]
=== 应用 Debug 工程师修复并重试 ===
```
**强制检查（输出模板 C 前必须执行）**：
1. 确认 `output/fix_instructions.json` 存在
2. Read `fix_instructions.json` → 确认 `status ∈ {"fix_available", "fix_verified"}`
3. 若 status 不满足 → STOP，回到模板 D，不输出此复诵
4. 照抄 diagnosis 和 steps 字段，不能自行概括或修改

### 模板 D —— 错误发生（Debug 工程师调用入口，任何错误都触发）

```
=== 错误发生（Debug 工程师入口）===
错误类型: [copy from error log, 1 line]
错误阶段: [Stage A / Stage B / server launch / subprocess / 其他]
迭代次数: [X/20]
约束: 本 Skill 禁止自行调试此错误。
禁止行为: 输出 "我认为...", "让我查查...", "可能是...", "应该是..." 等假设或探查性陈述
下一工具调用 (必须):
    bash ${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/build-agent-query.sh \
         debug_engineer "{WORKSPACE_DIR}" \
         --output "{WORKSPACE_DIR}/input/query_debug_engineer.txt"
    Task(
        subagent_type="debug-engineer",
        query=<文件 query_debug_engineer.txt 的内容>,
        description="Debug and fix"
    )
=== 立即执行上述 Task 调用。不做其他操作。===
```

***

## 执行流程

### 防遗忘清单（每个 Step 都要照做）

**每个 Step 之前**：

- 读 `adapter_state.json`、`task_plan.md`、`findings.md`、`progress.md`（4 份规划文件）
- 确认 `current_step`；若 `next_action == "call_debug_engineer"` 必须先调 Debug 工程师
- 确认 `iteration_count <= 20`
- 原样输出**模板 A**（包含强制检查）

**每个 Step 之后**：

1. 更新 4 份规划文件：
   - `{WORKSPACE_DIR}/task_plan.md` —— 阶段状态推进（pending → in\_progress → complete）、新决策
   - `{WORKSPACE_DIR}/findings.md` —— 本步的研究发现、技术洞见、问题根因
   - `{WORKSPACE_DIR}/progress.md` —— 本步的执行动作、被创建/修改的文件、时间戳
   - `{WORKSPACE_DIR}/adapter_state.json` —— 机器可读状态：`next_action`、错误信息、`iteration_count`、`last_update`
2. 运行 `bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/mark-step-complete.sh" N "{WORKSPACE_DIR}"` —— 把 `adapter_state.json.last_completed_step` 设为 N、`current_step` 设为 N+1。**pre-step-check 依赖这个字段；漏掉会导致下一个 Step 的前置校验失败。**

**验证失败时**（Step 5 专用）：

- 立即更新全部规划文件：完整错误日志写 `adapter_state.json`，原因分析写 `findings.md`，失败过程写 `progress.md`
- `iteration_count` 自增
- 调用 Debug 工程师（按模板 D 执行）

**2-Action 规则**：每做 2 次 {Read / Grep / Glob / SearchCodebase} 立即更新 `findings.md`。详细说明与背景见 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/references/shared/memory_mechanism.md`。

### Step 0：初始化

```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/init-adapter-session.sh" "{WORKSPACE_DIR}" "{ModelName}" "{ModelPath}"
```

创建 `input/`、`output/`、`logs/` 以及 `task_plan.md`、`findings.md`、`progress.md`、`adapter_state.json`、`input/input_params.json`。幂等——已存在的文件会跳过，因此续跑安全。

### Step 1：收集上下文

**职责：收集，不决策。**

1. `AskUserQuestion`：模型路径、目标设备（npu/gpu）、特殊需求
2. 运行环境审计：
   ```bash
   python "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/check_environment.py" \
       --output "{WORKSPACE_DIR}/input/environment.json" --quiet
   npu-smi info 2>/dev/null | grep "Ascend" | wc -l
   ```
3. 产出 `input/device_info.json`（供架构分析师使用的精简子集 + 用户输入——target\_device、device\_count、device\_model、memory\_per\_device）。`environment.json` 作为完整审计记录保留。

### Step 2：调用架构分析师

```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/pre-step-check.sh" 2 "{WORKSPACE_DIR}"
```

原样输出**模板 B**（Agent 调用），然后：

1. 填写 `input/input_params.json`
2. 用脚本构建完整 query（自动注入前置阅读、替换全部变量、绝对化路径）：
   ```bash
   bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/build-agent-query.sh" \
       architecture_analyst "{WORKSPACE_DIR}" \
       --output "{WORKSPACE_DIR}/input/query_architecture_analyst.txt"
   ```
3. `Task(subagent_type="architecture-analyst", query=<文件 query_architecture_analyst.txt 的内容>, description="Model architecture analysis")`
4. 解析 `output/output_summary.json`

### Step 3：选择适配策略

根据架构分析师的 `similarity` 字段：

- `high` → 直接复用参考模型
- `medium` → 复用并加条件分支
- `low` → 新建模型文件

**【快速跳过规则】**：若 Step 2 输出 `adapter_strategy=direct_use` 且 `requires_new_adapter=false`：
1. 更新 `task_plan.md` → 记录决策：`Step 3-4: direct_use 策略，无需代码修改`
2. 更新 `findings.md` → 记录跳过原因
3. 更新 `adapter_state.json` → `adapter_strategy: "direct_use"`, `requires_new_adapter: false`
4. 运行 `mark-step-complete.sh 3 {WORKSPACE_DIR}`
5. 运行 `mark-step-complete.sh 4 {WORKSPACE_DIR}`
6. 跳转到 **Step 5** 两阶段验证

### Step 4：实施代码修改

**【快速跳过检查】**：若 `adapter_state.json.adapter_strategy=direct_use`：
- 此 Step 已在 Step 3 快速跳过规则中完成，无需执行
- 直接进入 Step 5

**正常执行（仅当 `requires_new_adapter=true`）**：

原则：

1. 优先复用已有架构
2. 修改隔离：使用条件分支
3. NPU 兼容：优先 torch-native 实现
4. 最小化修改范围

### Step 5：两阶段验证

```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/pre-step-check.sh" 5 "{WORKSPACE_DIR}"
```

**【硬约束】任何错误都触发——包括：server 启动失败 / subprocess 非零退出 / `run_tests.py status != "passed"` / log 含 `Traceback` / `RuntimeError` / `AttributeError` / `Error code` / env 或 config 错误：**

- **禁止行为**：输出 "我认为..."、"让我查查..."、"可能是..."、"应该是..." 等假设性陈述；自行尝试 workaround
- **唯一允许动作**：依序执行下方 Error Handling Flow 步骤 1–10。最终工具调用必须是 `Task(subagent_type="debug-engineer", ...)`

**Error Handling Flow**（一旦发现错误——不要跳步、不要重排、不要自行调试）：

1. **STOP**——不再读 log、不再 grep、不再产出分析文本
2. 原样输出**模板 D**（错误发生入口，包含强制检查）
3. 填写 `input/error_context.json`（error\_log、error\_type、iteration、max\_iterations=20、previous\_fixes）
4. 将 `adapter_state.json.next_action` 置为 `"call_debug_engineer"`
5. 用脚本构建完整 query，再调用 Task：
   ```bash
   bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/build-agent-query.sh" \
       debug_engineer "{WORKSPACE_DIR}" \
       --output "{WORKSPACE_DIR}/input/query_debug_engineer.txt"
   ```
6. `Task(subagent_type="debug-engineer", query=<文件 query_debug_engineer.txt 的内容>, description="Debug and fix")`
7. 等待 Debug 工程师产出 `output/fix_instructions.json` 和 `output/debug_report.md`
8. 运行 `post-error-check.sh "{WORKSPACE_DIR}"`——要求 `fix_instructions.json.status ∈ {"fix_available","fix_verified"}`
9. 原样输出**模板 C**（错误修复复诵，包含强制检查）——所有字段此时均有来自 fix\_instructions.json 的真实内容
10. 应用修复，`iteration_count` 自增，重新验证。**最多迭代 20 次。超出 → 上报用户寻求帮助。**

**Stage A：Dummy 验证**

1. 以 `--load-format dummy` 启动服务。stdout → `{WORKSPACE_DIR}/logs/dummy_run.log`。PID → `{WORKSPACE_DIR}/logs/server.pid`。
2. **服务健康检查**（运行测试之前——用于捕获 env/启动错误）：
   - Server PID 仍存活（`kill -0 $(cat {WORKSPACE_DIR}/logs/server.pid)`）
   - 端口 8000 已开启（`curl -s localhost:8000/v1/models` 返回）
   - `{WORKSPACE_DIR}/logs/dummy_run.log` 不含 `Traceback` / `RuntimeError` / `AttributeError` / `Error code`
   - 任一不满足 → **STOP，进入 Error Handling Flow**
3. 跑推理冒烟测试（仅在健康检查通过时）：
   ```bash
   python "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/run_tests.py" \
       --port 8000 --wait 300 --mode quick \
       --output "{WORKSPACE_DIR}/logs/dummy_inference.json"
   ```
4. 读取 `{WORKSPACE_DIR}/logs/dummy_inference.json` 的 `status` 字段：
   - `"passed"` → 将 `adapter_state.json.validation.dummy_passed` 置为 true，关闭服务，进入 Stage B
   - 其他值 → **STOP，进入 Error Handling Flow**

**Stage B：真实权重验证**

1. 以真实权重启动服务。stdout → `{WORKSPACE_DIR}/logs/real_run.log`。PID → `{WORKSPACE_DIR}/logs/server.pid`。
2. **服务健康检查**（运行测试之前）：
   - Server PID 仍存活（`kill -0 $(cat {WORKSPACE_DIR}/logs/server.pid)`）
   - 端口 8000 已开启
   - `{WORKSPACE_DIR}/logs/real_run.log` 不含 `Traceback` / `RuntimeError` / `AttributeError` / `Error code`
   - 任一不满足 → **STOP，进入 Error Handling Flow**
3. 跑推理冒烟测试（仅在健康检查通过时）：
   ```bash
   python "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/run_tests.py" \
       --port 8000 --wait 300 --mode quick \
       --output "{WORKSPACE_DIR}/logs/real_inference.json"
   ```
4. 读取 `{WORKSPACE_DIR}/logs/real_inference.json` 的 `status` 字段：
   - `"passed"` → 将 `adapter_state.json.validation.real_weight_passed` 置为 true，关闭服务，进入 Step 6
   - 其他值 → **STOP，进入 Error Handling Flow**

### Step 6：调用验证工程师

```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/pre-step-check.sh" 6 "{WORKSPACE_DIR}"
```

原样输出**模板 B**（Agent 调用）。**职责：编排，不做测试决策。**

1. 填写 `input/test_config.json`（基于架构分析师输出 + Step 5 已验证的 launch\_command）
2. 用脚本构建完整 query：
   ```bash
   bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/build-agent-query.sh" \
       test_validator "{WORKSPACE_DIR}" \
       --output "{WORKSPACE_DIR}/input/query_test_validator.txt"
   ```
3. `Task(subagent_type="test-validator", query=<文件 query_test_validator.txt 的内容>, description="Test verification")`
4. 解析 `output/test_result.json`

### Step 7：生成产物并提交

1. 生成中文教程**到 workspace**（不是项目根目录）：
   ```bash
   python "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/generate_report.py" \
       --workspace "{WORKSPACE_DIR}" --model "<ModelName>" \
       --output "{WORKSPACE_DIR}/output/<ModelName>.md"
   ```
2. 复查生成的文件；必要时手工修订
3. **仅当**团队约定要求教程入库时：将其复制到规范的 docs 路径并 stage：
   ```bash
   mkdir -p docs/models && cp "{WORKSPACE_DIR}/output/<ModelName>.md" "docs/models/<ModelName>.md"
   git add docs/models/<ModelName>.md
   ```
   否则跳过本步——教程保留在 workspace 中作为交接产物。
4. 签名提交（仅代码改动，可选地包含上一步 stage 的 docs/models/\*.md）：`git commit -sm "feat: adapt <ModelName> for NPU support"`

### Step 8：交接产物

```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/check-step-complete.sh" "{WORKSPACE_DIR}"
```

交付物：中文分析报告、运行手册、功能状态矩阵、修改文件清单、提交 hash。

***

## 文件结构

```
{WORKSPACE_DIR}/
├── adapter_state.json         # 机器可读状态（每个 Step 读写）
├── task_plan.md               # Planning-with-files
├── findings.md                # 研究发现
├── progress.md                # 会话日志
├── input/
│   ├── input_params.json      # 架构分析师输入
│   ├── environment.json       # 主机全量审计（check_environment.py）
│   ├── device_info.json       # 精简版；架构分析师读取此文件
│   ├── error_context.json     # Debug 工程师输入
│   ├── test_config.json       # 验证工程师输入
│   └── query_<agent>.txt      # 由 build-agent-query.sh 生成的当次 query（会被下次覆盖）
├── output/
│   ├── output_summary.json    # 架构分析师输出
│   ├── analysis_report.md     # 架构分析师报告
│   ├── fix_instructions.json  # Debug 工程师输出
│   ├── debug_report.md        # Debug 工程师报告
│   ├── test_result.json       # 验证工程师输出
│   ├── test_report.md         # 验证工程师报告
│   └── <ModelName>.md         # 最终中文教程（Step 7；可选地复制到 docs/models/）
└── logs/
    ├── server.pid             # 服务进程 PID（用于健康检查 + 关停）
    ├── dummy_run.log          # Dummy 权重服务 stdout
    ├── dummy_inference.json   # run_tests.py（Stage A）
    ├── real_run.log           # 真实权重服务 stdout
    ├── real_inference.json    # run_tests.py（Stage B）
    └── agent_calls/           # 子 Agent 调用审计（由 build-agent-query.sh 自动维护）
        ├── index.jsonl        # 每次调用一行：时间、agent、字节数、sha256、快照路径
        └── <agent>_<ts>.txt   # 每次调用的完整 query 快照（永不覆盖）
```

***

## 脚本（索引——完整用法见 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/README.md`）

| 脚本                        | Step          | 作用                                       |
| ------------------------- | ------------- | ---------------------------------------- |
| `init-adapter-session.sh` | 0             | 创建所有状态/规划文件（含 `skill_dir`）               |
| `check_environment.py`    | 1             | 完整环境审计                                   |
| `pre-step-check.sh`       | Step 2/5/6 之前 | 校验前置条件                                   |
| `build-agent-query.sh`    | 子 Agent 调用之前  | 构建带前置阅读 PREAMBLE 的完整 query               |
| `mark-step-complete.sh`   | 每个 Step 之后    | 更新 last\_completed\_step / current\_step |
| `run_tests.py`            | 5（A/B）        | 推理冒烟测试                                   |
| `post-error-check.sh`     | 5（出错时）        | 校验 Debug 工程师是否已被调用                       |
| `check-step-complete.sh`  | Step 8 之前     | 质量门禁                                     |
| `generate_report.py`      | 7             | 生成最终中文教程                                 |

***

## 质量门禁

- [ ] 服务成功启动
- [ ] 推理请求成功（不仅仅是启动）
- [ ] 功能集已汇报：ACLGraph / DeepEP / MTP / 多模态
- [ ] 容量基线（128k + bs16）已汇报
- [ ] Dummy + 真实权重证据齐全
- [ ] 教程文档存在
- [ ] 单次签名提交
- [ ] 最终响应包含 commit hash、文件路径、关键命令

