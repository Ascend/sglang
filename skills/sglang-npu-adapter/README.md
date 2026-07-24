# SGLang NPU 适配 Skill — 多平台分发

把新模型适配到 SGLang 并跑通 NPU 设备的端到端 skill,提供三个 AI 编程工具的版本。

| 平台 | 子目录 | 子 agent 调用工具 | prompt 参数名 | 子 agent 注册位置 |
|------|--------|------------------|---------------|------------------|
| Claude Code | `claude-code/` | `Agent(...)` | `prompt` | `.claude/agents/` (项目级) |
| OpenCode | `opencode/` | `task(...)` 小写 | `prompt` | `.opencode/agents/` (项目级) |
| Trae | `trae/` | `Task(...)` 大写 | `query` | 由 Trae UI 创建,本目录 `trae/agents/` 提供 4 份配置素材供复制粘贴 |

> 三版的 **流程、脚本、参考文档、模板完全一致**,只有 `SKILL.md` 与 `references/shared/agent_call_templates.md` 因调用语法不同而内容有别。

---

## 目录结构

```
skills/sglang-npu-adapter/         
├── README.md                      
│
├── claude-code/
│   ├── skills/
│   │   └── sglang-npu-adapter/    # → 安装到目标项目的 .claude/skills/
│   └── agents/                    # → 安装到目标项目的 .claude/agents/
│       ├── architecture-analyst.md
│       ├── debug-engineer.md
│       ├── test-validator.md
│       └── precision-rca.md
│
├── opencode/
│   ├── skills/
│   │   └── sglang-npu-adapter/    # → 安装到目标项目的 .opencode/skills/
│   └── agent/                     # → 安装到目标项目的 .opencode/agents/
│       ├── architecture-analyst.md
│       ├── debug-engineer.md
│       ├── test-validator.md
│       └── precision-rca.md
│
└── trae/
    ├── skills/
    │   └── sglang-npu-adapter/    # → 安装到 ${TRAE_SKILLS_PATH}/
    └── agents/                    # → 4 份配置素材,用于在 Trae UI 创建自定义子 agent
        ├── architecture-analyst.md
        ├── debug-engineer.md
        ├── test-validator.md
        └── precision-rca.md
```

---

## 快速安装

把整个 repo clone 到本地后,在 **目标 SGLang 项目根目录** 执行下面对应平台的命令。`$SKILLS_REPO` 设为本 repo 的本地路径。

### Claude Code

```bash
# 1. 项目级安装 (推荐,只对当前 sglang 项目生效)
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/claude-code/skills/sglang-npu-adapter" .claude/skills/
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/claude-code/agents/." .claude/agents/

# 2. 或 用户级安装 (对所有项目生效)
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/claude-code/skills/sglang-npu-adapter" ~/.claude/skills/
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/claude-code/agents/." ~/.claude/agents/
```

启用后 Claude Code 会自动通过 frontmatter `description` 加载此 skill。子 agent 通过 `.claude/agents/<name>.md` 注册,主流程用 `Agent(subagent_type="<name>", prompt=...)` 调用。

**触发方式**:
- 自然语言:"在SGLang仓中适配NPU版本的xxx模型"
- 显式调用:`/sglang-npu-adapter`

### OpenCode

```bash
# 1. 项目级安装
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/opencode/skills/sglang-npu-adapter" .opencode/skills/
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/opencode/agent/." .opencode/agents/

# 2. 或 用户级
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/opencode/skills/sglang-npu-adapter" ~/.config/opencode/skills/
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/opencode/agent/." ~/.config/opencode/agents/
```

**注意:安装后需要重启 OpenCode。**

**触发方式**:在 OpenCode 会话中描述目标 → 模型识别 skill description 后自动启用。

### Trae

Trae 的安装分两步:**装 skill 文件** + **在 Trae UI 里把 4 个子 agent 配出来**。Claude Code / OpenCode 是把 agent 定义文件丢到 `.claude/agents/` 或 `.opencode/agents/` 就生效,Trae 没有这种本地目录约定,需要走 UI 创建。

#### Step 1 — 装 skill 文件

```bash
# Trae 的 skill 安装路径由 ${TRAE_SKILLS_PATH} 环境变量决定
cp -r "$SKILLS_REPO/skills/sglang-npu-adapter/trae/skills/sglang-npu-adapter" "${TRAE_SKILLS_PATH}/"
```

Trae 版的 `SKILL.md` frontmatter 设了 `user-invocable: true`,可在 Trae UI 中显式调用,也可由模型基于 description 自动识别。

#### Step 2 — 在 Trae 里创建 4 个自定义子 agent

`skills/sglang-npu-adapter/trae/agents/` 下的 4 个 `.md` 文件就是配置素材(每份带 frontmatter 的 `name` / `description` / `tools`,正文是子 agent 的系统提示词)。把它们一一对应导进 Trae:

| 子 agent 文件(`skills/sglang-npu-adapter/trae/agents/`) | 在 Trae 里建的智能体名称 | 建议工具 |
|--------------------------------------|--------------------------|----------|
| `architecture-analyst.md` | `architecture-analyst` | Read、Grep、Glob、Bash、Write、Edit |
| `debug-engineer.md` | `debug-engineer` | Read、Grep、Glob、Bash、Write、Edit、WebSearch、WebFetch |
| `test-validator.md` | `test-validator` | Read、Grep、Glob、Bash、Write、Edit |
| `precision-rca.md` | `precision-rca` | Read、Grep、Glob、Bash、Write、Edit |

**Trae UI 操作路径**(参考 [Trae 官方文档:创建并管理智能体](https://docs.trae.ai/ide/agent?_lang=zh)):


#### Step 3 — 验证

在 Trae 对话框里输入:

```
列出我已经创建的智能体
```

应看到 4 个新增条目。然后输入业务请求,例如:

```
适配 Qwen2.5-7B 到 NPU,模型路径在 /data/Qwen2.5-7B
```

Trae 应识别到 `sglang-npu-adapter` skill 并开始执行 Step 0,需要时会用 `Task(subagent_type="architecture-analyst", query=..., description=...)` 调你刚建好的子 agent——参数名是 `query` 不是 `prompt`,与 Claude Code / OpenCode 不同。

> **常见坑**:Trae 智能体名称大小写敏感,**hyphen 形式必须严格匹配**(`architecture-analyst` ≠ `Architecture-Analyst` ≠ `architecture_analyst`),否则主流程的 `Task(subagent_type=...)` 会找不到对应 agent。

---

## 使用流程

无论哪个平台,启用 skill 后流程相同:

| Step | 动作 | 关键文件 |
|------|------|----------|
| 0 | 初始化 workspace | `init-adapter-session.sh` 生成 `task_plan.md / findings.md / progress.md / adapter_state.json` |
| 1 | 收集环境信息 | `check_environment.py` → `input/environment.json` + `input/device_info.json` |
| 2 | 调用架构分析师子 agent | 产出 `output/output_summary.json` + 中文分析报告 |
| 3 | 选定适配策略 | direct_use / extend_existing / new_implementation |
| 4 | 实施代码修改 | 仅当 `requires_new_adapter=true` |
| 5 | 两阶段验证 | Dummy 权重 → 真实权重,健康检查 + 推理冒烟,失败时调 debug-engineer 子 agent 最多 20 次迭代 |
| 6 | 调用验证工程师子 agent | 跑功能集测试 (ACLGraph / DeepEP / DP-Attention / MTP / 多模态) |
| 6.5 | 精度根因定位 + 修复 (可选) | 设置 `precision_suspect=true` 触发 precision-rca,自动做 HF NPU eager 金标准层级 diff + 二分定位首坏层 + 算子下钻 + native 替换让 drift 归零 + 出 fix.patch |
| 7 | 生成中文教程 | `generate_report.py` |
| 8 | 交接产物 | `check-step-complete.sh` 跑质量门禁 |

---
