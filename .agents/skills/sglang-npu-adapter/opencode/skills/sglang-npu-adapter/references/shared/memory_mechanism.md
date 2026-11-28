# Planning-with-Files & 2-Action 规则

主流程在长链路任务里**靠四份文件做短期记忆**——上下文压缩后还能恢复执行所必需。

## 四份规划文件

| 文件 | 角色 | 谁写 | 何时写 |
|------|------|------|--------|
| `adapter_state.json` | 机器可读状态(Step 编号、迭代数、next_action、validation 状态) | 主流程 + 脚本 | 每个 Step 之后(强制) |
| `task_plan.md` | 阶段进度 + 计划调整 | 主流程 | 每个 Step 之后(强制) |
| `findings.md` | 研究发现、技术洞见、问题根因 | 主流程 | 2-Action 规则触发 + 每个 Step 之后 |
| `progress.md` | 行动日志(创建/修改的文件、命令、时间戳) | 主流程 | 每个 Step 之后(强制) |

由 `init-adapter-session.sh` 初始化模板。由 `mark-step-complete.sh` 校验"本步骤是否真的更新过"——sha256 与上一步快照对比,未变则拒绝标 complete。

## 2-Action 规则

每做 2 次 {Read / Grep / Glob / Bash} 之后,必须更新 `findings.md`。

**为什么**:Claude 的上下文是滑动窗口,长任务中过去的 grep 结果会被压缩出去。把要点落盘到 `findings.md`,后续可以 Read 回来,**等价于"短期记忆持久化"**。

**写什么**:不是流水账,而是"如果我现在被压缩,下一会话只看 findings.md 能不能接着干"的最小集合——关键文件路径、关键决策、未解决问题清单。

## 复活检查清单(每个 Step 开始前)

1. Read `adapter_state.json` → 确认 `current_step`
2. Read `task_plan.md` → 确认阶段状态
3. Read `findings.md` → 拿上下文摘要(给 Agent 调用用)
4. Read `progress.md` → 确认上次执行结果
5. 看 `next_action`:若 `== "call_debug_engineer"` → 必须先调 Debug,不能执行其他 Step
6. 看 `iteration_count`:>20 → 上报用户,不再迭代

## 为什么不直接靠对话上下文

长适配任务有 5-50 次工具调用,**Claude 的上下文会被压缩**。压缩后:
- 工具结果可能丢失(只保留对话主干)
- 但 SKILL.md + 这四份文件总能 Read 回来

所以"什么放 context 什么放文件"的边界是:
- **放 context**:当前 Step 的瞬时计算、子 Agent query 内容
- **放文件**:跨 Step 复用的状态、累积的发现、决策记录

## 反模式

- ❌ "我先把所有错误都记在脑子里,最后一次性更新 findings.md"——会忘
- ❌ "adapter_state.json 我直接手动 jq 改一改"——绕过 mark-step-complete.sh 的校验,后果是下个 Step 的 pre-check 失败
- ❌ "task_plan.md 我每步都重写一遍"——失去历史轨迹,失败时无法回溯
