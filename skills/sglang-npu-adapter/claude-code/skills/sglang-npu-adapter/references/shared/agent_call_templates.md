# Agent 调用模板

主流程调用 4 个子 agent 的标准模板。所有调用必须走 `build-agent-query.sh` 生成 query,不能手写。

## 通用模板

```python
# 1. 构建 query(每个 agent 文件名不同)
Bash("bash {{SKILL_DIR}}/scripts/build-agent-query.sh <agent_name> {{WORKSPACE_DIR}} --output {{WORKSPACE_DIR}}/input/query_<agent>.txt")

# 2. Read query 文件内容
Read("{{WORKSPACE_DIR}}/input/query_<agent>.txt")

# 3. 调用 Agent
Agent(
    subagent_type="<subagent-type>",   # 注意是 hyphen 形式
    prompt=<上一步 Read 的文件内容>,
    description="<3-5 词描述>"
)
```

`<agent_name>` 与 `<subagent-type>` 映射:

| build-agent-query.sh 参数 | Agent 工具 subagent_type | prompt 文件 |
|----|----|----|
| `architecture_analyst` | `architecture-analyst` | `prompts/model_analyzer.md` |
| `debug_engineer`       | `debug-engineer`       | `prompts/debug_engineer.md` |
| `test_validator`       | `test-validator`       | `prompts/test_validator.md` |
| `precision_rca`        | `precision-rca`        | `prompts/precision_rca.md` |

## 调用前置条件(由 pre-step-check.sh 校验)

| 调用 | 前置 |
|------|------|
| architecture-analyst (Step 2) | `input/input_params.json` + `input/device_info.json` 存在,`last_completed_step >= 1` |
| debug-engineer (任何 Step 出错时) | `input/error_context.json` 存在,`iteration < 20` |
| test-validator (Step 6) | `input/test_config.json` 存在,`last_completed_step >= 5`,`next_action != call_debug_engineer` |
| precision-rca (Step 6.5) | `input/precision_context.json` 存在且 schema 合法,`precision_suspect == true`,`last_completed_step >= 6` |

## 调用后处理

| 调用 | 期望产物 | 失败动作 |
|------|---------|---------|
| architecture-analyst | `output/output_summary.json` + `output/analysis_report.md` | 缺产物 → 调 debug-engineer |
| debug-engineer | `output/fix_instructions.json` + `output/debug_report.md` | `status=needs_human` → 上报用户;`status=inconclusive` → 升级人审 |
| test-validator | `output/test_result.json` + `output/test_report.md` | `status=failed` → 回 Step 5 调 debug-engineer;`status=partial` → 标 known issue 继续 |
| precision-rca | `output/root_cause.json` + `output/precision_rca_report.md` | `next_action=await_human_decision` |

## 审计轨迹

`build-agent-query.sh` 会自动:
- 把生成的 query 写到 `logs/agent_calls/<agent>_<timestamp>.txt`(永不覆盖)
- 在 `logs/agent_calls/index.jsonl` 追加一行(时间、agent、字节数、sha256、快照路径)

事后排查时,Read `index.jsonl` 可以回放每次调用。

## 反模式

- ❌ 跳过 `build-agent-query.sh`,手写 prompt 调用 Agent —— 失去 P0 前置阅读注入
- ❌ 在 prompt 里嵌占位符如 `{{WORKSPACE_DIR}}` —— 必须由脚本替换为绝对路径,子 agent 无法解析占位符
- ❌ 改 subagent_type 大小写(如 `Architecture-Analyst`)—— 项目级 agent 文件名必须 lowercase-with-hyphens
