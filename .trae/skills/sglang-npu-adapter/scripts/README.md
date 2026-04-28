# 脚本参考

sglang-npu-adapter skill 的辅助脚本。**所有脚本仅适用于 Linux 服务器。**

## 脚本目录

### 约束脚本（校验 / 状态）

| 脚本 | 用途 | 触发时机 |
|----|----|------|
| `init-adapter-session.sh` | 初始化所有规划文件 + adapter_state.json（含 `skill_dir`） | Step 0 |
| `pre-step-check.sh` | 关键步骤前校验前置条件 | Step 2、5、6 之前 |
| `build-agent-query.sh` | 生成含前置阅读 PREAMBLE 的完整 query，供 Task 调用使用 | 调用任一子 Agent 之前 |
| `mark-step-complete.sh` | 把 `last_completed_step` 设为 N、`current_step` 设为 N+1 | 每个 Step 结束后 |
| `post-error-check.sh` | 校验错误发生后 Debug 工程师是否已被调用 | Step 5 出错后 |
| `check-step-complete.sh` | 核对所有 Step 是否完成 + 质量门禁 | Step 8 之前 |

### 执行脚本（数据 / 产物）

| 脚本 | 用途 | 触发时机 |
|----|----|------|
| `check_environment.py` | 完整环境检查（Python、torch、transformers、sglang、GPU/NPU、内存、磁盘） | Step 1 |
| `run_tests.py` | 对运行中的服务执行基础推理冒烟测试（3 个标准用例） | Step 5 Stage A & Stage B |
| `generate_report.py` | 聚合 Agent 输出 + git 信息 → 中文教程文档 | Step 7 |

## 用法

### 会话初始化（Step 0）
```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/init-adapter-session.sh" \
    "{WORKSPACE_DIR}" "{ModelName}" "{ModelPath}"
```

### 环境检查（Step 1）
```bash
python "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/check_environment.py" \
    --output "{WORKSPACE_DIR}/input/environment.json" --quiet
```

### Step 前置校验（Step 2、5、6 之前）
```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/pre-step-check.sh" <step_num> "{WORKSPACE_DIR}"
```

### 构建子 Agent query（Task 调用之前）
```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/build-agent-query.sh" \
    <agent_name> "{WORKSPACE_DIR}" \
    --output "{WORKSPACE_DIR}/input/query_<agent_name>.txt"
```
- `<agent_name>`：`architecture_analyst` / `debug_engineer` / `test_validator`
- 脚本自动从 `adapter_state.json` 读 `skill_dir`，完成变量替换，并在 prompt 顶部注入 PREAMBLE（强制子 Agent 先 Read P0 参考文档再分析）
- **所有三种子 Agent 的 Task 调用前都必须走这一步**——手工替换变量容易漏 `{{SKILL_DIR}}`、路径不绝对等

**【审计】** 每次调用都会**自动**在 `{WORKSPACE_DIR}/logs/agent_calls/` 下留下记录：
- `<agent>_<YYYY-MM-DD_HH-MM-SS>.txt`——该次调用传给子 Agent 的完整 query 快照（不会被覆盖）
- `index.jsonl`——追加式索引，每行一个 JSON：`timestamp`、`agent`、`subagent_type`、`snapshot`（文件绝对路径）、`bytes`、`lines`、`sha256`

常用审计方式：
```bash
# 列出所有调用历史
ls "{WORKSPACE_DIR}/logs/agent_calls/"

# 检查最近一次的 debug_engineer 收到了什么
ls -t "{WORKSPACE_DIR}/logs/agent_calls/debug_engineer_"*.txt | head -1 | xargs cat | less

# 对比两次调用是否相同（sha256 匹配 = 同一份 query）
cat "{WORKSPACE_DIR}/logs/agent_calls/index.jsonl" | jq -s '.[] | {agent, sha256}'
```

### 标记 Step 完成（每个 Step 结束后）
```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/mark-step-complete.sh" <step_num> "{WORKSPACE_DIR}"
```
更新 `adapter_state.json` 的 `last_completed_step` 与 `current_step` 字段。**pre-step-check 依赖此字段；漏掉会导致后续 Step 的前置校验失败。**

### 推理冒烟测试（Step 5 Stage A / Stage B，服务启动后）
```bash
python "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/run_tests.py" \
    --port 8000 --wait 300 --mode quick \
    --output "{WORKSPACE_DIR}/logs/{dummy|real}_inference.json"
```

### 错误后校验（Step 5 出错后）
```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/post-error-check.sh" "{WORKSPACE_DIR}"
```

### 完成度检查（Step 8 之前）
```bash
bash "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/check-step-complete.sh" "{WORKSPACE_DIR}"
```

### 最终报告生成（Step 7）
```bash
python "${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/generate_report.py" \
    --workspace "{WORKSPACE_DIR}" --model "<ModelName>" \
    --output "{WORKSPACE_DIR}/output/<ModelName>.md"
```

## 校验矩阵（约束脚本）

| 检查点 | 脚本 | 校验内容 |
|-----|----|------|
| 架构分析师之前 | `pre-step-check.sh 2` | device_info.json、input_params.json、Step 1 已完成 |
| 验证之前 | `pre-step-check.sh 5` | Step 4 已完成、无待处理的 Debug 工程师调用 |
| 验证工程师之前 | `pre-step-check.sh 6` | Step 5 已通过、test_config.json、iteration ≤ 5 |
| 错误发生后 | `post-error-check.sh` | Debug 工程师已被调用、fix_instructions.json 合法 |
| 任务完成时 | `check-step-complete.sh` | Step 0–8 全部完成、质量门禁条件 |

## 数据流矩阵（执行脚本）

| 脚本 | 读取 | 写入 | 下游消费者 |
|----|----|----|------|
| `build-agent-query.sh` | `adapter_state.json`、`prompts/<agent>.md` | stdout 或 `input/query_<agent>.txt` | Task 调用的 `query` 参数 |
| `check_environment.py` | （主机状态） | `input/environment.json` | Step 1 → `device_info.json` 构造器 |
| `run_tests.py` | `http://localhost:8000/v1/chat/completions` | `{WORKSPACE_DIR}/logs/{dummy,real}_inference.json` | Step 5 阶段门禁 → `adapter_state.json.validation` |
| `generate_report.py` | `{WORKSPACE_DIR}/output/*.json`、`{WORKSPACE_DIR}/output/*.md`、`{WORKSPACE_DIR}/adapter_state.json`、git | `{WORKSPACE_DIR}/output/<ModelName>.md` | Step 7 交付 |
