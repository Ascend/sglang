# 子 Agent 调用模板

本文档提供三个子 Agent 的调用模板，供主 Skill 参考。

> **【硬约束】所有 Task 调用的 query 必须由 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/build-agent-query.sh` 生成。**
>
> 脚本承担三项工作，主 Skill 不要手工替身：
>
> 1. 从 `adapter_state.json` 读取 `skill_dir` 与 `workspace_dir`（绝对路径）
> 2. 读取 `prompts/<agent>.md`，替换 `{{WORKSPACE_DIR}}` 与 `{{SKILL_DIR}}`
> 3. 在 prompt 顶部注入 **PREAMBLE**——列出该 agent 的 P0 必读参考文档的绝对路径；子 Agent 收到 query 第一屏就是"必须先 Read 这些文件"
>
> 手工替换容易忘 `{{SKILL_DIR}}`、路径不绝对、PREAMBLE 缺失等——脚本把这些集中掉。

> **文件契约**（与 `SKILL.md` File Structure 一致，勿新增非授权输入文件）：
>
> | 子 Agent | `subagent_type` | Input | Output |
> |---------|-----------------|-------|--------|
> | 架构分析师 | `architecture-analyst` | `input/input_params.json`, `input/device_info.json` | `output/output_summary.json`, `output/analysis_report.md` |
> | Debug 工程师 | `debug-engineer` | `input/error_context.json`, `output/output_summary.json`, `input/device_info.json` | `output/fix_instructions.json`, `output/debug_report.md` |
> | 验证工程师 | `test-validator` | `input/test_config.json`, `output/output_summary.json`, `input/device_info.json` | `output/test_result.json`, `output/test_report.md` |
>
> 三个子 Agent 都已在 Trae 中注册为**具名 sub-agent**，**禁止**使用 `general_purpose_task`（会绕过 agent 的专属 system prompt）。Debug 工程师和验证工程师**直接读取** `output/output_summary.json`（架构分析师的输出），不重复拷贝到 input/。迭代上限在 `adapter_state.json.max_iterations=20`，由 error_context.json 透传给 Debug 工程师，prompt 里不再硬编码数字。

## 通用调用流程（三个子 Agent 都用这一套）

```
Step 1. 写 input/*.json（Agent 对应的输入文件）
Step 2. bash build-agent-query.sh <agent_name> <workspace_dir> \
            --output <workspace_dir>/input/query_<agent_name>.txt
Step 3. 用 Read 读回 query 文件内容，把内容作为 Task 的 query
Step 4. 解析 output/*.json
```

- **`<agent_name>` 取值**：`architecture_analyst` / `debug_engineer` / `test_validator`
- **`subagent_type`**：`architecture-analyst` / `debug-engineer` / `test-validator`（中划线）

## 架构分析师调用模板

```python
# 1. 创建输入文件
input_params = {
    "model_path": model_path,
    "target_device": target_device,
    "special_requirements": special_requirements,
    "task_id": task_id,
}
Write(f"{workspace_dir}/input/input_params.json", json.dumps(input_params, indent=2))

# 2. 构建 query（脚本处理所有变量替换 + 注入前置阅读 PREAMBLE）
Bash(f'bash "{skill_dir}/scripts/build-agent-query.sh" '
     f'architecture_analyst "{workspace_dir}" '
     f'--output "{workspace_dir}/input/query_architecture_analyst.txt"')

# 3. 读回 query 并发起 Task
query = Read(f"{workspace_dir}/input/query_architecture_analyst.txt")
result = Task(
    subagent_type="architecture-analyst",
    query=query,
    description="模型架构分析",
)

# 4. 解析输出摘要
output_summary = json.loads(Read(f"{workspace_dir}/output/output_summary.json"))
```

## Debug 工程师调用模板

```python
# 1. 创建唯一输入文件 error_context.json
# Debug 工程师会自行读取 output/output_summary.json 与 input/device_info.json
error_context = {
    "error_type": error_type,
    "error_message": error_message,
    "error_stacktrace": error_stacktrace,
    "error_location": error_location,
    "run_command": run_command,
    "iteration": iteration_count,          # 当前迭代数（0-based，上限 20）
    "max_iterations": 20,                  # 与 SKILL.md / adapter_state.json 保持一致
    "previous_fixes": previous_fixes,      # 历史修复摘要（可选，防止重复尝试）
    "timestamp": datetime.now().isoformat(),
}
Write(f"{workspace_dir}/input/error_context.json", json.dumps(error_context, indent=2))

# 2. 构建 query
Bash(f'bash "{skill_dir}/scripts/build-agent-query.sh" '
     f'debug_engineer "{workspace_dir}" '
     f'--output "{workspace_dir}/input/query_debug_engineer.txt"')

# 3. 读回 query 并发起 Task
query = Read(f"{workspace_dir}/input/query_debug_engineer.txt")
result = Task(
    subagent_type="debug-engineer",
    query=query,
    description="Debug 分析",
)

# 4. 解析修复指令
fix_instructions = json.loads(Read(f"{workspace_dir}/output/fix_instructions.json"))
```

## 验证工程师调用模板

```python
# 1. 创建唯一输入文件 test_config.json
test_config = {
    "model_path": model_path,
    "target_device": target_device,
    "tp_size": output_summary["parallel_config"]["tp_size"],
    "ep_size": output_summary["parallel_config"].get("ep_size"),
    "test_mode": test_mode,                # quick / standard / full
    "compare_with_hf": compare_with_hf,
    "server_port": 8000,
    "timeout_seconds": 300,
    "attention_backend": "ascend" if target_device == "npu" else "flashinfer",
    "context_length": output_summary["resource"].get("recommended_context_length", 4096),
    "max_running_requests": 16,
    "launch_command": launch_command,      # 主 Skill 已在 Step 5 验过的启动命令
    "adapter_file": adapter_file_path,     # Step 4 产出的适配文件路径
    "adapter_class": adapter_class,        # 适配类名
    "config_verified": True,
}
Write(f"{workspace_dir}/input/test_config.json", json.dumps(test_config, indent=2))

# 2. 构建 query
Bash(f'bash "{skill_dir}/scripts/build-agent-query.sh" '
     f'test_validator "{workspace_dir}" '
     f'--output "{workspace_dir}/input/query_test_validator.txt"')

# 3. 读回 query 并发起 Task
query = Read(f"{workspace_dir}/input/query_test_validator.txt")
result = Task(
    subagent_type="test-validator",
    query=query,
    description="测试验证",
)

# 4. 解析测试结果
test_result = json.loads(Read(f"{workspace_dir}/output/test_result.json"))
```

## 修复指令执行

```python
def apply_fix(fix_instructions):
    for fix in fix_instructions["fixes"]:
        fix_type = fix["fix_type"]
        target_file = fix["target_file"]
        
        if fix_type == "REPLACE_BLOCK":
            SearchReplace(target_file, fix["old_code"], fix["new_code"])
        elif fix_type == "INSERT_BEFORE":
            content = Read(target_file)
            new_content = content.replace(fix["anchor_code"], fix["new_code"] + fix["anchor_code"])
            Write(target_file, new_content)
        elif fix_type == "INSERT_AFTER":
            content = Read(target_file)
            new_content = content.replace(fix["anchor_code"], fix["anchor_code"] + fix["new_code"])
            Write(target_file, new_content)
        elif fix_type == "DELETE_BLOCK":
            SearchReplace(target_file, fix["old_code"], "")
        elif fix_type == "ADD_FILE":
            Write(target_file, fix["new_code"])
```

## fix_type说明

| 类型 | 说明 | 必填字段 |
|------|------|----------|
| REPLACE_BLOCK | 替换代码块 | old_code, new_code |
| INSERT_BEFORE | 在锚点前插入 | anchor_code, new_code |
| INSERT_AFTER | 在锚点后插入 | anchor_code, new_code |
| DELETE_BLOCK | 删除代码块 | old_code |
| ADD_FILE | 添加新文件 | new_code, target_file |
