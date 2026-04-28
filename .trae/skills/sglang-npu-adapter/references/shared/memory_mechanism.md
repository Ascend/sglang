# 任务追踪与记忆机制

> **说明**：可执行的防遗忘清单已内联到 `SKILL.md` → `执行流程` → `防遗忘清单`。本文件保留背景、生命周期、2-Action 规则的详细版本，供深入阅读。

## 四份规划文件

采用 planning-with-files 模式，用四份文件承载任务记忆：

1. **`{WORKSPACE_DIR}/task_plan.md`** —— 任务阶段、进度、关键决策（模板 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/templates/task_plan.md`）
2. **`{WORKSPACE_DIR}/findings.md`** —— 研究发现、模型分析、技术洞见（模板 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/templates/findings.md`）
3. **`{WORKSPACE_DIR}/progress.md`** —— 会话日志、执行步骤、测试结果（模板 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/templates/progress.md`）
4. **`{WORKSPACE_DIR}/adapter_state.json`** —— 机器可读状态（`current_step`、`iteration_count`、`next_action`、验证 flag）

## 生命周期

| 时机 | 动作 |
|---|---|
| Step 0 | 用 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/init-adapter-session.sh` 从模板创建全部 4 份文件 |
| 每个 Step 之前 | 读全部 4 份文件以恢复上下文 |
| 每个 Step 之后 | 用进度与发现更新全部 4 份文件 |
| Step 结束 | 运行 `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/mark-step-complete.sh N <workspace>`，推进 `last_completed_step` / `current_step` |
| 任务完成 | 在 4 份文件里做最终总结更新 |

## 2-Action 规则（研究捕获）

**每做 2 次 {Read / Grep / Glob / SearchCodebase} 操作后**，立即更新 `findings.md`。目的是防止上下文压缩/重置时研究成果丢失。

| 操作类型 | 需捕获的内容 | 写入 `findings.md` 的小节 |
|---|---|---|
| SGLang 模型注册表检索 | 相似架构、可复用的候选模型 | 研究发现 |
| NPU 规格查询 | 显存上限、算子支持 | 技术约束 |
| 模型 config.json 阅读 | 架构类型、hidden_size、num_layers | 模型分析 |
| 架构分析师输出解析 | JSON 结果（similarity、parallel config） | 研究发现 + `output_summary.json` |

**为什么**：架构分析师会在代码库里做大量检索。10+ 个参考模型看完后，如果不立即落盘，上下文一重置就全部丢失。

## 相关脚本

| 脚本 | 作用 |
|---|---|
| `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/init-adapter-session.sh` | Step 0 初始化所有规划文件 |
| `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/mark-step-complete.sh` | 每个 Step 结束后推进 `last_completed_step` / `current_step` |
| `${TRAE_SKILLS_PATH}/sglang-npu-adapter/scripts/pre-step-check.sh` | Step 2/5/6 前置校验，依赖 `last_completed_step` |
