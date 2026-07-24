# sglang-npu-adapter / scripts

主流程脚本索引与用法。所有脚本都是**幂等**或**可重入**的——会话中断后续跑安全。

## 总览

| 脚本 | 调用时机 | 作用 |
|------|---------|------|
| `init-adapter-session.sh` | Step 0 | 在 WORKSPACE_DIR 下创建所有状态/规划文件 |
| `check_environment.py` | Step 1 | 完整环境审计(Python/torch/torch_npu/transformers 版本、NPU 数量等) |
| `probe_model_arch.py` | Step 1 / 6.5 入口 | 从 model_path/config.json 提取关键架构字段 |
| `pre-step-check.sh` | Step 2/5/6/6.5 之前 | 校验前置条件(规划文件、上一步状态、所需输入文件) |
| `build-agent-query.sh` | 任何子 agent 调用前 | 构建带 P0 前置阅读 PREAMBLE 的完整 query |
| `mark-step-complete.sh` | 每个 Step 之后 | 更新 last_completed_step / current_step;校验三份规划文件已更新 |
| `run_tests.py` | Step 5 (Stage A/B) | 推理冒烟测试 |
| `post-error-check.sh` | Step 5 (出错时,debug-engineer 返回后) | 校验 fix_instructions.json status |
| `check-step-complete.sh` | Step 8 之前 | 质量门禁(所有产物齐全 + 状态合法) |
| `generate_report.py` | Step 7 | 聚合产物生成最终中文教程 |
| `dump_hf_layer_outputs.py` | precision-rca | HF NPU eager 模型逐层 dump |
| `dump_sgl_layer_outputs.py` | precision-rca | SGLang 模型逐层 dump |
| `find_first_bad_module.py` | precision-rca | 二分比对 HF vs SGLang 层级输出,定位首坏模块 |

## 通用约定

- **所有路径用绝对路径**(WORKSPACE_DIR / SKILL_DIR 都是绝对的)
- **失败用非零退出码**——主流程通过 exit code 决定下一步动作
- **stderr 给人看,stdout 留给数据**——脚本输出 JSON 时用 stdout,提示性文本走 stderr

## 详细用法

### init-adapter-session.sh

```bash
bash {{SKILL_DIR}}/scripts/init-adapter-session.sh <workspace_dir> <model_name> <model_path>
```

- 幂等:已存在文件跳过
- 会自动把 SKILL_DIR 写入 adapter_state.json.skill_dir(后续脚本读取)

### build-agent-query.sh

```bash
bash {{SKILL_DIR}}/scripts/build-agent-query.sh <agent_name> <workspace_dir> [--output <file>]
```

- `<agent_name>`:`architecture_analyst` | `debug_engineer` | `test_validator` | `precision_rca`(下划线版本)
- 输出:渲染好的 query 文本(stdout 或 --output 指定的文件)
- 副作用:`logs/agent_calls/<agent>_<ts>.txt` 快照 + `logs/agent_calls/index.jsonl` 一行索引

### pre-step-check.sh

```bash
bash {{SKILL_DIR}}/scripts/pre-step-check.sh <step_num> <workspace_dir>
```

- 退出码:0 通过 / 1 失败 / 2 跳过(仅 Step 6.5,precision_suspect=false 时)

### mark-step-complete.sh

```bash
bash {{SKILL_DIR}}/scripts/mark-step-complete.sh <step_num> <workspace_dir>
```

- **硬性门禁**:三份规划文件(task_plan.md / progress.md / findings.md)自上一步快照以来必须有 sha256 变化,否则拒绝标 complete
- 支持小数 step:`6.5` 也合法(precision-rca 后)

### run_tests.py

```bash
python {{SKILL_DIR}}/scripts/run_tests.py --port 8000 --wait 300 --mode quick --output <out.json>
```

- `--mode quick`:单条 chat completion(< 10 tokens)
- `--mode full`:chat + completions 两种,断言响应非空
- 退出码:0 通过 / 1 失败 / 2 不可达
- 输出 JSON:`{"status": "passed|failed", "phases": {...}, ...}`

### post-error-check.sh

```bash
bash {{SKILL_DIR}}/scripts/post-error-check.sh <workspace_dir>
```

- 退出码:0 修复可用 / 1 缺产物 / 2 需要人介入(status=needs_human|inconclusive)

### check-step-complete.sh

```bash
bash {{SKILL_DIR}}/scripts/check-step-complete.sh <workspace_dir>
```

- Step 8 交接前的最终门禁。所有交付物齐全才返回 0。

### generate_report.py

```bash
python {{SKILL_DIR}}/scripts/generate_report.py --workspace <ws> --model <name> --output <out.md>
```

- 聚合 adapter_state.json / output_summary.json / test_result.json / root_cause.json(若有)生成中文教程。

## 调试技巧

- 看一次 agent 调用的完整 query:`cat logs/agent_calls/<agent>_<ts>.txt`
- 看 agent 调用历史:`tail logs/agent_calls/index.jsonl | jq .`
- 重置某一步:手动改 `adapter_state.json.last_completed_step` 然后重跑(**不推荐**,会绕过 mark-step-complete 的 sha 校验)
