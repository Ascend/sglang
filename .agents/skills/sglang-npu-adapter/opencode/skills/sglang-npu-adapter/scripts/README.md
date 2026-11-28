# sglang-npu-adapter / scripts

主流程脚本索引与用法。所有脚本都是**幂等**或**可重入**的——会话中断后续跑安全。

## 总览

| 脚本 | 调用时机 | 作用 |
|------|---------|------|
| `init-adapter-session.sh` | Step 0 | 在 WORKSPACE_DIR 下创建所有状态/规划文件 |
| `check_environment.py` | Step 1 | 完整环境审计(Python/torch/torch_npu/transformers 版本、NPU 数量等) |
| `probe_model_arch.py` | Step 1 / 6.5 入口 | 从 model_path/config.json 提取关键架构字段 |
| `pre-step-check.sh` | Step 2/5/5.5/5.6/6/6.5 之前 | 校验前置条件(规划文件、上一步状态、所需输入文件) |
| `build-agent-query.sh` | 任何子 agent 调用前 | 构建带 P0 前置阅读 PREAMBLE 的完整 query |
| `mark-step-complete.sh` | 每个 Step 之后 | 更新 last_completed_step / current_step;校验三份规划文件已更新 |
| `run_tests.py` | Step 5 (Stage A/B) | 推理冒烟测试(自动读取 /v1/models 的 model id) |
| `audit_model_parity.py` | Step 2.5 | 对比同族参考实现的类/方法/import/设备分支差异,防止 `direct_use` 误判 |
| `triton_op_replace.py` | Step 5.5 | native 算子 → sglang Triton 版;支持 --dry-run / 编译校验 / --rollback |
| `gsm8k_accuracy_test.py` | Step 5.6 | GSM8K 精度测试(本机并发或 docker 内执行) |
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
- Step 6.5 支持独立入口:设置 `adapter_state.json.precision_only=true` 后,允许在完成 Step 1-2 时直接进入(不再要求 `last_completed_step >= 6`)

### audit_model_parity.py

```bash
python {{SKILL_DIR}}/scripts/audit_model_parity.py \
    --model-file <sglang_python_dir>/<reference_file> \
    --models-dir <sglang_python_dir>/sglang/srt/models \
    --output <ws>/output/reference_audit.json
```

- 只对比目标模型文件与同族参考实现的结构差异，不预设具体算子名：
  - 方法 / 类：reference 有、target 没有
  - import：reference 有、target 没有
  - 设备/后端相关分支：`if` 条件里含 npu/cuda/device/backend/platform 等通用词
- `verdict`:`delta` / `parity` / `no_reference`
- `verdict=delta` 时禁止只凭 config 判 `direct_use`；把 `device_*_only_in_reference` 交给 agent 人工判断
- 可选 `--reference <file>` 显式指定参考实现；`--fail-on-delta` 让 CI 在 delta 时退出 1
- 差异是候选，不是“必须全部照抄”

### mark-step-complete.sh

```bash
bash {{SKILL_DIR}}/scripts/mark-step-complete.sh <step_num> <workspace_dir>
```

- **硬性门禁**:三份规划文件(task_plan.md / progress.md / findings.md)自上一步快照以来必须有 sha256 变化,否则拒绝标 complete
- **上一步快照取自 `adapter_state.json.last_completed_step`**,因此 `5.5 → 5.6` 会与 5.5 比较(不会绕过 5.6 的文档门禁)
- 支持小数 step:`5.5` / `5.6` / `6.5`;`current_step` 按 `0→1→…→5→5.5→5.6→6→6.5→7→8` 推进

### run_tests.py

```bash
python {{SKILL_DIR}}/scripts/run_tests.py --port 8000 --wait 300 --mode quick --output <out.json>
```

- `--mode quick`:单条 chat completion(< 10 tokens)
- `--mode full`:chat + completions 两种,断言响应非空
- `--model`:可选;默认从 `/v1/models` 读取真实 model id
- 退出码:0 通过 / 1 失败 / 2 不可达
- 输出 JSON:`{"status": "passed|failed", "phases": {...}, ...}`

### triton_op_replace.py

```bash
# 预览
python {{SKILL_DIR}}/scripts/triton_op_replace.py \
    --sglang-dir <sglang/python> --sgl-kernel-npu-dir <sgl_kernel_npu> \
    --workspace-dir <ws> --dry-run

# 执行
python {{SKILL_DIR}}/scripts/triton_op_replace.py \
    --sglang-dir <sglang/python> --sgl-kernel-npu-dir <sgl_kernel_npu> \
    --workspace-dir <ws>

# 回滚(只需 npu 包目录)
python {{SKILL_DIR}}/scripts/triton_op_replace.py \
    --sgl-kernel-npu-dir <sgl_kernel_npu> --rollback
```

- 匹配要求:文件名命中 **且** 与 native 文件至少共享一个顶层函数/类名,避免误替换 `norm.py` / `rope.py`
- 首次替换前备份 `<file>.native_backup`;已有备份不覆盖
- 替换后 `py_compile` 校验,失败自动回滚该文件
- 清单:`<npu_pkg>/.triton_replace_manifest.json`,`--rollback` 据此恢复并清理复制过来的依赖
- `--mapping custom.json`:补充 `{native_filename: triton_filename}` 映射
- 不再修改 sglang 导入路径;不需要 `.pre_triton_backup`

### gsm8k_accuracy_test.py

```bash
python {{SKILL_DIR}}/scripts/gsm8k_accuracy_test.py \
    --port 8000 --workspace-dir <ws> \
    --data-path <ws>/data/gsm8k.jsonl \
    --num-questions 10 --parallel 4 --min-accuracy auto
```

- `--min-accuracy auto`:参数量 >=5B 用 0.8,<5B 用 0.3(读 `output_summary.json.params_b`)
- 默认本机直接请求 `--port`;需要容器内执行时再加 `--docker-container <name>`
- `--parallel` 仅本机模式生效;docker 模式串行
- `--model-name`:可选;默认从 `/v1/models` 读取
- 数据路径默认 `adapter_state.paths.gsm8k_data_path` → `<ws>/data/gsm8k.jsonl`

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
