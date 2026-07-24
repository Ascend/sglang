#!/bin/bash
# 初始化一个 sglang-npu-adapter 会话:在 WORKSPACE_DIR 下创建所有必要文件
# 幂等——已存在的文件会跳过,因此续跑安全。
#
# 用法: ./init-adapter-session.sh <workspace_dir> <model_name> <model_path>

set -e

WORKSPACE_DIR="${1}"
MODEL_NAME="${2:-unknown}"
MODEL_PATH="${3:-/path/to/model}"

if [ -z "${WORKSPACE_DIR}" ]; then
    echo "用法: $0 <workspace_dir> <model_name> <model_path>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[init] workspace = ${WORKSPACE_DIR}"
echo "[init] skill_dir = ${SKILL_DIR}"
echo "[init] model     = ${MODEL_NAME} @ ${MODEL_PATH}"

mkdir -p "${WORKSPACE_DIR}/input" \
         "${WORKSPACE_DIR}/output" \
         "${WORKSPACE_DIR}/logs" \
         "${WORKSPACE_DIR}/logs/agent_calls"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

# adapter_state.json（机器可读状态)
STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"
if [ ! -f "${STATE_FILE}" ]; then
    cat > "${STATE_FILE}" <<EOF
{
    "skill_dir": "${SKILL_DIR}",
    "workspace_dir": "${WORKSPACE_DIR}",
    "model_name": "${MODEL_NAME}",
    "model_path": "${MODEL_PATH}",
    "current_step": 1,
    "last_completed_step": 0,
    "iteration_count": 0,
    "max_iterations": 20,
    "next_action": "collect_context",
    "precision_suspect": false,
    "adapter_strategy": null,
    "requires_new_adapter": null,
    "validation": {
        "dummy_passed": false,
        "real_weight_passed": false
    },
    "step_snapshots": {},
    "created_at": "${TIMESTAMP}",
    "last_update": "${TIMESTAMP}"
}
EOF
    echo "[init] 写入 ${STATE_FILE}"
else
    echo "[init] 跳过 ${STATE_FILE}(已存在)"
    # 修补缺失的 skill_dir
    python3 - <<PYEOF
import json, os
with open("${STATE_FILE}") as f:
    s = json.load(f)
changed = False
if not s.get("skill_dir"):
    s["skill_dir"] = "${SKILL_DIR}"; changed = True
if not s.get("workspace_dir"):
    s["workspace_dir"] = "${WORKSPACE_DIR}"; changed = True
if changed:
    with open("${STATE_FILE}", "w") as f:
        json.dump(s, f, indent=4, ensure_ascii=False)
    print("[init] 已修补缺失的 skill_dir/workspace_dir")
PYEOF
fi

# task_plan.md
PLAN_FILE="${WORKSPACE_DIR}/task_plan.md"
if [ ! -f "${PLAN_FILE}" ]; then
    cat > "${PLAN_FILE}" <<EOF
# 适配任务计划: ${MODEL_NAME}

> 创建时间: ${TIMESTAMP}
> 工作目录: ${WORKSPACE_DIR}
> Skill: sglang-npu-adapter

## 阶段进度

- [x] Step 0 - 初始化 (本次)
- [ ] Step 1 - 收集上下文(环境审计 + 设备信息)
- [ ] Step 2 - 调用 architecture-analyst
- [ ] Step 3 - 选择适配策略
- [ ] Step 4 - 实施代码修改(若需要)
- [ ] Step 5 - 两阶段验证(Dummy + Real Weight)
- [ ] Step 6 - 调用 test-validator(功能集 + 容量基线)
- [ ] Step 6.5 - 精度根因定位(可选,触发式)
- [ ] Step 7 - 生成产物
- [ ] Step 8 - 交接

## 决策日志

(每个 Step 完成后追加一条:决策/调整理由)

EOF
    echo "[init] 写入 ${PLAN_FILE}"
fi

# findings.md
FIND_FILE="${WORKSPACE_DIR}/findings.md"
if [ ! -f "${FIND_FILE}" ]; then
    cat > "${FIND_FILE}" <<EOF
# 研究发现: ${MODEL_NAME}

> 2-Action 规则:每做 2 次 {Read/Grep/Glob/Bash} 必须更新这里。
> 这是跨 Step 持久化的短期记忆——上下文被压缩后,主流程通过 Read 这里恢复执行。

## 上下文摘要(每次更新放最新版,旧的下沉到下方)

(待填)

## 历史发现

### Step 0 - 初始化
- workspace: ${WORKSPACE_DIR}
- skill_dir: ${SKILL_DIR}
- model: ${MODEL_NAME} @ ${MODEL_PATH}

EOF
    echo "[init] 写入 ${FIND_FILE}"
fi

# progress.md
PROG_FILE="${WORKSPACE_DIR}/progress.md"
if [ ! -f "${PROG_FILE}" ]; then
    cat > "${PROG_FILE}" <<EOF
# 执行日志: ${MODEL_NAME}

## Step 0 - 初始化 @ ${TIMESTAMP}

- 创建目录: input/, output/, logs/, logs/agent_calls/
- 创建文件: adapter_state.json, task_plan.md, findings.md, progress.md, input/input_params.json

EOF
    echo "[init] 写入 ${PROG_FILE}"
fi

# input/input_params.json(架构分析师输入骨架)
PARAMS_FILE="${WORKSPACE_DIR}/input/input_params.json"
if [ ! -f "${PARAMS_FILE}" ]; then
    cat > "${PARAMS_FILE}" <<EOF
{
    "model_name": "${MODEL_NAME}",
    "model_path": "${MODEL_PATH}",
    "target_device": "npu",
    "special_requirements": []
}
EOF
    echo "[init] 写入 ${PARAMS_FILE}"
fi

echo ""
echo "[init] 完成。下一步:执行 Step 1(收集上下文)"
echo "  bash ${SKILL_DIR}/scripts/pre-step-check.sh 2 ${WORKSPACE_DIR}  # Step 1 之后调用"
