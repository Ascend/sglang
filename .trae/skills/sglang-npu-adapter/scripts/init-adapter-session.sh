#!/bin/bash
# 为 sglang-npu-adapter 模型适配任务初始化规划文件
# 用法: ./init-adapter-session.sh <workspace_dir> <model_name> <model_path>

set -e

WORKSPACE_DIR="${1:-.trae/workspace/default}"
MODEL_NAME="${2:-UnknownModel}"
MODEL_PATH="${3:-/path/to/model}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATES_DIR="${SKILL_DIR}/templates"

DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

echo "[sglang-npu-adapter] 正在为模型初始化会话: ${MODEL_NAME}"
echo "工作目录: ${WORKSPACE_DIR}"

mkdir -p "${WORKSPACE_DIR}/input"
mkdir -p "${WORKSPACE_DIR}/output"
mkdir -p "${WORKSPACE_DIR}/logs"

if [ ! -d "${TEMPLATES_DIR}" ]; then
    echo "[sglang-npu-adapter] 错误: 未找到模板目录: ${TEMPLATES_DIR}"
    exit 1
fi

init_file() {
    local template_file="${1}"
    local output_file="${2}"
    local description="${3}"

    if [ -f "${output_file}" ]; then
        echo "[sglang-npu-adapter] ${description} 已存在，跳过"
        return 0
    fi

    if [ ! -f "${template_file}" ]; then
        echo "[sglang-npu-adapter] 警告: 未找到模板: ${template_file}"
        return 1
    fi

    MODEL_NAME="${MODEL_NAME}" \
    MODEL_PATH="${MODEL_PATH}" \
    WORKSPACE_DIR="${WORKSPACE_DIR}" \
    python3 - "${template_file}" "${output_file}" <<'PYEOF'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()
content = (content
    .replace("<ModelName>", os.environ["MODEL_NAME"])
    .replace("<ModelPath>", os.environ["MODEL_PATH"])
    .replace("{WORKSPACE_DIR}", os.environ["WORKSPACE_DIR"]))
with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)
PYEOF

    echo "[sglang-npu-adapter] 已创建 ${description}"
}

init_file "${TEMPLATES_DIR}/task_plan.md" "${WORKSPACE_DIR}/task_plan.md" "task_plan.md"
init_file "${TEMPLATES_DIR}/findings.md" "${WORKSPACE_DIR}/findings.md" "findings.md"
init_file "${TEMPLATES_DIR}/progress.md" "${WORKSPACE_DIR}/progress.md" "progress.md"

if [ ! -f "${WORKSPACE_DIR}/adapter_state.json" ]; then
    # 记录初始化时的 git HEAD，作为 generate_report 计算"本任务实际改动"的基线 commit
    BASE_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "")
    BASE_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    
    # 解析执行环境限制（如果提供）
    if [ -n "${EXECUTION_ENV}" ]; then
        # 验证JSON格式并提取字段
        EXEC_CONTAINER=$(echo "${EXECUTION_ENV}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('container',''))" 2>/dev/null || echo "")
        EXEC_WRAPPER=$(echo "${EXECUTION_ENV}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('command_wrapper',''))" 2>/dev/null || echo "")
        EXEC_TYPE=$(echo "${EXECUTION_ENV}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('type','docker_container'))" 2>/dev/null || echo "docker_container")
    else
        EXEC_CONTAINER=""
        EXEC_WRAPPER=""
        EXEC_TYPE="local"
    fi
    
    MODEL_NAME="${MODEL_NAME}" \
    MODEL_PATH="${MODEL_PATH}" \
    WORKSPACE_DIR="${WORKSPACE_DIR}" \
    SKILL_DIR="${SKILL_DIR}" \
    DATE="${DATE}" \
    TIMESTAMP="${TIMESTAMP}" \
    BASE_COMMIT="${BASE_COMMIT}" \
    BASE_BRANCH="${BASE_BRANCH}" \
    EXEC_CONTAINER="${EXEC_CONTAINER}" \
    EXEC_WRAPPER="${EXEC_WRAPPER}" \
    EXEC_TYPE="${EXEC_TYPE}" \
    python3 - "${WORKSPACE_DIR}/adapter_state.json" <<'PYEOF'
import json, os, sys
state = {
    "task_id": f"{os.environ['MODEL_NAME']}_{os.environ['DATE']}",
    "model_name": os.environ["MODEL_NAME"],
    "model_path": os.environ["MODEL_PATH"],
    "workspace_dir": os.environ["WORKSPACE_DIR"],
    "skill_dir": os.environ["SKILL_DIR"],
    "target_device": "npu",
    "base_commit": os.environ.get("BASE_COMMIT") or None,
    "base_branch": os.environ.get("BASE_BRANCH") or None,
    "current_step": 0,
    "current_phase": 1,
    "last_completed_step": None,
    "iteration_count": 0,
    "max_iterations": 20,
    "next_action": "proceed",
    "last_update": os.environ["TIMESTAMP"],
    "execution_environment": {
        "type": os.environ.get("EXEC_TYPE", "local"),
        "container": os.environ.get("EXEC_CONTAINER") or None,
        "command_wrapper": os.environ.get("EXEC_WRAPPER") or None,
    },
    "agent_outputs": {
        "architecture_analyst_output": None,
        "debug_engineer_outputs": [],
        "test_validator_output": None,
    },
    "validation": {
        "dummy_passed": False,
        "real_weight_passed": False,
        "dummy_inference_log": None,
        "real_inference_log": None,
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(state, f, indent=4, ensure_ascii=False)
PYEOF
    echo "[sglang-npu-adapter] 已创建 adapter_state.json"
else
    echo "[sglang-npu-adapter] adapter_state.json 已存在，跳过"
fi

if [ ! -f "${WORKSPACE_DIR}/input/input_params.json" ]; then
    MODEL_NAME="${MODEL_NAME}" \
    MODEL_PATH="${MODEL_PATH}" \
    WORKSPACE_DIR="${WORKSPACE_DIR}" \
    python3 - "${WORKSPACE_DIR}/input/input_params.json" <<'PYEOF'
import json, os, sys
params = {
    "model_path": os.environ["MODEL_PATH"],
    "model_name": os.environ["MODEL_NAME"],
    "target_device": "npu",
    "workspace_dir": os.environ["WORKSPACE_DIR"],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)
PYEOF
    echo "[sglang-npu-adapter] 已创建 input/input_params.json"
fi

echo ""
echo "[sglang-npu-adapter] === 初始化完成 ==="
echo "已创建文件:"
echo "  - ${WORKSPACE_DIR}/task_plan.md"
echo "  - ${WORKSPACE_DIR}/findings.md"
echo "  - ${WORKSPACE_DIR}/progress.md"
echo "  - ${WORKSPACE_DIR}/adapter_state.json"
echo "  - ${WORKSPACE_DIR}/input/input_params.json"
echo ""
echo "已创建目录:"
echo "  - ${WORKSPACE_DIR}/input/"
echo "  - ${WORKSPACE_DIR}/output/"
echo "  - ${WORKSPACE_DIR}/logs/"
echo ""
echo "下一步: 进入 Step 1（收集上下文）"
