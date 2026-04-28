#!/bin/bash
# 检查模型适配任务的所有步骤（Step 0-8）是否完成
# 同时校验质量门（Quality Gate）条件
# 用法: ./check-step-complete.sh [workspace_dir]

set -e

WORKSPACE_DIR="${1:-.trae/workspace/default}"
PROGRESS_FILE="${WORKSPACE_DIR}/progress.md"
STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"

STEPS=(0 1 2 3 4 5 6 7 8)
TOTAL_STEPS=${#STEPS[@]}

echo "[sglang-npu-adapter] 正在检查完成状态..."
echo "工作目录: ${WORKSPACE_DIR}"

if [ ! -d "${WORKSPACE_DIR}" ]; then
    echo "[sglang-npu-adapter] 错误: 未找到 workspace 目录"
    exit 1
fi

if [ ! -f "${PROGRESS_FILE}" ]; then
    echo "[sglang-npu-adapter] 未找到 progress.md —— 当前没有进行中的适配会话。"
    exit 0
fi

# 注意：以下 grep 模式（### Step N:、Status: 行、complete/in_progress/pending）
# 必须与 templates/progress.md 写入的英文标识一致，不要翻译。
check_step_status() {
    local step_num="${1}"
    local step_pattern="### Step ${step_num}:"

    if ! grep -q "${step_pattern}" "${PROGRESS_FILE}"; then
        return 2
    fi

    local status_line=$(grep -A 2 "${step_pattern}" "${PROGRESS_FILE}" | grep "\- \*\*Status:\*\*" || true)

    if [ -z "${status_line}" ]; then
        return 3
    fi

    if echo "${status_line}" | grep -qE '\bcomplete\b'; then
        return 0
    elif echo "${status_line}" | grep -qE '\bin_progress\b'; then
        return 1
    elif echo "${status_line}" | grep -qE '\bpending\b'; then
        return 4
    fi

    return 3
}

COMPLETE=0
IN_PROGRESS=0
PENDING=0
MISSING=0
UNKNOWN=0

for step in ${STEPS[@]}; do
    check_step_status "${step}"
    case $? in
        0) COMPLETE=$((COMPLETE + 1)); echo "  Step ${step}: ✅ 已完成" ;;
        1) IN_PROGRESS=$((IN_PROGRESS + 1)); echo "  Step ${step}: 🔄 进行中" ;;
        2) MISSING=$((MISSING + 1)); echo "  Step ${step}: ⚠️  缺失（未初始化）" ;;
        3) UNKNOWN=$((UNKNOWN + 1)); echo "  Step ${step}: ❓ 状态未知" ;;
        4) PENDING=$((PENDING + 1)); echo "  Step ${step}: ⏳ 待处理" ;;
    esac
done

echo ""
echo "=== 步骤状态汇总 ==="
echo "已完成:    ${COMPLETE}/${TOTAL_STEPS}"
echo "进行中:    ${IN_PROGRESS}"
echo "待处理:    ${PENDING}"
echo "缺失:      ${MISSING}"
echo "未知:      ${UNKNOWN}"
echo ""

if [ -f "${STATE_FILE}" ]; then
    echo "=== 适配器状态 ==="

    CURRENT_STEP=$(grep -oE '"current_step":[[:space:]]*[0-9]+' "${STATE_FILE}" | head -1 | grep -oE '[0-9]+')
    CURRENT_STEP="${CURRENT_STEP:-unknown}"
    LAST_COMPLETED=$(grep -oE '"last_completed_step":[[:space:]]*[0-9]+' "${STATE_FILE}" | head -1 | grep -oE '[0-9]+')
    LAST_COMPLETED="${LAST_COMPLETED:-null}"
    NEXT_ACTION=$(grep '"next_action"' "${STATE_FILE}" | head -1 | sed 's/.*"next_action"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    NEXT_ACTION="${NEXT_ACTION:-unknown}"
    ITERATION=$(grep -oE '"iteration_count":[[:space:]]*[0-9]+' "${STATE_FILE}" | head -1 | grep -oE '[0-9]+')
    ITERATION="${ITERATION:-0}"

    echo "  当前 Step:    Step ${CURRENT_STEP}"
    echo "  最后完成:     Step ${LAST_COMPLETED}"
    echo "  下一步动作:   ${NEXT_ACTION}"
    echo "  迭代次数:     ${ITERATION}/20"
    echo ""
fi

QUALITY_GATE_PASSED=true
GATE_CHECKS=()

if [ "${COMPLETE}" -eq "${TOTAL_STEPS}" ]; then
    echo "=== 质量门检查 ==="

    # 注意：grep 的英文短语来自 templates/progress.md 与外部脚本的输出，不要翻译。
    if grep -q "Service started successfully" "${PROGRESS_FILE}" || \
       grep -qE "Server starts.*✅" "${PROGRESS_FILE}"; then
        GATE_CHECKS+=("✓ 服务已启动")
    else
        GATE_CHECKS+=("✗ 服务启动 - 未确认")
        QUALITY_GATE_PASSED=false
    fi

    if grep -q "Inference request succeeded" "${PROGRESS_FILE}" || \
       grep -qE "TC001.*passed|TC002.*passed|TC003.*passed" "${PROGRESS_FILE}"; then
        GATE_CHECKS+=("✓ 推理测试已通过")
    else
        GATE_CHECKS+=("✗ 推理测试 - 未确认")
        QUALITY_GATE_PASSED=false
    fi

    if [ -f "${WORKSPACE_DIR}/output/output_summary.json" ]; then
        GATE_CHECKS+=("✓ architecture-analyst 输出已存在")
    else
        GATE_CHECKS+=("✗ architecture-analyst 输出 - 缺失")
        QUALITY_GATE_PASSED=false
    fi

    if [ -f "${WORKSPACE_DIR}/output/test_result.json" ]; then
        GATE_CHECKS+=("✓ test-validator 输出已存在")
    else
        GATE_CHECKS+=("✗ test-validator 输出 - 缺失")
        QUALITY_GATE_PASSED=false
    fi

    MODEL_DOC=$(find "${WORKSPACE_DIR}" -name "*.md" -type f ! -name "task_plan.md" ! -name "progress.md" ! -name "findings.md" ! -name "analysis_report.md" ! -name "debug_report.md" ! -name "test_report.md" 2>/dev/null | head -1)
    if [ -n "${MODEL_DOC}" ]; then
        GATE_CHECKS+=("✓ 教程文档: ${MODEL_DOC}")
    else
        GATE_CHECKS+=("✗ 教程文档 - 缺失")
        QUALITY_GATE_PASSED=false
    fi

    for check in "${GATE_CHECKS[@]}"; do
        echo "  ${check}"
    done
    echo ""
fi

echo "=== 最终状态 ==="
if [ "${COMPLETE}" -eq "${TOTAL_STEPS}" ] && [ "${TOTAL_STEPS}" -gt 0 ]; then
    if [ "${QUALITY_GATE_PASSED}" = true ]; then
        echo "[sglang-npu-adapter] 全部 Step 已完成 & 质量门通过"
        echo "任务可进入最终交付。"
        echo "下一步: 准备交付物（Step 8）"
    else
        echo "[sglang-npu-adapter] 全部 Step 已完成，但质量门未通过"
        echo "请回顾上方失败项并在交付前确认。"
    fi
elif [ "${IN_PROGRESS}" -gt 0 ]; then
    echo "[sglang-npu-adapter] 任务进行中（${COMPLETE}/${TOTAL_STEPS} Step 已完成）"
    echo "请从当前 Step 继续。停止前请先更新 progress.md。"
else
    echo "[sglang-npu-adapter] 任务未开始或未完成（${COMPLETE}/${TOTAL_STEPS} Step 已完成）"
    if [ "${MISSING}" -gt 0 ]; then
        echo "警告: 部分 Step 未初始化。请先运行 init-adapter-session.sh。"
    fi
fi

exit 0
