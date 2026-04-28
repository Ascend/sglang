#!/bin/bash
# 错误后校验：确保 debug-engineer 被正确调用以处理错误
# 防止 Main Agent 自行调试（SKILL.md 硬性约束）
# 用法: ./post-error-check.sh <workspace_dir>

set -e

WORKSPACE_DIR="${1:-.trae/workspace/default}"
STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"
FIX_FILE="${WORKSPACE_DIR}/output/fix_instructions.json"
DEBUG_REPORT="${WORKSPACE_DIR}/output/debug_report.md"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

parse_json_number() {
    local field="${1}"
    local file="${2}"
    local val
    val=$(grep -oE "\"${field}\"[[:space:]]*:[[:space:]]*[0-9]+" "${file}" 2>/dev/null | head -1 | grep -oE '[0-9]+')
    echo "${val:--1}"
}

parse_json_string() {
    local field="${1}"
    local file="${2}"
    local val
    val=$(grep "\"${field}\"" "${file}" 2>/dev/null | head -1 | sed 's/.*"'"${field}"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
    echo "${val}"
}

echo "[sglang-npu-adapter] 错误后校验检查..."
echo "工作目录: ${WORKSPACE_DIR}"

if [ ! -f "${STATE_FILE}" ]; then
    echo "错误: 未找到 adapter_state.json"
    exit 1
fi

CURRENT_STEP=$(parse_json_number "current_step" "${STATE_FILE}")
NEXT_ACTION=$(parse_json_string "next_action" "${STATE_FILE}")
ITERATION=$(parse_json_number "iteration_count" "${STATE_FILE}")

echo ""
echo "=== 当前状态 ==="
echo "current_step: ${CURRENT_STEP}"
echo "next_action: ${NEXT_ACTION}"
echo "iteration_count: ${ITERATION}/20"
echo ""

VIOLATION_FOUND=false

echo "=== 校验项 ==="
echo ""

echo "检查 1: 是否检测到错误（iteration_count > 0 或 next_action = call_debug_engineer）"
if [ "${ITERATION}" -gt 0 ] || [ "${NEXT_ACTION}" = "call_debug_engineer" ]; then
    echo "  ✓ 检测到错误场景"
    echo "    iteration_count: ${ITERATION}"
    echo "    next_action: ${NEXT_ACTION}"

    echo ""
    echo "检查 2: fix_instructions.json 已存在（debug-engineer 输出）"
    if [ ! -f "${FIX_FILE}" ]; then
        echo "  ✗ 违规: 未找到 fix_instructions.json"
        echo "    原因: Main Agent 试图在未调用 debug-engineer 的情况下调试"
        echo "    约束: '遇到任何错误都必须先调用 debug-engineer'"
        echo "    SKILL.md: '主技能禁止自行调试'"
        VIOLATION_FOUND=true
    else
        echo "  ✓ 通过: fix_instructions.json 已存在"

        echo ""
        echo "检查 3: fix_instructions.json 包含合法的 debug-engineer 状态"
        FIX_STATUS=$(parse_json_string "status" "${FIX_FILE}")
        if [ "${FIX_STATUS}" != "fix_available" ] && [ "${FIX_STATUS}" != "fix_verified" ]; then
            echo "  ✗ 违规: 非法 fix 状态: '${FIX_STATUS}'"
            echo "    预期: 'fix_available' 或 'fix_verified'"
            echo "    原因: 该输出可能不是合法的 debug-engineer 产物"
            VIOLATION_FOUND=true
        else
            echo "  ✓ 通过: status = '${FIX_STATUS}'"

            echo ""
            echo "检查 4: 修复方案包含 debug-engineer 给出的诊断"
            DIAGNOSIS=$(grep -c '"diagnosis"' "${FIX_FILE}" 2>/dev/null || echo "0")
            if [ "${DIAGNOSIS}" -eq 0 ]; then
                echo "  ⚠ 警告: fix_instructions.json 中缺少 diagnosis 字段"
                echo "    debug-engineer 应当提供结构化诊断"
            else
                echo "  ✓ 通过: diagnosis 字段已存在"
            fi

            echo ""
            echo "检查 5: 修复方案包含可执行步骤"
            STEPS=$(grep -c '"steps"' "${FIX_FILE}" 2>/dev/null || echo "0")
            if [ "${STEPS}" -eq 0 ]; then
                echo "  ⚠ 警告: fix_instructions.json 中缺少 steps 字段"
            else
                echo "  ✓ 通过: steps 字段已存在"
                FIX_TYPE=$(parse_json_string "type" "${FIX_FILE}")
                echo "    fix_type: ${FIX_TYPE}"
            fi
        fi
    fi

    echo ""
    echo "检查 6: debug_report.md 已存在（debug-engineer 报告）"
    if [ ! -f "${DEBUG_REPORT}" ]; then
        echo "  ⚠ 警告: 未找到 debug_report.md"
        echo "    debug-engineer 应当生成详细的调试报告"
    else
        echo "  ✓ 通过: debug_report.md 已存在"
    fi

else
    echo "  ✓ 未检测到错误场景"
    echo "    属于正常执行路径"
fi

echo ""

if [ "${NEXT_ACTION}" = "call_debug_engineer" ]; then
    echo "=== 必须调用 debug-engineer ==="
    echo ""
    echo "next_action 为 'call_debug_engineer'"
    echo "Main Agent 现在必须调用 debug-engineer。"
    echo ""
    echo "必做事项:"
    echo "  1. 输出复诵: '错误发生，准备调用 Debug 工程师'"
    echo "  2. 阅读: ${SKILL_DIR}/prompts/debug_engineer.md"
    echo "  3. 创建: input/error_context.json"
    echo "  4. 通过 Task 调用 subagent_type='debug-engineer'"
    echo "  5. 等待 debug-engineer 输出"
    echo "  6. 解析: output/fix_instructions.json"
    echo ""
    echo "禁止事项:"
    echo "  - Main Agent 自行修复错误"
    echo "  - 在没有 debug-engineer 指引的情况下修改代码"
    echo "  - 跳过 debug-engineer 调用"
fi

echo ""

if [ "${VIOLATION_FOUND}" = true ]; then
    echo "=== 校验失败 ==="
    echo ""
    echo "[sglang-npu-adapter] 检测到违规"
    echo "Main Agent 试图绕过 debug-engineer 直接调试"
    echo ""
    echo "要求: 任何修复尝试前必须先调用 debug-engineer"
    echo ""
    echo "正确流程:"
    echo "  错误 → adapter_state.json (next_action: call_debug_engineer)"
    echo "       → 调用 debug-engineer → fix_instructions.json"
    echo "       → 应用修复 → 重新验证"
    echo ""
    exit 1
else
    echo "=== 校验通过 ==="
    echo ""
    echo "[sglang-npu-adapter] 错误后校验完成"
    if [ "${NEXT_ACTION}" = "call_debug_engineer" ]; then
        echo "需立即操作: 调用 debug-engineer"
    else
        echo "可继续当前任务流程"
    fi
fi

exit 0
