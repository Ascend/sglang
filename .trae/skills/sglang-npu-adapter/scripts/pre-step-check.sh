#!/bin/bash
# sglang-npu-adapter 关键节点的步骤前校验
# 在执行关键步骤前确认前置条件已满足
# 用法: ./pre-step-check.sh <step_num> <workspace_dir>

set -e

STEP_NUM="${1:-0}"
WORKSPACE_DIR="${2:-.trae/workspace/default}"
STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"

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

echo "[sglang-npu-adapter] Pre-Step ${STEP_NUM} 检查..."
echo "工作目录: ${WORKSPACE_DIR}"

if [ ! -d "${WORKSPACE_DIR}" ]; then
    echo "错误: 未找到 workspace 目录"
    exit 1
fi

if [ ! -f "${STATE_FILE}" ]; then
    echo "错误: 未找到 adapter_state.json"
    echo "请先运行 init-adapter-session.sh"
    exit 1
fi

case "${STEP_NUM}" in
    2)
        echo ""
        echo "=== Pre-Step 2: 调用 architecture-analyst 之前 ==="
        echo ""

        echo "检查 1: device_info.json 已存在（architecture-analyst 输入）"
        if [ ! -f "${WORKSPACE_DIR}/input/device_info.json" ]; then
            echo "  ✗ 失败: 未找到 device_info.json"
            echo "  原因: architecture-analyst 推导并行配置时需要设备信息"
            echo "  操作: 请先完成 Step 1（收集上下文）"
            exit 1
        fi
        echo "  ✓ 通过: device_info.json 已存在"

        echo "检查 2: input_params.json 已存在（architecture-analyst 输入）"
        if [ ! -f "${WORKSPACE_DIR}/input/input_params.json" ]; then
            echo "  ✗ 失败: 未找到 input_params.json"
            echo "  原因: architecture-analyst 需要模型路径与参数"
            echo "  操作: 请先完成 Step 1（收集上下文）"
            exit 1
        fi
        echo "  ✓ 通过: input_params.json 已存在"

        echo "检查 3: last_completed_step >= 1"
        LAST_STEP=$(parse_json_number "last_completed_step" "${STATE_FILE}")
        if [ "${LAST_STEP}" -lt 1 ]; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 调用 architecture-analyst 前必须完成 Step 1"
            echo "  操作: 执行 Step 1 并更新 adapter_state.json"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 2 检查通过"
        echo "下一步: 阅读 ${SKILL_DIR}/prompts/model_analyzer.md，调用 architecture-analyst"
        ;;

    5)
        echo ""
        echo "=== Pre-Step 5: 进入验证阶段之前 ==="
        echo ""

        echo "检查 1: last_completed_step >= 4"
        LAST_STEP=$(parse_json_number "last_completed_step" "${STATE_FILE}")
        if [ "${LAST_STEP}" -lt 4 ]; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 进入验证前必须完成代码修改（Step 4）"
            echo "  操作: 执行 Step 4 并更新 adapter_state.json"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo "检查 2: iteration_count 已重置或被正确跟踪"
        ITERATION=$(parse_json_number "iteration_count" "${STATE_FILE}")
        if [ "${ITERATION}" -gt 0 ]; then
            echo "  ⚠ 警告: iteration_count = ${ITERATION}"
            echo "  说明: 这可能是带有历史调试记录的恢复会话"
            echo "  操作: 若为新任务，请将 iteration_count 重置为 0"
        else
            echo "  ✓ 通过: iteration_count = 0（干净起点）"
        fi

        echo "检查 3: 不存在待处理的 debug-engineer 调用"
        NEXT_ACTION=$(parse_json_string "next_action" "${STATE_FILE}")
        if [ "${NEXT_ACTION}" = "call_debug_engineer" ]; then
            echo "  ✗ 失败: next_action = 'call_debug_engineer'"
            echo "  原因: 上一次错误要求调用 debug-engineer，但尚未执行"
            echo "  违规: Main Agent 试图跳过 debug-engineer 进行调试"
            echo "  操作: 调用 debug-engineer 处理该错误"
            exit 1
        fi
        echo "  ✓ 通过: next_action = ${NEXT_ACTION}"

        echo "检查 4: output_summary.json 已存在（依赖 architecture-analyst 输出）"
        if [ ! -f "${WORKSPACE_DIR}/output/output_summary.json" ]; then
            echo "  ✗ 失败: 未找到 output_summary.json"
            echo "  原因: 验证需要 architecture-analyst 给出的并行配置"
            echo "  操作: 确认 Step 2 已正确调用 architecture-analyst"
            exit 1
        fi
        echo "  ✓ 通过: output_summary.json 已存在"

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 5 检查通过"
        echo "下一步: 先 Dummy 验证 → 再真实权重验证"
        ;;

    6)
        echo ""
        echo "=== Pre-Step 6: 调用 test-validator 之前 ==="
        echo ""

        echo "检查 1: 验证已通过（next_action != call_debug_engineer）"
        NEXT_ACTION=$(parse_json_string "next_action" "${STATE_FILE}")
        if [ "${NEXT_ACTION}" = "call_debug_engineer" ]; then
            echo "  ✗ 失败: next_action = 'call_debug_engineer'"
            echo "  原因: Step 5 验证仍在进行中（需要继续调试）"
            echo "  违规: 调试未完成就试图跳到 test-validator"
            echo "  操作: 先完成 Step 5 的调试迭代"
            exit 1
        fi
        echo "  ✓ 通过: next_action = ${NEXT_ACTION}"

        echo "检查 2: last_completed_step >= 5"
        LAST_STEP=$(parse_json_number "last_completed_step" "${STATE_FILE}")
        if [ "${LAST_STEP}" -lt 5 ]; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 调用 test-validator 前 Step 5 必须验证通过"
            echo "  操作: 完成 Step 5 验证后再继续"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo "检查 3: test_config.json 已就绪"
        if [ ! -f "${WORKSPACE_DIR}/input/test_config.json" ]; then
            echo "  ✗ 失败: 未找到 test_config.json"
            echo "  原因: test-validator 需要测试配置文件"
            echo "  操作: 调用 test-validator 前先准备 test_config.json"
            exit 1
        fi
        echo "  ✓ 通过: test_config.json 已存在"

        echo "检查 4: iteration_count <= max_iterations (20)"
        ITERATION=$(parse_json_number "iteration_count" "${STATE_FILE}")
        if [ "${ITERATION}" -gt 20 ]; then
            echo "  ✗ 失败: iteration_count = ${ITERATION} > 20"
            echo "  原因: 已超出最大调试迭代次数"
            echo "  操作: 上报用户，请其给出后续指导"
            exit 1
        fi
        echo "  ✓ 通过: iteration_count = ${ITERATION}/20"

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 6 检查通过"
        echo "下一步: 阅读 ${SKILL_DIR}/prompts/test_validator.md，调用 test-validator"
        ;;

    *)
        echo ""
        echo "说明: Step ${STEP_NUM} 未定义前置检查"
        echo "Step 0、1、3、4、7、8 不需要强制校验"
        ;;
esac

exit 0
