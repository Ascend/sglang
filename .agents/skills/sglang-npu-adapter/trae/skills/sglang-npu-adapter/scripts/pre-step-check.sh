#!/bin/bash
# sglang-npu-adapter 关键节点的步骤前校验
# 在执行关键步骤前确认前置条件已满足
# 用法: ./pre-step-check.sh <step_num> <workspace_dir>
#
# 退出码:
#   0 - 通过
#   1 - 前置条件不满足
#   2 - 本步可跳过(目前仅 Step 6.5 在 precision_suspect != true 时)

set -euo pipefail

STEP_NUM="${1:-0}"
WORKSPACE_DIR="${2:-.trae/workspace/default}"
STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 统一 Python 解释器:优先 python3,失败再退到 python
if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "错误: 未找到 python3 或 python" >&2
        exit 1
    fi
fi

# 用 Python 解析 JSON,避免 grep/sed 对布尔值、换行、转义的误判。
json_get() {
    local field="${1}"
    local file="${2}"
    "${PYTHON_BIN}" - "${field}" "${file}" <<'PYEOF'
import json
import sys

field, path = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as f:
        cur = json.load(f)
except Exception:
    sys.exit(0)

for part in field.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        sys.exit(0)

if cur is None:
    print("")
elif isinstance(cur, bool):
    print("true" if cur else "false")
elif isinstance(cur, (dict, list)):
    print(json.dumps(cur, ensure_ascii=False))
else:
    print(cur)
PYEOF
}

# 浮点安全比较:num_ge A B  => A >= B
num_ge() {
    awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { exit !(a + 0 >= b + 0) }'
}

require_eq() {
    local actual="${1}"
    local expected="${2}"
    local label="${3}"
    if [ "${actual}" != "${expected}" ]; then
        echo "  ✗ 失败: ${label} = '${actual}'(期望 '${expected}')"
        return 1
    fi
    echo "  ✓ 通过: ${label} = ${actual}"
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

LAST_STEP="$(json_get last_completed_step "${STATE_FILE}")"
LAST_STEP="${LAST_STEP:-0}"
NEXT_ACTION="$(json_get next_action "${STATE_FILE}")"
ITERATION="$(json_get iteration_count "${STATE_FILE}")"
ITERATION="${ITERATION:-0}"

check_iteration_cap() {
    echo "检查: iteration_count <= 20"
    if ! num_ge 20 "${ITERATION}"; then
        echo "  ✗ 失败: iteration_count = ${ITERATION} > 20"
        echo "  原因: 已超出最大调试迭代次数"
        echo "  操作: 上报用户,请其给出后续指导"
        return 1
    fi
    echo "  ✓ 通过: iteration_count = ${ITERATION}/20"
}

case "${STEP_NUM}" in
    2)
        echo ""
        echo "=== Pre-Step 2: 调用 architecture-analyst 之前 ==="
        echo ""

        echo "检查 1: device_info.json 已存在(architecture-analyst 输入)"
        if [ ! -f "${WORKSPACE_DIR}/input/device_info.json" ]; then
            echo "  ✗ 失败: 未找到 device_info.json"
            echo "  原因: architecture-analyst 推导并行配置时需要设备信息"
            echo "  操作: 请先完成 Step 1(收集上下文)"
            exit 1
        fi
        echo "  ✓ 通过: device_info.json 已存在"

        echo "检查 2: input_params.json 已存在(architecture-analyst 输入)"
        if [ ! -f "${WORKSPACE_DIR}/input/input_params.json" ]; then
            echo "  ✗ 失败: 未找到 input_params.json"
            echo "  原因: architecture-analyst 需要模型路径与参数"
            echo "  操作: 请先完成 Step 1(收集上下文)"
            exit 1
        fi
        echo "  ✓ 通过: input_params.json 已存在"

        echo "检查 3: last_completed_step >= 1"
        if ! num_ge "${LAST_STEP}" 1; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 调用 architecture-analyst 前必须完成 Step 1"
            echo "  操作: 执行 Step 1 并更新 adapter_state.json"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 2 检查通过"
        echo "下一步: 阅读 ${SKILL_DIR}/prompts/model_analyzer.md,调用 architecture-analyst"
        ;;

    5)
        echo ""
        echo "=== Pre-Step 5: 进入验证阶段之前 ==="
        echo ""

        echo "检查 1: last_completed_step >= 4"
        if ! num_ge "${LAST_STEP}" 4; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 进入验证前必须完成代码修改(Step 4)"
            echo "  操作: 执行 Step 4 并更新 adapter_state.json"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo "检查 2: 不存在待处理的 debug-engineer 调用"
        if [ "${NEXT_ACTION}" = "call_debug_engineer" ]; then
            echo "  ✗ 失败: next_action = 'call_debug_engineer'"
            echo "  原因: 上一次错误要求调用 debug-engineer,但尚未执行"
            echo "  违规: Main Agent 试图跳过 debug-engineer 进行调试"
            echo "  操作: 调用 debug-engineer 处理该错误"
            exit 1
        fi
        echo "  ✓ 通过: next_action = ${NEXT_ACTION}"

        echo "检查 3: output_summary.json 已存在(依赖 architecture-analyst 输出)"
        if [ ! -f "${WORKSPACE_DIR}/output/output_summary.json" ]; then
            echo "  ✗ 失败: 未找到 output_summary.json"
            echo "  原因: 验证需要 architecture-analyst 给出的并行配置"
            echo "  操作: 确认 Step 2 已正确调用 architecture-analyst"
            exit 1
        fi
        echo "  ✓ 通过: output_summary.json 已存在"

        check_iteration_cap

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 5 检查通过"
        echo "下一步: 先 Dummy 验证 → 再真实权重验证"
        ;;

    5.5)
        echo ""
        echo "=== Pre-Step 5.5: Triton 算子替换之前 ==="
        echo ""

        echo "检查 1: last_completed_step >= 5"
        if ! num_ge "${LAST_STEP}" 5; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 必须先完成 Step 5 两阶段验证"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo "检查 2: validation.real_weight_passed == true"
        if [ "$(json_get validation.real_weight_passed "${STATE_FILE}")" != "true" ]; then
            echo "  ✗ 失败: validation.real_weight_passed != true"
            echo "  操作: 回到 Step 5 完成真实权重验证"
            exit 1
        fi
        echo "  ✓ 通过: validation.real_weight_passed = true"

        SGLANG_PYTHON_DIR="$(json_get paths.sglang_python_dir "${STATE_FILE}")"
        SGL_KERNEL_NPU_DIR="$(json_get paths.sgl_kernel_npu_dir "${STATE_FILE}")"
        if [ -z "${SGLANG_PYTHON_DIR}" ] || [ -z "${SGL_KERNEL_NPU_DIR}" ]; then
            echo "  ⚠ 警告: adapter_state.json.paths 缺少 sglang_python_dir / sgl_kernel_npu_dir"
            echo "  操作: 在 Step 1 收集并写入这两个绝对路径后继续"
        else
            echo "  ✓ 通过: paths.sglang_python_dir / paths.sgl_kernel_npu_dir 已填写"
        fi

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 5.5 检查通过"
        echo "下一步: 先 --dry-run 预览替换计划"
        ;;

    5.6)
        echo ""
        echo "=== Pre-Step 5.6: GSM8K 精度测试之前 ==="
        echo ""

        echo "检查 1: last_completed_step >= 5.5"
        if ! num_ge "${LAST_STEP}" 5.5; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 必须先完成 Step 5.5(Triton 替换或确认无需替换)"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo "检查 2: output/triton_replace_result.json 已存在"
        if [ ! -f "${WORKSPACE_DIR}/output/triton_replace_result.json" ]; then
            echo "  ✗ 失败: 未找到 triton_replace_result.json"
            echo "  操作: 运行 triton_op_replace.py(即使不需要替换也会写该文件)"
            exit 1
        fi
        echo "  ✓ 通过: triton_replace_result.json 已存在"

        echo "检查 3: GSM8K 数据文件"
        GSM8K_DATA_PATH="$(json_get paths.gsm8k_data_path "${STATE_FILE}")"
        if [ -z "${GSM8K_DATA_PATH}" ]; then
            GSM8K_DATA_PATH="${WORKSPACE_DIR}/data/gsm8k.jsonl"
        fi
        if [ ! -f "${GSM8K_DATA_PATH}" ]; then
            echo "  ✗ 失败: 未找到 GSM8K 数据: ${GSM8K_DATA_PATH}"
            echo "  操作: 下载 GSM8K jsonl 到该路径,或在 adapter_state.json.paths.gsm8k_data_path 指定"
            echo "  参考: https://github.com/openai/grade-school-math"
            exit 1
        fi
        echo "  ✓ 通过: ${GSM8K_DATA_PATH}"

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 5.6 检查通过"
        echo "下一步: 真实权重启动服务后运行 gsm8k_accuracy_test.py"
        ;;

    6)
        echo ""
        echo "=== Pre-Step 6: 调用 test-validator 之前 ==="
        echo ""

        echo "检查 1: 验证已通过(next_action != call_debug_engineer)"
        if [ "${NEXT_ACTION}" = "call_debug_engineer" ]; then
            echo "  ✗ 失败: next_action = 'call_debug_engineer'"
            echo "  原因: Step 5 验证仍在进行中(需要继续调试)"
            echo "  违规: 调试未完成就试图跳到 test-validator"
            echo "  操作: 先完成 Step 5 的调试迭代"
            exit 1
        fi
        echo "  ✓ 通过: next_action = ${NEXT_ACTION}"

        echo "检查 2: last_completed_step >= 5.6"
        if ! num_ge "${LAST_STEP}" 5.6; then
            echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
            echo "  原因: 调用 test-validator 前必须完成 Step 5.5(Triton)与 Step 5.6(GSM8K)"
            echo "  操作: 完成 5.5 / 5.6 后再继续"
            exit 1
        fi
        echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"

        echo "检查 3: validation.gsm8k_passed == true"
        if [ "$(json_get validation.gsm8k_passed "${STATE_FILE}")" != "true" ]; then
            echo "  ✗ 失败: validation.gsm8k_passed != true"
            echo "  操作: 完成 Step 5.6 并通过精度测试"
            exit 1
        fi
        echo "  ✓ 通过: validation.gsm8k_passed = true"

        echo "检查 4: test_config.json 已就绪"
        if [ ! -f "${WORKSPACE_DIR}/input/test_config.json" ]; then
            echo "  ✗ 失败: 未找到 test_config.json"
            echo "  原因: test-validator 需要测试配置文件"
            echo "  操作: 调用 test-validator 前先准备 test_config.json"
            exit 1
        fi
        echo "  ✓ 通过: test_config.json 已存在"

        check_iteration_cap

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 6 检查通过"
        echo "下一步: 阅读 ${SKILL_DIR}/prompts/test_validator.md,调用 test-validator"
        ;;

    6.5)
        echo ""
        echo "=== Pre-Step 6.5: 调用 precision-rca 之前 ==="
        echo ""

        echo "检查 1: precision_suspect == true"
        if [ "$(json_get precision_suspect "${STATE_FILE}")" != "true" ]; then
            echo "  ⊝ 跳过: precision_suspect != true(精度未被标记可疑)"
            echo "  说明: Step 6.5 仅在用户/同事检出精度问题后触发"
            echo "  操作: 跳过本步,直接进 Step 7"
            exit 2
        fi
        echo "  ✓ 通过: precision_suspect = true"

        PRECISION_ONLY="$(json_get precision_only "${STATE_FILE}")"
        if [ "${PRECISION_ONLY}" = "true" ]; then
            echo "检查 2: precision_only 模式,last_completed_step >= 2"
            if ! num_ge "${LAST_STEP}" 2; then
                echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
                echo "  原因: 独立精度排查至少需要完成 Step 1-2(模型信息 + 架构摘要)"
                exit 1
            fi
            echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"
        else
            echo "检查 2: last_completed_step >= 6"
            if ! num_ge "${LAST_STEP}" 6; then
                echo "  ✗ 失败: last_completed_step = ${LAST_STEP}"
                echo "  原因: precision-rca 在 Step 6(test_validator)之后运行"
                echo "  操作: 完成 Step 6 后再继续;或是独立精度排查,请设置 precision_only=true"
                exit 1
            fi
            echo "  ✓ 通过: last_completed_step = ${LAST_STEP}"
        fi

        echo "检查 3: precision_context.json 已存在且 schema 合法"
        if [ ! -f "${WORKSPACE_DIR}/input/precision_context.json" ]; then
            echo "  ✗ 失败: 未找到 precision_context.json"
            echo "  原因: precision-rca 需要 input/precision_context.json"
            echo "  操作: 用户/同事先填好 precision_context.json"
            exit 1
        fi
        if ! WORKSPACE_DIR="${WORKSPACE_DIR}" "${PYTHON_BIN}" - <<'PYEOF'
import json, os, sys
ctx_path = os.path.join(os.environ["WORKSPACE_DIR"], "input/precision_context.json")
with open(ctx_path) as f:
    ctx = json.load(f)
required = ["model_path", "failing_prompts", "failing_evidence", "tolerance", "dtype"]
missing = [k for k in required if k not in ctx]
if missing:
    sys.stderr.write(f"  ✗ 失败: precision_context.json 缺字段 {missing}\n")
    sys.exit(1)
PYEOF
        then
            exit 1
        fi
        echo "  ✓ 通过: precision_context.json 合法"

        echo "检查 4: output_summary.json 已存在(architecture-analyst 输出)"
        if [ ! -f "${WORKSPACE_DIR}/output/output_summary.json" ]; then
            echo "  ✗ 失败: 未找到 output_summary.json"
            echo "  原因: precision-rca 需要架构信息"
            echo "  操作: 嵌入式入口确保 Step 2 完成;独立式入口由 probe_model_arch.py 写入"
            exit 1
        fi
        echo "  ✓ 通过: output_summary.json 已存在"

        echo ""
        echo "[sglang-npu-adapter] Pre-Step 6.5 检查通过"
        echo "下一步: 阅读 ${SKILL_DIR}/prompts/precision_rca.md,调用 precision-rca"
        ;;

    *)
        echo ""
        echo "说明: Step ${STEP_NUM} 未定义前置检查"
        echo "Step 0、1、3、4、7、8 不需要强制校验"
        ;;
esac

exit 0
