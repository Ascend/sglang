#!/bin/bash
# Step 8 之前的质量门禁:验证所有交付物齐全、关键验证确实通过。
#
# 用法: ./check-step-complete.sh <workspace_dir>
#
# 退出码:
#   0  - 全部通过
#   1  - 缺产物或验证未通过

set -euo pipefail

WORKSPACE_DIR="${1:-}"
if [ -z "${WORKSPACE_DIR}" ]; then
    echo "用法: $0 <workspace_dir>" >&2
    exit 1
fi

STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"
if [ ! -f "${STATE_FILE}" ]; then
    echo "✗ 未找到 ${STATE_FILE}" >&2
    exit 1
fi

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

echo "[check-step-complete] 质量门禁检查..."

FAIL=0

check_file() {
    local path="${1}"
    local desc="${2}"
    if [ -f "${path}" ]; then
        echo "  ✓ ${desc}: ${path}"
    else
        echo "  ✗ ${desc}: 缺失 ${path}"
        FAIL=1
    fi
}

check_json_bool() {
    local file="${1}"
    local field="${2}"
    local desc="${3}"
    local value
    value="$(json_get "${field}" "${file}")"
    if [ "${value}" = "true" ]; then
        echo "  ✓ ${desc}: ${field} = true"
    else
        echo "  ✗ ${desc}: ${field} = '${value}'(期望 true)"
        FAIL=1
    fi
}

# 规划文件
check_file "${WORKSPACE_DIR}/task_plan.md" "task_plan.md"
check_file "${WORKSPACE_DIR}/findings.md" "findings.md"
check_file "${WORKSPACE_DIR}/progress.md" "progress.md"
check_file "${WORKSPACE_DIR}/adapter_state.json" "adapter_state.json"

# 架构分析师产物
check_file "${WORKSPACE_DIR}/output/output_summary.json" "output_summary.json"
check_file "${WORKSPACE_DIR}/output/analysis_report.md" "analysis_report.md"

# Dummy + 真实权重验证(文件存在且 status=passed)
for stage in dummy real; do
    RESULT_FILE="${WORKSPACE_DIR}/logs/${stage}_inference.json"
    if [ -f "${RESULT_FILE}" ]; then
        STATUS="$(json_get status "${RESULT_FILE}")"
        if [ "${STATUS}" = "passed" ]; then
            echo "  ✓ ${stage} 验证: status = passed"
        else
            echo "  ✗ ${stage} 验证: status = '${STATUS}'(期望 passed)"
            FAIL=1
        fi
    else
        echo "  ✗ ${stage} 验证: 缺失 ${RESULT_FILE}"
        FAIL=1
    fi
done

# Triton 算子替换(必须真实执行过,不能只有 dry-run)
TRITON_FILE="${WORKSPACE_DIR}/output/triton_replace_result.json"
check_file "${TRITON_FILE}" "triton_replace_result.json"
if [ -f "${TRITON_FILE}" ]; then
    DRY_RUN="$(json_get dry_run "${TRITON_FILE}")"
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  ✗ Triton 替换: dry_run = true(只预览未执行)"
        FAIL=1
    else
        echo "  ✓ Triton 替换: 已执行"
    fi
fi

# GSM8K 精度
GSM8K_FILE="${WORKSPACE_DIR}/output/gsm8k_accuracy_result.json"
check_file "${GSM8K_FILE}" "gsm8k_accuracy_result.json"
if [ -f "${GSM8K_FILE}" ]; then
    check_json_bool "${GSM8K_FILE}" "passed" "GSM8K 精度"
fi

# 测试工程师产物
check_file "${WORKSPACE_DIR}/output/test_result.json" "test_result.json"
check_file "${WORKSPACE_DIR}/output/test_report.md" "test_report.md"

# validation 状态
DUMMY_PASSED="$(json_get validation.dummy_passed "${STATE_FILE}")"
REAL_PASSED="$(json_get validation.real_weight_passed "${STATE_FILE}")"
GSM8K_PASSED="$(json_get validation.gsm8k_passed "${STATE_FILE}")"
PRECISION_SUSPECT="$(json_get precision_suspect "${STATE_FILE}")"
echo "  validation: dummy=${DUMMY_PASSED} real=${REAL_PASSED} gsm8k=${GSM8K_PASSED} prec_suspect=${PRECISION_SUSPECT}"

for item in "dummy_passed:${DUMMY_PASSED}" "real_weight_passed:${REAL_PASSED}" "gsm8k_passed:${GSM8K_PASSED}"; do
    key="${item%%:*}"
    value="${item##*:}"
    if [ "${value}" != "true" ]; then
        echo "  ✗ adapter_state.json.validation.${key} = '${value}'(期望 true)"
        FAIL=1
    fi
done

# precision_suspect=true 时强制要求 RCA 产物
if [ "${PRECISION_SUSPECT}" = "true" ]; then
    echo ""
    echo "  precision_suspect=true,检查 RCA 产物:"
    check_file "${WORKSPACE_DIR}/output/root_cause.json" "root_cause.json"
    check_file "${WORKSPACE_DIR}/output/precision_rca_report.md" "precision_rca_report.md"

    RCA_FILE="${WORKSPACE_DIR}/output/root_cause.json"
    if [ -f "${RCA_FILE}" ]; then
        RCA_STATUS="$(json_get status "${RCA_FILE}")"
        case "${RCA_STATUS}" in
            fixed|located_needs_human_fix|located_inconclusive|cannot_reproduce|hf_load_failed|launch_failed_handoff)
                echo "  ✓ RCA status = ${RCA_STATUS}"
                ;;
            *)
                echo "  ✗ RCA status = '${RCA_STATUS}'(合法值: fixed / located_needs_human_fix / located_inconclusive / cannot_reproduce / hf_load_failed / launch_failed_handoff)"
                FAIL=1
                ;;
        esac

        # status=fixed 时额外校验:fix.patch 存在 + drift_after_fix.max_abs < tolerance.atol
        if [ "${RCA_STATUS}" = "fixed" ]; then
            if [ ! -f "${WORKSPACE_DIR}/output/fix.patch" ]; then
                echo "  ✗ status=fixed 但 output/fix.patch 缺失"
                FAIL=1
            else
                echo "  ✓ output/fix.patch 存在"
            fi

            if WORKSPACE_DIR="${WORKSPACE_DIR}" "${PYTHON_BIN}" - <<'PYEOF'
import json
import os
import sys

ws = os.environ["WORKSPACE_DIR"]
try:
    with open(os.path.join(ws, "output", "root_cause.json"), encoding="utf-8") as f:
        rca = json.load(f)
    with open(os.path.join(ws, "input", "precision_context.json"), encoding="utf-8") as f:
        ctx = json.load(f)
except Exception as exc:
    print(f"  ✗ 无法校验 drift: {exc}")
    sys.exit(1)

max_abs = (rca.get("drift_after_fix") or {}).get("max_abs")
atol = (ctx.get("tolerance") or {}).get("atol")
if max_abs is None or atol is None:
    print(f"  ✗ drift 校验缺少字段: drift_after_fix.max_abs={max_abs}, tolerance.atol={atol}")
    sys.exit(1)
if float(max_abs) >= float(atol):
    print(f"  ✗ drift 未归零: max_abs={max_abs} >= atol={atol}")
    sys.exit(1)
print(f"  ✓ drift 已归零: max_abs={max_abs} < atol={atol}")
PYEOF
            then
                :
            else
                FAIL=1
            fi
        fi
    fi
fi

# 最终教程
MODEL_NAME="$(json_get model_name "${STATE_FILE}")"
TUTORIAL="${WORKSPACE_DIR}/output/${MODEL_NAME}.md"
if [ -f "${TUTORIAL}" ]; then
    echo "  ✓ 最终教程: ${TUTORIAL}"
else
    echo "  ✗ 最终教程未生成: ${TUTORIAL}(Step 7 生成)"
    FAIL=1
fi

echo ""
if [ "${FAIL}" -eq 0 ]; then
    echo "[check-step-complete] 全部通过,可以进入 Step 8 交接"
    exit 0
else
    echo "[check-step-complete] 失败:补齐缺失产物后重新检查"
    exit 1
fi
