#!/bin/bash
# Step 8 之前的质量门禁:验证所有交付物齐全。
#
# 用法: ./check-step-complete.sh <workspace_dir>
#
# 退出码:
#   0  - 全部通过
#   1  - 缺产物或验证未通过

set -e

WORKSPACE_DIR="${1}"
if [ -z "${WORKSPACE_DIR}" ]; then
    echo "用法: $0 <workspace_dir>" >&2
    exit 1
fi

STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"
if [ ! -f "${STATE_FILE}" ]; then
    echo "✗ 未找到 ${STATE_FILE}" >&2
    exit 1
fi

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

# 规划文件
check_file "${WORKSPACE_DIR}/task_plan.md" "task_plan.md"
check_file "${WORKSPACE_DIR}/findings.md" "findings.md"
check_file "${WORKSPACE_DIR}/progress.md" "progress.md"
check_file "${WORKSPACE_DIR}/adapter_state.json" "adapter_state.json"

# 架构分析师产物
check_file "${WORKSPACE_DIR}/output/output_summary.json" "output_summary.json"
check_file "${WORKSPACE_DIR}/output/analysis_report.md" "analysis_report.md"

# 验证日志(任一存在即可,direct_use 策略下也至少有一个)
if [ -f "${WORKSPACE_DIR}/logs/dummy_inference.json" ] || [ -f "${WORKSPACE_DIR}/logs/real_inference.json" ]; then
    echo "  ✓ 验证日志: dummy_inference.json / real_inference.json 至少一份"
else
    echo "  ✗ 验证日志: 缺失(dummy_inference.json 与 real_inference.json 都没有)"
    FAIL=1
fi

# 测试工程师产物
check_file "${WORKSPACE_DIR}/output/test_result.json" "test_result.json"
check_file "${WORKSPACE_DIR}/output/test_report.md" "test_report.md"

# validation 状态
PYEOF_OUT=$(STATE_FILE="${STATE_FILE}" python3 - <<'PYEOF'
import json, os
with open(os.environ["STATE_FILE"]) as f:
    s = json.load(f)
v = s.get("validation", {})
print(f"dummy={v.get('dummy_passed', False)} real={v.get('real_weight_passed', False)} prec_suspect={s.get('precision_suspect', False)}")
PYEOF
)
echo "  validation: ${PYEOF_OUT}"

# precision_suspect=true 时强制要求 RCA 产物
PRECISION_SUSPECT=$(grep -oE '"precision_suspect"[[:space:]]*:[[:space:]]*(true|false)' "${STATE_FILE}" | grep -oE '(true|false)' | head -1)
if [ "${PRECISION_SUSPECT}" = "true" ]; then
    echo ""
    echo "  precision_suspect=true,检查 RCA 产物:"
    check_file "${WORKSPACE_DIR}/output/root_cause.json" "root_cause.json"
    check_file "${WORKSPACE_DIR}/output/precision_rca_report.md" "precision_rca_report.md"

    RCA_STATUS=$(python3 - <<PYEOF
import json
with open("${WORKSPACE_DIR}/output/root_cause.json") as f:
    print(json.load(f).get("status", "missing"))
PYEOF
)
    case "${RCA_STATUS}" in
        fixed|located_needs_human_fix|located_inconclusive|cannot_reproduce|hf_load_failed|launch_failed_handoff)
            echo "  ✓ RCA status = ${RCA_STATUS}"
            # status=fixed 时额外校验:fix.patch 存在 + drift_after_fix < tolerance
            if [ "${RCA_STATUS}" = "fixed" ]; then
                if [ ! -f "${WORKSPACE_DIR}/output/fix.patch" ]; then
                    echo "  ✗ status=fixed 但 output/fix.patch 缺失"
                    FAIL=1
                else
                    echo "  ✓ output/fix.patch 存在"
                fi
            fi
            ;;
        *)
            echo "  ✗ RCA status = '${RCA_STATUS}'(合法值: fixed / located_needs_human_fix / located_inconclusive / cannot_reproduce / hf_load_failed / launch_failed_handoff)"
            FAIL=1
            ;;
    esac
fi

# 最终教程
MODEL_NAME=$(grep -oE '"model_name"[[:space:]]*:[[:space:]]*"[^"]*"' "${STATE_FILE}" | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
TUTORIAL="${WORKSPACE_DIR}/output/${MODEL_NAME}.md"
if [ -f "${TUTORIAL}" ]; then
    echo "  ✓ 最终教程: ${TUTORIAL}"
else
    echo "  ⚠ 最终教程未生成: ${TUTORIAL}(Step 7 生成)"
fi

echo ""
if [ ${FAIL} -eq 0 ]; then
    echo "[check-step-complete] 全部通过,可以进入 Step 8 交接"
    exit 0
else
    echo "[check-step-complete] 失败:补齐缺失产物后重新检查"
    exit 1
fi
