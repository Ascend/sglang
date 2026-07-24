#!/bin/bash
# 校验 Debug 工程师调用结果是否合法。在主流程应用修复之前调用。
#
# 用法: ./post-error-check.sh <workspace_dir>
#
# 退出码:
#   0  - 通过(可以应用修复)
#   1  - fix_instructions.json 缺失或 status 不可用(需用户介入)
#   2  - debug-engineer 报告需人介入(status=needs_human / inconclusive)

set -e

WORKSPACE_DIR="${1}"
if [ -z "${WORKSPACE_DIR}" ]; then
    echo "用法: $0 <workspace_dir>" >&2
    exit 1
fi

FIX_FILE="${WORKSPACE_DIR}/output/fix_instructions.json"
REPORT_FILE="${WORKSPACE_DIR}/output/debug_report.md"

echo "[post-error-check] 检查 debug-engineer 产物..."

if [ ! -f "${FIX_FILE}" ]; then
    echo "  ✗ 失败: 未找到 ${FIX_FILE}"
    echo "  原因: debug-engineer 必须在 output/ 下写 fix_instructions.json"
    exit 1
fi
echo "  ✓ ${FIX_FILE} 存在"

if [ ! -f "${REPORT_FILE}" ]; then
    echo "  ⚠ 警告: 未找到 ${REPORT_FILE}(debug-engineer 应同时输出中文报告)"
fi

# 校验 status 字段
STATUS=$(FIX_FILE="${FIX_FILE}" python3 - <<'PYEOF'
import json, os
with open(os.environ["FIX_FILE"]) as f:
    data = json.load(f)
print(data.get("status", "missing"))
PYEOF
)

echo "  fix_instructions.status = ${STATUS}"

case "${STATUS}" in
    fix_available|fix_verified)
        echo "[post-error-check] 通过。可以应用修复。"
        exit 0
        ;;
    needs_human)
        echo "[post-error-check] 阻塞: status=needs_human"
        echo "  原因: debug-engineer 判定需要人介入"
        echo "  动作: 主流程应上报用户,展示 debug_report.md"
        exit 2
        ;;
    inconclusive)
        echo "[post-error-check] 阻塞: status=inconclusive"
        echo "  原因: debug-engineer 未能定位根因"
        echo "  动作: 主流程应上报用户,展示 debug_report.md 的已尝试步骤"
        exit 2
        ;;
    *)
        echo "[post-error-check] 失败: 未知 status = '${STATUS}'"
        echo "  合法值: fix_available | fix_verified | needs_human | inconclusive"
        exit 1
        ;;
esac
