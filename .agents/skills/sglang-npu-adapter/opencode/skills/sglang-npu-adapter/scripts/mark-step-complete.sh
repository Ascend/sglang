#!/bin/bash
# 在 adapter_state.json 中标记某个 Step 为完成
# 更新字段:last_completed_step=N, current_step=<下一步>, last_update=<now>
# 硬性门禁:若 task_plan.md / progress.md / findings.md 自上一步快照以来
# 没有任何修改,则拒绝标记完成(退出码 1)。
# 仅当三份规划文档全部为本步骤更新过时才退出 0。
# 用法: ./mark-step-complete.sh <step_num> <workspace_dir>

set -euo pipefail

STEP_NUM="${1:-}"
WORKSPACE_DIR="${2:-.trae/workspace/default}"
STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"

if [ -z "${STEP_NUM}" ]; then
    echo "用法: $0 <step_num> <workspace_dir>"
    exit 2
fi

if [ ! -f "${STATE_FILE}" ]; then
    echo "错误: 未找到 adapter_state.json,路径: ${STATE_FILE}"
    exit 2
fi

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

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
        exit 2
    fi
fi

STEP_NUM="${STEP_NUM}" \
STATE_FILE="${STATE_FILE}" \
WORKSPACE_DIR="${WORKSPACE_DIR}" \
TIMESTAMP="${TIMESTAMP}" \
SKILL_DIR="${SKILL_DIR}" \
"${PYTHON_BIN}" <<'PYEOF'
import hashlib
import json
import os
import sys

raw_step = os.environ["STEP_NUM"]
# 支持小数 step(5.5 Triton 替换、5.6 GSM8K、6.5 precision-rca)
if "." in raw_step:
    step_num = float(raw_step)
else:
    step_num = int(raw_step)
step_key = raw_step

# 显式的步骤顺序,避免 "step + 1" 对小数步产生错误推进。
STEP_ORDER = ["0", "1", "2", "3", "4", "5", "5.5", "5.6", "6", "6.5", "7", "8"]
if step_key in STEP_ORDER:
    idx = STEP_ORDER.index(step_key)
    next_step = STEP_ORDER[idx + 1] if idx + 1 < len(STEP_ORDER) else 9
    next_step = int(next_step) if float(next_step).is_integer() else float(next_step)
else:
    next_step = step_num + 1

state_file = os.environ["STATE_FILE"]
workspace_dir = os.environ["WORKSPACE_DIR"]
skill_dir = os.environ["SKILL_DIR"]

# 必须在每个 Step 中更新的三份规划文档
PLANNING_DOCS = ["task_plan.md", "progress.md", "findings.md"]


def sha256_of(path):
    """计算文件 sha256;文件不存在返回 None"""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


with open(state_file, "r", encoding="utf-8") as f:
    state = json.load(f)

# 上一步 = state 里真实记录的 last_completed_step,而不是按步号推导。
# 这样 5.5 完成后,5.6 会与 5.5 的快照比较,而不是与 Step 5 比较。
last_completed = state.get("last_completed_step", 0)
prev_step_key = None
if last_completed is not None:
    prev_step_key = str(last_completed)
    if prev_step_key.endswith(".0"):
        prev_step_key = prev_step_key[:-2]

snapshots = state.get("step_snapshots", {})
prev_snapshot = snapshots.get(prev_step_key) if prev_step_key else None

# 计算当前三份文档的哈希,并与上一步快照对比
current_snapshot = {}
unchanged = []
missing = []
for name in PLANNING_DOCS:
    digest = sha256_of(os.path.join(workspace_dir, name))
    if digest is None:
        missing.append(name)
        continue
    current_snapshot[name] = digest
    if prev_snapshot and prev_snapshot.get(name) == digest:
        unchanged.append(name)

# 情况 1:必需文档缺失 → 阻塞
if missing:
    print(f"[sglang-npu-adapter] 已阻塞: 无法标记 Step {step_num} 为完成")
    print("")
    print("原因: 以下规划文档缺失:")
    for name in missing:
        print(f"  - {workspace_dir}/{name}")
    print("")
    print("恢复步骤:")
    print(f"  1. 执行 {skill_dir}/scripts/init-adapter-session.sh 重新初始化 workspace,或")
    print(f"     从 {skill_dir}/templates/ 复制模板到")
    print(f"     {workspace_dir}/")
    print(f"  2. 填写文档,反映 Step {step_num} 的工作内容。")
    print(f"  3. 重新运行: {skill_dir}/scripts/mark-step-complete.sh {step_num} {workspace_dir}")
    sys.exit(1)

# 情况 2:本步骤未修改任一规划文档 → 阻塞
if unchanged:
    print(f"[sglang-npu-adapter] 已阻塞: 无法标记 Step {step_num} 为完成")
    print("")
    print(f"原因: 以下规划文档在 Step {step_num} 期间未被更新")
    print(f"      (其内容与 Step {prev_step_key} 完成时的快照完全一致):")
    for name in unchanged:
        print(f"  - {workspace_dir}/{name}")
    print("")
    print("为什么必须更新:")
    print("  - task_plan.md   记录下一步的计划调整。")
    print("  - progress.md    记录本步骤已完成的事项(状态、产物、决策)。")
    print("  - findings.md    记录本步骤新学到的内容(架构要点、坑、链接)。")
    print("  三份文档均由 SKILL.md 强制要求,用于可追溯性与 debug-engineer 交接。")
    print("")
    print("恢复步骤:")
    print(f"  1. 打开并编辑上面列出的每个文件,使其反映 Step {step_num} 的工作:")
    print(f"     - progress.md 新增一节 '## Step {step_num}'(写状态、产物)。")
    print(f"     - findings.md 追加 step-{step_num} 的发现。")
    print(f"     - task_plan.md 若计划/顺序有变则更新;无变化也写一行 'Step {step_num}: 无计划调整'。")
    print(f"  2. 重新运行: {skill_dir}/scripts/mark-step-complete.sh {step_num} {workspace_dir}")
    print("")
    print("不要通过手动修改 adapter_state.json 来绕过此检查。")
    sys.exit(1)

# 情况 3:通过校验 → 写入新状态与新快照
state["last_completed_step"] = step_num
state["current_step"] = next_step
state["last_update"] = os.environ["TIMESTAMP"]
snapshots = state.setdefault("step_snapshots", {})
snapshots[step_key] = current_snapshot

with open(state_file, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=4, ensure_ascii=False)

print(f"[sglang-npu-adapter] Step {step_num} 已标记为完成")
print(f"  last_completed_step: {state['last_completed_step']}")
print(f"  current_step:        {state['current_step']}")
print(f"  last_update:         {state['last_update']}")
if prev_snapshot:
    print(f"  规划文档:            自 Step {prev_step_key} 起三份均已更新 ✓")
else:
    print(f"  规划文档:            已记录基线快照(无上一步可对比)")
PYEOF
