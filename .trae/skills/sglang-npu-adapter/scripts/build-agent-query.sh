#!/bin/bash
# 为子 Agent 构造一份开箱即用的完整 query。
#
# 用法:    ./build-agent-query.sh <agent_name> <workspace_dir>
#          agent_name: architecture_analyst | debug_engineer | test_validator
#
# 行为:
#   1. 读取 adapter_state.json 解析 skill_dir（绝对路径）
#   2. 读取 prompts/<agent>.md，替换 {{WORKSPACE_DIR}} / {{SKILL_DIR}}
#   3. 在前面注入 PREAMBLE，强制子 Agent 先 Read() 全部 P0 参考文档
#      再进行任何推理 —— 避免子 Agent 在缺乏领域知识时盲目作答
#   4. 将最终 query 写到 stdout（或 --output 指定的文件）
#
# 示例:
#   QUERY=$(bash build-agent-query.sh debug_engineer "$WORKSPACE_DIR")
#   # 然后: Task(subagent_type="debug-engineer", query="$QUERY", ...)

set -e

AGENT_NAME="${1}"
WORKSPACE_DIR="${2}"
OUTPUT_FILE=""

# 可选 --output <file> 参数（第 3、4 个参数）
if [ "${3}" = "--output" ] && [ -n "${4}" ]; then
    OUTPUT_FILE="${4}"
fi

if [ -z "${AGENT_NAME}" ] || [ -z "${WORKSPACE_DIR}" ]; then
    echo "用法: $0 <agent_name> <workspace_dir> [--output <file>]" >&2
    echo "  agent_name: architecture_analyst | debug_engineer | test_validator" >&2
    exit 1
fi

STATE_FILE="${WORKSPACE_DIR}/adapter_state.json"
if [ ! -f "${STATE_FILE}" ]; then
    echo "错误: 未找到 adapter_state.json，路径: ${STATE_FILE}" >&2
    echo "      请先运行 init-adapter-session.sh" >&2
    exit 1
fi

# 优先从 state 文件读取 skill_dir；对未记录该字段的旧会话，退回到环境变量。
SKILL_DIR=$(STATE_FILE="${STATE_FILE}" python3 <<'PYEOF'
import json, os, sys
with open(os.environ["STATE_FILE"]) as f:
    s = json.load(f)
val = s.get("skill_dir")
if not val:
    tsp = os.environ.get("TRAE_SKILLS_PATH", "")
    if not tsp:
        sys.stderr.write(
            "错误: adapter_state.json 中没有 skill_dir，且 TRAE_SKILLS_PATH 也未设置。\n"
            "      请重新运行 init-adapter-session.sh 或导出 TRAE_SKILLS_PATH。\n"
        )
        sys.exit(1)
    val = os.path.join(tsp, "sglang-npu-adapter")
print(val)
PYEOF
)

# 将 agent_name 映射到对应的 prompt 文件与 P0（必读）参考文档
case "${AGENT_NAME}" in
    architecture_analyst)
        PROMPT_FILE="${SKILL_DIR}/prompts/model_analyzer.md"
        SUBAGENT_TYPE="architecture-analyst"
        P0_REFS=(
            "${SKILL_DIR}/references/architecture_analyst/llm_architecture.md"
            "${SKILL_DIR}/references/architecture_analyst/moe_architecture.md"
            "${SKILL_DIR}/references/architecture_analyst/npu_specifications.md"
        )
        ;;
    debug_engineer)
        PROMPT_FILE="${SKILL_DIR}/prompts/debug_engineer.md"
        SUBAGENT_TYPE="debug-engineer"
        P0_REFS=(
            "${SKILL_DIR}/references/debug_engineer/common_errors.md"
            "${SKILL_DIR}/references/debug_engineer/npu_specific_issues.md"
            "${SKILL_DIR}/references/debug_engineer/attention_debug.md"
            "${SKILL_DIR}/references/shared/npu_basics.md"
        )
        ;;
    test_validator)
        PROMPT_FILE="${SKILL_DIR}/prompts/test_validator.md"
        SUBAGENT_TYPE="test-validator"
        P0_REFS=(
            "${SKILL_DIR}/references/test_validator/basic_inference_test.md"
            "${SKILL_DIR}/references/test_validator/npu_validation.md"
        )
        ;;
    *)
        echo "错误: 未知 agent '${AGENT_NAME}'" >&2
        echo "      合法取值: architecture_analyst | debug_engineer | test_validator" >&2
        exit 1
        ;;
esac

if [ ! -f "${PROMPT_FILE}" ]; then
    echo "错误: 未找到 prompt 文件: ${PROMPT_FILE}" >&2
    exit 1
fi

for ref in "${P0_REFS[@]}"; do
    if [ ! -f "${ref}" ]; then
        echo "错误: 未找到 P0 参考文档: ${ref}" >&2
        exit 1
    fi
done

# 构建输出
BUILD=$(mktemp)
trap 'rm -f "${BUILD}"' EXIT

{
    cat <<EOF
=== PREAMBLE（由 build-agent-query.sh 注入，不可忽略）===
身份: ${SUBAGENT_TYPE}
工作目录: ${WORKSPACE_DIR}
Skill 目录: ${SKILL_DIR}

【强制前置阅读】
你必须**首先**用 Read 工具读完以下参考文档，才能进行任何分析、推理或输出决策。
在读完所有文档之前，禁止输出 "我认为..." / "可能是..." / "让我查查..." 等推测性或探查性陈述。

必读列表（顺序不限，全部读完为止）：
EOF
    for ref in "${P0_REFS[@]}"; do
        echo "  - ${ref}"
    done
    cat <<EOF

读完后，按下方原 prompt 的执行流程推进。中途如需补充阅读，优先查看 prompt 中"知识库参考 P1"小节列出的文档。
=== PREAMBLE END ===

EOF

    # 输出已填充变量的 prompt
    PROMPT_FILE="${PROMPT_FILE}" \
    WORKSPACE_DIR="${WORKSPACE_DIR}" \
    SKILL_DIR="${SKILL_DIR}" \
    python3 <<'PYEOF'
import os, sys
with open(os.environ["PROMPT_FILE"], "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("{{WORKSPACE_DIR}}", os.environ["WORKSPACE_DIR"])
content = content.replace("{{SKILL_DIR}}", os.environ["SKILL_DIR"])
sys.stdout.write(content)
PYEOF
} > "${BUILD}"

# 始终在 logs/agent_calls/ 下归档带时间戳的快照与索引条目，
# 让每次子 Agent 调用都留下审计痕迹。
ARCHIVE_DIR="${WORKSPACE_DIR}/logs/agent_calls"
mkdir -p "${ARCHIVE_DIR}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
SNAPSHOT="${ARCHIVE_DIR}/${AGENT_NAME}_${TIMESTAMP}.txt"
cp "${BUILD}" "${SNAPSHOT}"

INDEX="${ARCHIVE_DIR}/index.jsonl"
SNAPSHOT="${SNAPSHOT}" AGENT_NAME="${AGENT_NAME}" SUBAGENT_TYPE="${SUBAGENT_TYPE}" \
INDEX="${INDEX}" TIMESTAMP="${TIMESTAMP}" python3 <<'PYEOF'
import hashlib, json, os, datetime

snapshot = os.environ["SNAPSHOT"]
with open(snapshot, "rb") as f:
    data = f.read()

entry = {
    "timestamp": os.environ["TIMESTAMP"],
    "iso_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "agent": os.environ["AGENT_NAME"],
    "subagent_type": os.environ["SUBAGENT_TYPE"],
    "snapshot": snapshot,
    "bytes": len(data),
    "lines": data.count(b"\n") + (0 if data.endswith(b"\n") else 1),
    "sha256": hashlib.sha256(data).hexdigest(),
}
with open(os.environ["INDEX"], "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
PYEOF

echo "[build-agent-query] 快照: ${SNAPSHOT}" >&2
echo "[build-agent-query] 索引: ${INDEX}" >&2

if [ -n "${OUTPUT_FILE}" ]; then
    mkdir -p "$(dirname "${OUTPUT_FILE}")"
    cp "${BUILD}" "${OUTPUT_FILE}"
    echo "[build-agent-query] 已写入 query 到 ${OUTPUT_FILE}" >&2
else
    cat "${BUILD}"
fi
