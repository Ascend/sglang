#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CACHE_ROOT="/root/.cache/tests/precise-test"
LOG_DIR="${CACHE_ROOT}/logs"
COV_ROOT="${CACHE_ROOT}/coverage"

mkdir -p "${LOG_DIR}" "${COV_ROOT}"

targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
  echo "Usage: $0 <test> [test ...]"
  exit 1
fi

overall_status=0

setup_coverage() {
  local target="$1"
  local name="${target%.py}"
  name="${name//\//__}"
  name="${name//::/--}"
  name="${name//[^a-zA-Z0-9_.-]/_}"
  local covdir="${COV_ROOT}/${name}"
  mkdir -p "${covdir}"
  export COVERAGE_FILE="${covdir}/coverage"
}

run_one() {
  local target="$1"
  local name="${target%.py}"
  name="${name//\//__}"
  name="${name//::/--}"
  name="${name//[^a-zA-Z0-9_.-]/_}"
  local log_file="${LOG_DIR}/${name}.log"

  echo "=== Running: ${target} ==="
  setup_coverage "${target}"

  set +e
  python -m coverage run --rcfile="${SCRIPT_DIR}/coveragerc" -m pytest -sv --color=yes "${target}" 2>&1 | tee "${log_file}"
  local status=$?
  set -e

  if [ "${status}" -ne 0 ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
    echo "=== FAILED: ${target} (log: ${log_file}) ==="
    overall_status=1
  else
    echo "=== PASSED: ${target} ==="
  fi
}

for target in "${targets[@]}"; do
  run_one "${target}"
done

echo "=== Done. Logs: ${LOG_DIR}/, Coverage: ${COV_ROOT}/ ==="
exit "${overall_status}"
