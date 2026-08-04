#!/bin/bash
# verify_directory_structure.sh
# Verifies the directory creation logic from all three NPU test execution paths.
# This can be run locally or via CI without using any NPU/k8s resources.
#
# Usage:
#   bash scripts/ci/verify_directory_structure.sh
#
# Simulates the path construction from:
#   1. nightly-test-npu-e2e-single-node.yml
#   2. nightly-test-npu-e2e-multi-node.yml
#   3. single-test-npu.yml

set -e

# ============================================================
# Simulated GitHub Actions context variables
# ============================================================
GITHUB_RUN_ID="${GITHUB_RUN_ID:-12345678901}"
GITHUB_REF_NAME="${GITHUB_REF_NAME:-pr_branch_plli_for_check_in_2}"
HOSTNAME="${HOSTNAME:-test-host-001}"
RUN_ID="${RUN_ID}"  # keep empty to test fallback

# ============================================================
# Use a temp directory instead of /root/.cache
# ============================================================
BASE_DIR="/tmp/verify-dirs-$(date +%Y%m%d-%H%M%S)"
rm -rf "$BASE_DIR"
mkdir -p "$BASE_DIR"

PASS_COUNT=0
FAIL_COUNT=0

echo "============================================"
echo " Directory Structure Verification Test"
echo "============================================"
echo " Base dir:  $BASE_DIR"
echo " Run ID:    $GITHUB_RUN_ID"
echo " Ref name:  $GITHUB_REF_NAME"
echo " HOSTNAME:  $HOSTNAME"
echo ""

pass() {
  echo "  [PASS] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  echo "  [FAIL] $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

# ============================================================
# Helper: verify a single path construction
# Mirrors the EXACT logic from each workflow
# ============================================================
verify_single_node_e2e() {
  local path_label="$1"
  local workflow_type="$2"
  local test_type="$3"
  local test_case="$4"

  echo "--- $path_label ---"

  # --- EXACT LOGIC from nightly-test-npu-e2e-single-node.yml lines 85-94 ---
  tc_name=${test_case##*/}
  tc_name=${tc_name%.*}
  run_id=${RUN_ID:-${GITHUB_RUN_ID}}
  # Note: ${{ inputs.workflow_type }} and ${{ inputs.test_type }} replaced at
  # workflow template expansion time. Simulation passes them as function args.
  branch=$(echo "$GITHUB_REF_NAME" | tr '/' '-')
  timestamp=$(date +%H%M%S)
  test_data_output_path=${BASE_DIR}/tests/output/${branch}-${run_id}/${workflow_type}/${test_type}/${tc_name}-${timestamp}
  mkdir -p ${test_data_output_path}
  # METRICS_DATA_FILE export is simulated by writing a file
  echo "  output_path=$test_data_output_path"

  # --- EXACT LOGIC from nightly-test-npu-e2e-single-node.yml lines 152-155 ---
  log_path="${BASE_DIR}/tests/logs/log/${branch}-${run_id}/${workflow_type}/${tc_name}-${timestamp}/${HOSTNAME}"
  rm -rf ${log_path} 2>/dev/null || true
  mkdir -p ${log_path}
  echo "  log_path=$log_path"

  # Write marker files to simulate test output
  echo "test output" > "${test_data_output_path}/test_output.log"
  echo '{"metrics": {}}' > "${test_data_output_path}/metrics.json"
  echo "test log content" > "${log_path}/${tc_name}.log"

  # Verify
  if [ -f "${test_data_output_path}/test_output.log" ] && [ -f "${test_data_output_path}/metrics.json" ]; then
    pass "Output directory created and files written"
  else
    fail "Output directory or files missing"
  fi

  if [ -f "${log_path}/${tc_name}.log" ]; then
    pass "Log directory created and files written"
  else
    fail "Log directory or files missing"
  fi

  echo ""
}

# ============================================================
# Path 3: Standalone (single-test-npu.yml lines 79-98)
# Uses the EXACT logic with for-loop over test cases
# ============================================================
verify_standalone() {
  local path_label="$1"
  shift
  local test_cases=("$@")

  echo "--- $path_label ---"

  # --- EXACT LOGIC from single-test-npu.yml lines 79-83 ---
  workflow_type="single"
  run_id=${RUN_ID:-${GITHUB_RUN_ID}}
  branch=$(echo "$GITHUB_REF_NAME" | tr '/' '-')
  timestamp=$(date +%H%M%S)
  log_path_base="${BASE_DIR}/tests/logs/log/${branch}-${run_id}/${workflow_type}/${timestamp}"
  echo "  log_path_base=$log_path_base"

  local tc_idx=0
  for test_case in "${test_cases[@]}"; do
    tc_name=$(basename ${test_case} .py)

    # --- EXACT LOGIC from single-test-npu.yml lines 90-98 ---
    test_data_output_path=${BASE_DIR}/tests/output/${branch}-${run_id}/${workflow_type}/single/${tc_name}-${timestamp}
    mkdir -p ${test_data_output_path}

    log_path="${log_path_base}/${tc_name}/${HOSTNAME}"
    rm -rf ${log_path} 2>/dev/null || true
    mkdir -p ${log_path}

    echo "  [$tc_idx] tc_name=$tc_name"
    echo "          output_path=$test_data_output_path"
    echo "          log_path=$log_path"

    # Write marker files
    echo "test output" > "${test_data_output_path}/test_output.log"
    echo '{"metrics": {}}' > "${test_data_output_path}/metrics.json"
    echo "test log content" > "${log_path}/${tc_name}.log"

    # Verify
    if [ -f "${test_data_output_path}/test_output.log" ] && [ -f "${test_data_output_path}/metrics.json" ]; then
      pass "  [$tc_idx] Output directory OK: $tc_name"
    else
      fail "  [$tc_idx] Output directory missing: $tc_name"
    fi

    if [ -f "${log_path}/${tc_name}.log" ]; then
      pass "  [$tc_idx] Log directory OK: $tc_name"
    else
      fail "  [$tc_idx] Log directory missing: $tc_name"
    fi

    tc_idx=$((tc_idx + 1))
  done
  echo ""
}

# ============================================================
# Run all verification paths
# ============================================================

# Path 1: Single-node E2E with workflow_type=nightly, test_type=perf
verify_single_node_e2e \
  "Single-node E2E (nightly/perf)" \
  "nightly" \
  "perf" \
  "test/registered/ascend/performance/test_npu_bench_serving_performance.py"

# Path 1: Single-node E2E with workflow_type=fulltest, test_type=accuracy
verify_single_node_e2e \
  "Single-node E2E (fulltest/accuracy)" \
  "fulltest" \
  "accuracy" \
  "test/registered/ascend/accuracy/test_npu_qwen3_next_80b_a3b_sglang.py"

# Path 2: Multi-node E2E (uses identical path construction as single-node)
verify_single_node_e2e \
  "Multi-node E2E (nightly/perf)" \
  "nightly" \
  "perf" \
  "test/registered/ascend/performance/test_npu_deepseek_v3_multi_node_ep.py"

# Path 3: Standalone (single-test-npu.yml style)
verify_standalone \
  "Standalone (single-test-npu)" \
  "test/registered/ascend/performance/test_npu_example.py" \
  "test/registered/ascend/accuracy/test_npu_accuracy_example.py"

# ============================================================
# Verify top-level {branch}-{run_id} directory structure
# ============================================================
echo "============================================"
echo " Top-Level Directory Structure Check"
echo "============================================"

EXPECTED_TOP="${BASE_DIR}/tests/output/${branch}-${run_id}"
echo " Expected top-level output: $EXPECTED_TOP"

if [ -d "$EXPECTED_TOP" ]; then
  pass "Top-level output directory exists: {branch}-{run_id} = ${branch}-${run_id}"
else
  fail "Top-level output directory MISSING: ${EXPECTED_TOP}"
fi

EXPECTED_LOG_TOP="${BASE_DIR}/tests/logs/log/${branch}-${run_id}"
if [ -d "$EXPECTED_LOG_TOP" ]; then
  pass "Top-level log directory exists: {branch}-{run_id} = ${branch}-${run_id}"
else
  fail "Top-level log directory MISSING: ${EXPECTED_LOG_TOP}"
fi

# Show expected sub-directories for each workflow_type
echo ""
echo "--- Sub-directories under {branch}-{run_id}/output ---"
for wf_type in "nightly" "fulltest" "single"; do
  wf_dir="${EXPECTED_TOP}/${wf_type}"
  if [ -d "$wf_dir" ]; then
    echo "  workflow_type=$wf_type: EXISTS"
    ls -d "$wf_dir"/*/ 2>/dev/null | while read d; do
      echo "    test_type=$(basename $d)/"
    done
  else
    echo "  workflow_type=$wf_type: NOT FOUND"
  fi
done

echo ""
echo "--- Sub-directories under {branch}-{run_id}/logs/log ---"
for wf_type in "nightly" "fulltest" "single"; do
  wf_dir="${EXPECTED_LOG_TOP}/${wf_type}"
  if [ -d "$wf_dir" ]; then
    echo "  workflow_type=$wf_type: EXISTS"
    ls -d "$wf_dir"/*/ 2>/dev/null | while read d; do
      echo "    $(basename $d)/"
    done
  else
    echo "  workflow_type=$wf_type: NOT FOUND"
  fi
done

# ============================================================
# Full directory tree
# ============================================================
echo ""
echo "============================================"
echo " Full Directory Tree"
echo "============================================"
find "$BASE_DIR/tests" -type d | sort
echo ""
echo "Files:"
find "$BASE_DIR/tests" -type f | sort

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo " RESULTS: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "============================================"

# Cleanup
rm -rf "$BASE_DIR"
echo "Cleaned up: $BASE_DIR"

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo ""
  echo "DIRECTORY STRUCTURE VERIFICATION FAILED!"
  exit 1
else
  echo ""
  echo "DIRECTORY STRUCTURE VERIFICATION PASSED!"
  exit 0
fi
