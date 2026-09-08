#!/bin/bash
# ==============================================
# DeepSeek-V4-Flash W8A8 单机 16 卡服务启动脚本（CI 适配版）
# 源自 .claude/2.sh（手动验证可正常拉起服务、graph capture 不崩溃的版本），
# 适配 CI 容器环境：
#   - 模型路径 / 端口 / IP 通过环境变量或位置参数覆盖
#   - sysctl 等调优命令失败不阻断（CI 容器可能无权限）
#   - nohup 后台启动 + 健康检查轮询 + PID 文件（供 CI 清理步骤 kill）
#   - 不硬编码 PYTHONPATH，默认用镜像预装的 sglang；
#     如需用源码版启动，由调用方在运行前 export PYTHONPATH=<源码路径>/python
# 用法:
#   bash scripts/ci/npu/launch_dsv4_flash_w8a8_server.sh [port]
# 产物:
#   /tmp/dsv4_flash_server.pid  — 服务主进程 PID，供 CI 清理步骤 kill
#   /tmp/dsv4_flash_server.log  — 服务完整日志
# ==============================================
set -uo pipefail

PORT="${1:-${SGLANG_PORT:-30000}}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}"
IP="${SGLANG_HOST_IP:-0.0.0.0}"
PID_FILE="/tmp/dsv4_flash_server.pid"
LOG_FILE="/tmp/dsv4_flash_server.log"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
# 16 卡加载权重 + graph capture 较慢，健康检查超时默认给足余量（秒）
WAIT_TIMEOUT="${WAIT_TIMEOUT:-3600}"

# ---------- 清理代理（避免拉起进程访问外网） ----------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy || true

# ---------- 系统性能调优（容器内可能无权限，失败不阻断） ----------
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
sysctl -w vm.swappiness=0 2>/dev/null || true
sysctl -w kernel.numa_balancing=0 2>/dev/null || true

# ---------- 加载昇腾环境（缺失的组件跳过） ----------
source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
source /usr/local/Ascend/nnal/atb/set_env.sh || true
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash || true
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash || true

# ========== 环境变量（与 .claude/2.sh 完全一致，即测试用例 DEEPSEEK_V4_FLASH_W8A8_8P_ENVS） ==========
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV

# deepep 相关
export DEEPEP_HCCL_BUFFSIZE=1000
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

# 跳过 GPU 分支优化
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False

# MTP (EAGLE) 相关
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# 额外稳定 / 性能环境变量
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35

echo "MODEL_PATH=${MODEL_PATH}"
echo "IP=${IP} PORT=${PORT}"
echo "PYTHONPATH=${PYTHONPATH:-<not set, using pre-installed sglang>}"

# ---------- 后台启动服务 ----------
# 注意：这里用 python3 -m sglang.launch_server（与 2.sh 一致），
# 而非 CI 框架的入口 —— 这是本次验证的核心差异点之一。
nohup python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host "${IP}" \
    --port "${PORT}" \
    --mem-fraction-static 0.68 \
    --prefill-max-requests 2 \
    --disable-radix-cache \
    --chunked-prefill-size 131072 \
    --max-running-requests 160 \
    --dp-size 16 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --quantization modelslim \
    --enable-dp-lm-head \
    --kv-cache-dtype bfloat16 \
    --cuda-graph-bs 1 2 4 8 10 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 3 \
    > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "Server started, pid=${SERVER_PID}, log=${LOG_FILE}"

# ---------- 健康检查轮询 ----------
start=$(date +%s)
while true; do
    # 进程退出视为启动失败，直接打印日志尾部并报错
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server process exited during startup. Last 80 log lines:"
        tail -n 80 "${LOG_FILE}" || true
        exit 1
    fi
    if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "Server is ready at ${HEALTH_URL}"
        break
    fi
    now=$(date +%s)
    if [ $((now - start)) -gt "${WAIT_TIMEOUT}" ]; then
        echo "ERROR: timeout waiting for server health after ${WAIT_TIMEOUT}s. Last 80 log lines:"
        tail -n 80 "${LOG_FILE}" || true
        exit 1
    fi
    sleep 15
done
