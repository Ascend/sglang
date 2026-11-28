# NPU 验证流程要点

## 启动前

```bash
# 1. 确认 PYTHONPATH
export PYTHONPATH=${PWD}/python:$PYTHONPATH

# 2. 确认 NPU 可见
npu-smi info | grep "Ascend" | wc -l   # 应 >= --tp 值

# 3. 检查端口未占
ss -tnlp 2>/dev/null | grep 8000

# 4. 用 nohup + tee 保留 log + PID
nohup python -m sglang.launch_server \
    --model-path "<model_path>" \
    --tp <N> \
    --device npu \
    --port 8000 \
    > "${WORKSPACE_DIR}/logs/feature_<name>.log" 2>&1 &
echo $! > "${WORKSPACE_DIR}/logs/server.pid"
```

## 启动后健康检查(三关)

```bash
# 关 1: PID 仍存活
kill -0 $(cat "${WORKSPACE_DIR}/logs/server.pid") || echo "DEAD"

# 关 2: 端口已开
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models | grep -q 200; then
        echo "READY in ${i}*30s"; break
    fi
    sleep 30
done

# 关 3: log 没异常
grep -E "Traceback|RuntimeError|AttributeError|Error code|ACL Error" "${WORKSPACE_DIR}/logs/feature_<name>.log" && echo "LOG HAS ERROR"
```

任一关失败 → 该特性 status=failed,evidence 指向 log。

## 关闭 server(共享 NPU 礼仪)

```bash
PID=$(cat "${WORKSPACE_DIR}/logs/server.pid")
kill ${PID} 2>/dev/null
sleep 5
kill -0 ${PID} 2>/dev/null && kill -9 ${PID}
sleep 2
# 等 NPU 显存真的释放
npu-smi info | grep "Ascend"
```

**重要**:不要 `pkill -f sglang`——会杀掉别人的 server。**只杀自己写到 server.pid 的 PID**。

## 特性测试矩阵典型 flag

| 特性 | 启动 flag(在 launch_command_base 上追加) |
|------|--------------------------------------|
| basic_inference | (无) |
| aclgraph | `--enable-aclgraph`(警告:首次启动慢) |
| deepep | `--enable-deepep`(MoE only) |
| dp_attention | `--enable-dp-attention --dp-size <N>` |
| mtp | `--speculative-algorithm EAGLE3` 或具体的 MTP flag |
| 多模态 | (取决于模型,通常 base 就支持) |
| chunked_prefill | `--chunked-prefill-size 8192` |

每个特性测之前都要**关掉前一次的 server**——NPU 显存不释放就启不来下一个。

## 等待显存释放

```bash
# 简单等
sleep 10

# 严格等
while [ "$(npu-smi info | grep -oE '[0-9]+%' | head -1 | tr -d %)" -gt 5 ]; do
    sleep 5
done
```

## 容量基线

```bash
nohup python -m sglang.launch_server \
    --model-path "<model_path>" \
    --tp <N> \
    --device npu \
    --context-length 131072 \
    --max-running-requests 16 \
    --port 8000 \
    > "${WORKSPACE_DIR}/logs/capacity_run.log" 2>&1 &
echo $! > "${WORKSPACE_DIR}/logs/server.pid"

# 健康检查 → 三关都过后
python -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 8 \
    --random-input-len 4096 \
    --random-output-len 256 \
    2>&1 | tee "${WORKSPACE_DIR}/logs/capacity_128k_bs16.log"
```

OOM 则降到 `--context-length 65536 --max-running-requests 8` 重试,记录实际跑通的配置。

## 超时阈值(单特性测试)

- 启动等待:最多 12 × 30s = 6 分钟
- 推理请求:single short prompt 应 < 30s 响应
- bench_serving:30 分钟硬上限,超时直接 kill,记 failed + `reason: timeout`
