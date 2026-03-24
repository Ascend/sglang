# Ascend KV Transfer for Qwen3 MoE PD+PCP

This document describes Qwen3 MoE support for PD disaggregation with prefill-side context parallelism (referred to as PCP in this document) and Ascend KV transfer in SGLang.

## What This Feature Is

This feature supports the following execution path:

- the prefill worker runs Qwen3 MoE with prefill-side CP enabled
- the prefill worker generates KV cache with PCP-aware layout
- KV cache is transferred from prefill to decode through the PD path
- the decode worker remaps pages and continues generation

This is not a generic PD transfer path. It is a **Qwen3 MoE KV transfer path for prefill-side CP**.

In this setup, the KV cache generated on the prefill side is not stored in a normal contiguous layout. KV allocation, KV writes, and KV transfer all follow the PCP block layout. The decode side must restore the KV view with the same layout semantics before it can continue decoding.

The goal of this feature is to make the full path work on Ascend: **prefill-side CP -> PD transfer -> decode continuation**.

## Target Scenario

This feature is intended for the following setup:

- the model is Qwen3 MoE
- the hardware platform is Ascend NPU
- the deployment mode is PD disaggregation
- prefill-side CP is enabled
- KV cache generated on the prefill side must be transferred to the decode side

This feature is not intended for:

- non-Qwen3 MoE models
- non-Ascend deployments
- unified execution without PD disaggregation
- standard prefill paths without PCP

In short, this feature targets the **Qwen3 MoE + Ascend + PD + PCP** combination.

## Architecture

At a high level, this feature has four stages:

1. **Build the PCP view on the prefill side**

   The scheduler builds PCP metadata for the request. KV allocation, KV writes, and attention computation all use this metadata.

2. **Generate and write KV on the prefill side**

   Qwen3 MoE generates KV in the PCP path. KV is written into `req_to_token` and the underlying pages with PCP block layout instead of a normal contiguous layout.

3. **Transfer KV from prefill to decode**

   The disaggregation layer handles bootstrap, rank mapping, and transfer request management. In the PCP case, the transfer carries both KV data and the block/page ordering implied by the prefill layout.

4. **Remap on the decode side and continue execution**

   The decode worker restores the logical page order from the PCP metadata, rebuilds the local KV view and attention inputs, and then continues Qwen3 MoE decoding.

The key requirement is that **the same PCP metadata must be used consistently across allocation, transfer, and computation**.

## Requirements and Limitations

The following table summarizes the current requirements and limitations.

| Item | Status | Notes |
|------|--------|-------|
| Model architecture | Supported | Qwen3 MoE only |
| Hardware platform | Supported | Ascend NPU only |
| Attention backend | Supported | `ascend` is required |
| Transfer backend | Supported | `ascend` is required |
| Deployment mode | Supported | PD disaggregation is required |
| Prefill-side CP | Supported | Enabled with `--attn-cp-size` on the prefill side |
| Decode continuation after KV transfer | Supported | Requires the same PCP layout semantics across prefill, transfer, and decode |
| Radix cache | Not supported | Must be disabled |
| Chunked prefill | Not supported | Must be disabled with `--chunked-prefill-size -1` |
| Prefill batch size > 1 | Not supported | The prefill side currently requires `batch_size = 1` |
| Data parallelism on the prefill side | Not supported | The prefill side does not support DP in this path |
| Unified execution without PD | Not supported | This feature is designed for PD disaggregation |

## Usage

### Single-Node Example

The following example shows the basic single-node setup for Qwen3 MoE PD+CP with Ascend KV transfer.

First, set the shared environment variables:

```bash
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH

export MODEL_PATH=/path/to/Qwen3-30B-A3B

export PREFILL_HOST=127.0.0.1
export PREFILL_PORT=8231
export PREFILL_BOOTSTRAP_PORT=8995

export DECODE_HOST=127.0.0.1
export DECODE_PORT=8232

export ROUTER_HOST=127.0.0.1
export ROUTER_PORT=6689
export ROUTER_PROM_PORT=29010

export ASCEND_MF_STORE_URL="tcp://<store_host>:12345"
export ASCEND_USE_FIA=True
```

Start the prefill worker:

```bash
export SGLANG_SET_CPU_AFFINITY=1

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port ${PREFILL_BOOTSTRAP_PORT} \
    --attention-backend ascend \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --skip-server-warmup \
    --device npu \
    --base-gpu-id 2 \
    --tp-size 4 \
    --attn-cp-size 2 \
    --max-running-requests 1 \
    --host ${PREFILL_HOST} \
    --port ${PREFILL_PORT}
```

Start the decode worker:

```bash
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --mem-fraction-static 0.8 \
    --disable-cuda-graph \
    --device npu \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --skip-server-warmup \
    --base-gpu-id 12 \
    --tp-size 2 \
    --max-running-requests 32 \
    --host ${DECODE_HOST} \
    --port ${DECODE_PORT}
```

Start the router:

```bash
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://${PREFILL_HOST}:${PREFILL_PORT} ${PREFILL_BOOTSTRAP_PORT} \
    --decode http://${DECODE_HOST}:${DECODE_PORT} \
    --host ${ROUTER_HOST} \
    --port ${ROUTER_PORT} \
    --prometheus-port ${ROUTER_PROM_PORT}
```

### Multi-Node Example

The following example shows the basic setup where the prefill worker and decode worker run on different nodes. In this example, the prefill side and decode side each use single-node TP. If you need multi-node execution on either side, continue setting `--nnodes`, `--node-rank`, and `--dist-init-addr` as needed.

First, set the shared environment variables:

```bash
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH

export MODEL_PATH=/path/to/Qwen3-235B-A22B-Instruct
export QUANTIZATION=modelslim

export PREFILL_HOST=<prefill_host>
export PREFILL_PORT=8000
export PREFILL_BOOTSTRAP_PORT=8995
export PREFILL_DIST_INIT_ADDR=${PREFILL_HOST}:6688

export DECODE_HOST=<decode_host>
export DECODE_PORT=8001
export DECODE_DIST_INIT_ADDR=${DECODE_HOST}:6688

export ROUTER_HOST=127.0.0.1
export ROUTER_PORT=6689
export ROUTER_PROM_PORT=29010

export ASCEND_MF_STORE_URL="tcp://<store_host>:12345"
export ASCEND_USE_FIA=True
```

Start the prefill worker on the prefill node:

```bash
export SGLANG_SET_CPU_AFFINITY=1

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --quantization ${QUANTIZATION} \
    --disaggregation-bootstrap-port ${PREFILL_BOOTSTRAP_PORT} \
    --attention-backend ascend \
    --disable-radix-cache \
    --mem-fraction-static 0.7 \
    --chunked-prefill-size -1 \
    --skip-server-warmup \
    --device npu \
    --base-gpu-id 0 \
    --tp-size 16 \
    --attn-cp-size 2 \
    --max-running-requests 1 \
    --host ${PREFILL_HOST} \
    --port ${PREFILL_PORT} \
    --nnodes 1 \
    --node-rank 0 \
    --dist-init-addr ${PREFILL_DIST_INIT_ADDR}
```

Start the decode worker on the decode node:

```bash
python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --quantization ${QUANTIZATION} \
    --attention-backend ascend \
    --disable-radix-cache \
    --mem-fraction-static 0.7 \
    --disable-cuda-graph \
    --chunked-prefill-size -1 \
    --skip-server-warmup \
    --device npu \
    --base-gpu-id 8 \
    --tp-size 8 \
    --max-running-requests 32 \
    --host ${DECODE_HOST} \
    --port ${DECODE_PORT} \
    --nnodes 1 \
    --node-rank 0 \
    --dist-init-addr ${DECODE_DIST_INIT_ADDR}
```

Start the router:

```bash
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy cache_aware \
    --prefill http://${PREFILL_HOST}:${PREFILL_PORT} ${PREFILL_BOOTSTRAP_PORT} \
    --decode http://${DECODE_HOST}:${DECODE_PORT} \
    --host ${ROUTER_HOST} \
    --port ${ROUTER_PORT} \
    --prometheus-port ${ROUTER_PROM_PORT}
```

### Request Example

After the service is ready, you can send a request through the router:

```bash
curl -X POST http://${ROUTER_HOST}:${ROUTER_PORT}/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Two people start from the same location. A leaves 1 hour earlier at speed 5, and B leaves later at speed 6. How long after A starts will they meet?",
    "sampling_params": {
      "temperature": 0.6,
      "max_new_tokens": 512
    }
  }'
```

### Key Arguments

- `--disaggregation-mode`
  Sets the worker role. Use `prefill` on the prefill side and `decode` on the decode side.

- `--disaggregation-transfer-backend`
  Sets the PD transfer backend. This feature requires `ascend`.

- `--attn-cp-size`
  Sets the context parallel size on the prefill side. This controls whether prefill-side CP is enabled and how many CP ranks are used.

- `--disable-radix-cache`
  Required by the current implementation.

- `--chunked-prefill-size -1`
  Required by the current implementation.

- `--max-running-requests 1`
  Required on the prefill side because the current implementation requires `batch_size = 1`.
