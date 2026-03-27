## Qwen3 examples

### Running Qwen3

#### Running Qwen3-32B on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-32B)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-32B \
   --mem-fraction-static 0.8
```

#### Running Qwen3-32B on 1 x Atlas 800I A3 with Qwen3-32B-Eagle3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-32B)

Speculative model weights could be found [here](https://huggingface.co/Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-32B \
   --mem-fraction-static 0.8 \
   --speculative-algorithm EAGLE3 \
   --speculative-draft-model-path Qwen/Qwen3-32B-Eagle3 \
   --speculative-num-steps 1 \
   --speculative-eagle-topk 1 \
   --speculative-num-draft-tokens 2
```

#### Running Qwen3-30B-A3B MOE on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-30B-A3B)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1

python -m sglang.launch_server \
   --device npu \
   --attention-backend ascend \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-30B-A3B \
   --mem-fraction-static 0.8
```

#### Running Qwen3-235B-A22B-Instruct-2507 MOE on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1

python -m sglang.launch_server \
   --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
   --tp-size 16 \
   --trust-remote-code \
   --attention-backend ascend \
   --device npu \
   --watchdog-timeout 9000 \
   --mem-fraction-static 0.8
```

#### Running Qwen3 MOE with Prefill Context Parallel (PCP) on Ascend NPU

For Qwen3 MoE models, you can enable **Prefill Context Parallel (PCP)** to split the context across devices during the prefill phase, which improves TTFT (Time To First Token) performance for long sequences.

Use the `--attn-cp-size` argument to set the context parallelism size.

> **Important constraints:**
> - The prefill instance only supports **batch_size=1** when PCP is enabled.
> - PCP requires **PD disaggregation** (Prefill/Decode separated deployment) to produce correct inference results. See [PD Disaggregation](../advanced_features/pd_disaggregation.md) for details.

Example: Running Qwen3-235B-A22B-Instruct-2507 with PCP on 1 x Atlas 800I A3 (PD disaggregation with Ascend transfer backend):

**Prefill Instance:**

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1
export ASCEND_USE_FIA=1

python -m sglang.launch_server \
   --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
   --tp-size 16 \
   --attn-cp-size 4 \
   --trust-remote-code \
   --attention-backend ascend \
   --device npu \
   --watchdog-timeout 9000 \
   --mem-fraction-static 0.8 \
   --max-running-requests 1 \
   --disable-radix-cache \
   --disaggregation-transfer-backend ascend \
   --disaggregation-mode prefill \
   --port 30000
```

**Decode Instance:**

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DEEPEP_BF16_DISPATCH=1

python -m sglang.launch_server \
   --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
   --tp-size 16 \
   --trust-remote-code \
   --attention-backend ascend \
   --device npu \
   --watchdog-timeout 9000 \
   --mem-fraction-static 0.8 \
   --disaggregation-transfer-backend ascend \
   --disaggregation-mode decode \
   --port 30001
```

**Router:**

```shell
python -m sglang_router.launch_router \
   --pd-disaggregation \
   --prefill http://127.0.0.1:30000 \
   --decode http://127.0.0.1:30001 \
   --host 0.0.0.0 --port 8000
```

#### Running Qwen3-VL-8B-Instruct on 1 x Atlas 800I A3.

Model weights could be found [here](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

```shell
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV

python -m sglang.launch_server \
   --enable-multimodal \
   --attention-backend ascend \
   --mm-attention-backend ascend_attn \
   --trust-remote-code \
   --tp-size 4 \
   --model-path Qwen/Qwen3-VL-8B-Instruct \
   --mem-fraction-static 0.8
```
