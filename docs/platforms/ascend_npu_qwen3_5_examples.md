# Qwen3.5 examples

## Environment Preparation

### Installation

The dependencies required for the NPU runtime environment have been integrated into a Docker image and uploaded to the Huawei Cloud Platform. You can directly pull it.

```{code-block} bash
#Atlas 800 A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:0.5.9.rc1-npu-a3
#Atlas 800 A2
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:0.5.9.rc1-npu-910b

#start container
docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
--privileged=true --net=host \
-v /var/queue_schedule:/var/queue_schedule \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
--device=/dev/davinci0:/dev/davinci0  \
--device=/dev/davinci1:/dev/davinci1  \
--device=/dev/davinci2:/dev/davinci2  \
--device=/dev/davinci3:/dev/davinci3  \
--device=/dev/davinci4:/dev/davinci4  \
--device=/dev/davinci5:/dev/davinci5  \
--device=/dev/davinci6:/dev/davinci6  \
--device=/dev/davinci7:/dev/davinci7  \
--device=/dev/davinci8:/dev/davinci8  \
--device=/dev/davinci9:/dev/davinci9  \
--device=/dev/davinci10:/dev/davinci10  \
--device=/dev/davinci11:/dev/davinci11  \
--device=/dev/davinci12:/dev/davinci12  \
--device=/dev/davinci13:/dev/davinci13  \
--device=/dev/davinci14:/dev/davinci14  \
--device=/dev/davinci15:/dev/davinci15  \
--device=/dev/davinci_manager:/dev/davinci_manager \
--device=/dev/hisi_hdc:/dev/hisi_hdc \
--entrypoint=bash \
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:{tag}
```

## Deployment

### Single-node Deployment

Run the following script to execute online inference. The Qwen3.5 397B moe model and Qwen3.5 27B dense model are used as examples.

#### Qwen3.5 397B

#####
Model: Qwen3.5 397B W4A8

Hardware: Atlas 800I A3 8Card

DeployMode: PD Hybrid

Dataset: Random

Input Output Length: 3.5K+1.5K

TPOT: 50ms

```shell
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_USE_FIA=1
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=3000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 16 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size -1 --max-prefill-tokens 14336 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 127.0.0.1 --max-running-requests 192 \
        --mem-fraction-static 0.68 \
        --port 8000 \
        --cuda-graph-bs 2 4 6 8 10 12 14 16 18 20 22 24 \
        --quantization modelslim \
        --enable-multimodal --moe-a2a-backend deepep --deepep-mode auto \
        --mm-attention-backend ascend_attn --max-total-tokens 130000 \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 --speculative-draft-model-quantization unquant --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

#####
Model: Qwen3.5 397B W4A8

Hardware: Atlas 800I A3 8Card

DeployMode: PD Hybrid

Dataset: Random

Input Output Length: 3.5K+1.5K

TPOT: 20ms

```shell
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_USE_FIA=1
export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL_BUFFSIZE=2500
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 16 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size -1 --max-prefill-tokens 7168 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 127.0.0.1 --max-running-requests 256 \
        --mem-fraction-static 0.85 \
        --port 8000 \
        --cuda-graph-bs 1 2 4 8 9 10 11 12 13 14 15 16 \
        --quantization modelslim \
        --enable-multimodal --moe-a2a-backend deepep --deepep-mode auto \
        --mm-attention-backend ascend_attn --max-total-tokens 100000 \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 --speculative-draft-model-quantization unquant \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

#### Qwen3.5 27B

#####
Model: Qwen3.5 27B W8A8

Hardware: Atlas 800I A3 8Card

DeployMode: PD Hybrid

Dataset: Random

Input Output Length: 3.5K+1.5K

TPOT: 50&20ms

```shell
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd /home/hexq/mtp/sglang_project
export PYTHONPATH=${PWD}/python:$PYTHONPATH

export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=3000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_PROFILING_STAGE="prefill"
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:24669"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 4 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size -1 --max-prefill-tokens 28672 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 127.0.0.1 --max-running-requests 60 --max-mamba-cache-size 60 \
        --mem-fraction-static 0.9 \
        --port 8000 \
        --cuda-graph-bs 4 8 12 16 20 28 32 36 40 44 48 52 56 60 \
        --enable-multimodal \
        --quantization modelslim \
        --mm-attention-backend ascend_attn \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 --max-total-tokens 310000 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

### Prefill-Decode Disaggregation
Running Example For Qwen3.5 27B W8A8

Run Prefill
```shell

# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
export ASCEND_LAUNCH_BLOCKING=1
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=3000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:24669"

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --quantization modelslim \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --attention-backend ascend \
        --device npu \
        --tp-size 4 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size -1 --max-prefill-tokens 28472 \
        --disable-radix-cache \
        --trust-remote-code --port 10000 \
        --host 127.0.0.1 --max-running-requests 32 --max-mamba-cache-size 32 \
        --mem-fraction-static 0.9 \
        --cuda-graph-bs 4 8 12 16 20 28 32 \
        --enable-multimodal \
        --mm-attention-backend ascend_attn --max-total-tokens 310000 \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 --disaggregation-mode prefill --disaggregation-transfer-backend ascend

```
Run Decode
```shell
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=3000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export DEEPEP_NORMAL_LONG_SEQ_ROUND=32
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:24669"
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 4 --nnodes 1 --node-rank 0 \
        --chunked-prefill-size -1 --max-prefill-tokens 28672 \
        --disable-radix-cache \
        --trust-remote-code \
        --host 127.0.0.1 --max-running-requests 32 --max-mamba-cache-size 32 \
        --mem-fraction-static 0.9 \
        --port 10001 \
        --cuda-graph-bs 4 8 12 16 20 28 32 \
        --enable-multimodal \
        --quantization modelslim \
        --mm-attention-backend ascend_attn \
        --dtype bfloat16 --mamba-ssm-dtype bfloat16 --max-total-tokens 310000 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --disaggregation-mode decode --disaggregation-transfer-backend ascend
```

### Using Benchmark

Refer to [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md) for details.
