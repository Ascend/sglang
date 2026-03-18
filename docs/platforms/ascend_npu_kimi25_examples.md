## Kimi-K2.5 examples

## Environment Preparation

### Installation

The dependencies required for the NPU runtime environment have been integrated into a Docker image. You can directly pull it.

```{code-block}
#Atlas 800 A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:0.5.9.rc1-npu-a3

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
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:0.5.9.rc1-npu-a3
```



### Running Kimi-K2.5

#### Running Kimi-K2.5 on 1 x Atlas 800I A3.

- Model weights could be found [here](https://modelscope.cn/models/Eco-Tech/Kimi-K2.5-w4a8)
- Speculative model weights could be found [here](https://huggingface.co/AQ-MedAI/Kimi-K25-eagle3)
- The following command is recommended for **LLM text input scenarios with short sequence cases**.

```shell
# System Settings
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

# Deepep communication settings
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=2100
# spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
# scheduler optimize
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

python -m sglang.launch_server --skip-server-warmup \
    --model-path /xxx/Kimi-K2.5-w4a8 --quantization modelslim --dtype bfloat16 \
    --host 127.0.0.1 --port 8100 \
    --trust-remote-code --device npu --attention-backend ascend \
    --tp-size 16 --mem-fraction-static 0.77 --max-running-requests 96 \
    --chunked-prefill-size 65536 --context-length 8192 --max-prefill-tokens 16384 \
    --enable-dp-attention --dp-size 16 --moe-a2a-backend deepep --deepep-mode auto \
    --cuda-graph-bs 1 2 4 6 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /xxx/Kimi-K25-eagle3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant
```



#### Running Kimi-K2.5 on 1 x Atlas 800I A3.

- Model weights could be found [here](https://modelscope.cn/models/Eco-Tech/Kimi-K2.5-w4a8)
- The following command is recommended for **multimodal image input scenarios**.

```shell
# System Settings
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

# Deepep communication settings
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=2100
# spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
# scheduler optimize
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

python -m sglang.launch_server --skip-server-warmup  \
    --model-path /xxx/Kimi-K2.5-w4a8 --quantization modelslim --dtype bfloat16 \
    --host 127.0.0.1 --port 8100 \
    --trust-remote-code --device npu --attention-backend ascend \
    --tp-size 16 --mem-fraction-static 0.82 --max-running-requests 128 \
    --chunked-prefill-size 32768 --context-length 8192 --max-prefill-tokens 16384 \
    --enable-multimodal --mm-attention-backend ascend_attn --sampling-backend ascend \
    --enable-dp-attention --dp-size 16 --enable-dp-lm-head \
    --moe-a2a-backend deepep --deepep-mode auto \
    --cuda-graph-bs 1 2 4 8
```



#### Running Kimi-K2.5 on 2 x Atlas 800I A3.

- Model weights could be found [here](https://modelscope.cn/models/Eco-Tech/Kimi-K2.5-w4a8)
- The following command is recommended for **LLM long sequence** and **multimodal image input scenarios**.

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#Deepep communication settings
export HCCL_BUFFSIZE=3072
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=88
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

#spec overlap
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1

export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=200

MIX_IP=('node1_ip' 'node2_ip')
MODEL_PATH=/xxx/Kimi-K2.5-w4a8

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!MIX_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${MIX_IP[$i]}" || "$LOCAL_HOST2" == "${MIX_IP[$i]}" ]];
    then
        echo "${MIX_IP[$i]}"

		export HCCL_SOCKET_IFNAME=xxx
		export GLOO_SOCKET_IFNAME=xxx

        python -m sglang.launch_server --model-path ${MODEL_PATH} \
		--host ${MIX_IP[$i]} --port 8100 --skip-server-warmup \
        --trust-remote-code --dist-init-addr ${MIX_IP[0]}:5000 --nnodes 2 --node-rank $i \
        --tp-size 32 --mem-fraction-static 0.66 --attention-backend ascend --device npu \
        --max-running-requests 384 --context-length 160000 \
		--chunked-prefill-size 132000 --max-prefill-tokens 32768 \
        --dp-size 32 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --enable-dp-lm-head \
        --enable-multimodal --mm-attention-backend ascend_attn --sampling-backend ascend \
		--disable-shared-experts-fusion --cuda-graph-bs 1 2 4 8 12 \
		--tokenizer-worker-num 4 --dtype bfloat16

        exit 1
    fi
done
```

