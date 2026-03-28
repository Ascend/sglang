# GLM-4.7-Flash examples

## Environment Preparation

### Model Weight

- `GLM-4.7-Flash`(BF16 version): [Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-4.7-Flash).

### Installation

The dependencies required for the NPU runtime environment have been integrated into a Docker image and uploaded to the quay.io platform. You can directly pull it.

```bash
#Atlas 800 A3
docker pull quay.io/ascend/sglang:main-cann8.5.0-a3
#Atlas 800 A2
docker pull quay.io/ascend/sglang:main-cann8.5.0-910b

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
quay.io/ascend/sglang:${tag}
```

Note: When using this image, you need to update Transformers to version 5.3.0.

``` shell
# reinstall transformers
pip install transformers==5.3.0
```

## Running GLM-4.7-Flash

### Running GLM-4.7-Flash on 1 x Atlas 800I A3.

Run the following script to execute online inference.

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
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --tp-size 2 \
        --attention-backend ascend \
        --device npu \
        --chunked-prefill-size 16384 \
        --max-prefill-tokens 150000 \
        --dtype bfloat16 \
        --max-running-requests 32 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.75 \
        --port 8000 \
        --cuda-graph-bs 1 2 4 8 16 32 \
        --watchdog-timeout 9000
```

Note: TP size is currently limited to 2 or 4.

### Running GLM-4.7-Flash on 1 x Atlas 800I A3 in slime-ascend.

#### Preparation

- [slime-ascend](https://gitcode.com/Ascend/slime-ascend) code

#### Installation

Run the following commands to install sglang. (Please replace '<slime-ascend-root>' with the path to the root directory of the slime codebase.')

```bash
git clone -b v0.5.8 https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_other.toml python/pyproject.toml
pip install -e python[srt_npu]
git checkout . && git checkout sglang-slime
git am <slime-ascend-root>/docker/npu_patch/v0.2.2/sglang/*
```

Note: Make sure you are using Transformers 5.3.0.

#### Execution

Run the following script to execute online **inference**.

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
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --tp-size 2 \
        --attention-backend ascend \
        --device npu \
        --chunked-prefill-size 16384 \
        --max-prefill-tokens 150000 \
        --dtype bfloat16 \
        --max-running-requests 32 \
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.75 \
        --port 8000 \
        --cuda-graph-bs 1 2 4 8 16 32 \
        --watchdog-timeout 9000
```

Refer to [Training and Deployment Example](https://gitcode.com/Ascend/slime-ascend/blob/main/docs/ascend_tutorial/examples/glm4.7-30B-A3B.md) for training and deployment.

### Using Benchmark

Refer to [Benchmark and Profiling](../developer_guide/benchmark_and_profiling.md) for details.
