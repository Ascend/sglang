#关于基础镜像版本配套
ARG CANN_VERSION=9.2.0
ARG DEVICE_TYPE=950
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

#重新引用对应配套
ARG CANN_VERSION
ARG DEVICE_TYPE

#PTA版本配套，pytorch属于框架本体，用于训练模型；torch_npu属于设备后端扩展，将算子和设备资源接入 PyTorch；pta和torch_npu是同一种东西的不同叫法；
#具体的版本配套可以到这里看 https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md
ARG PYTORCH_VERSION="2.13.0"             #pytorch版本，框架版本
ARG TORCHVISION_VERSION="0.28.0"         #视觉库版本，torch 的次版本号 + 15
ARG TORCHAUDIO_VERSION="2.11.0"          #音频库版本，与pytorch版本保持一致
ARG TORCH_NPU_VERSION="2.10.0.post6"     #torch_npu版本，设备后端扩展版本
ARG TORCH_NPU_INDEX_URL="https://ascend.devcloud.huaweicloud.com/pypi/simple/"   #这个源是用来下载torch_npu的包的，和前面的pytorch使用的不是同一个源

#sglang对应分支
ARG SGLANG_TAG=main

#kernel算子包对应版本
ARG SGLANG_KERNEL_NPU_TAG=2026.9.0.post10

#memfabric-hybrid / memcache-hybrid的版本
ARG MF_VERSION="1.3.0"
ARG MF_WHEEL_URL_AARCH64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/release/1.3/20261019.10/memfabric_hybrid-1.3.0-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
ARG MF_WHEEL_URL_X86_64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/release/1.3/20261019.10/memfabric_hybrid-1.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
ARG MC_WHEEL_URL_AARCH64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/release/1.3/20261019.10/memcache_hybrid-1.3.0-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
ARG MC_WHEEL_URL_X86_64="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/release/1.3/20261019.10/memcache_hybrid-1.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG TARGETARCH

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url $PIP_INDEX_URL

# Install development tools and utilities
RUN apt-get update -y && apt upgrade -y && apt-get install -y \
    unzip \
    build-essential \
    cmake \
    vim \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    openssl \
    libssl-dev \
    pkg-config \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ca-certificates \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

### Install MemFabric and MemCache
RUN set -eux; \
    case "$DEVICE_TYPE" in \
      950) MF_SOC_VERSION="A5" ;; \
      a3)  MF_SOC_VERSION="A3" ;; \
      *)   echo "Unsupported DEVICE_TYPE for mfcli kernel install: $DEVICE_TYPE" >&2; \
           exit 1 ;; \
    esac; \
    case "$TARGETARCH" in \
      arm64) MF_WHEEL_URL="${MF_WHEEL_URL_AARCH64}"; \
             MC_WHEEL_URL="${MC_WHEEL_URL_AARCH64}" ;; \
      amd64) MF_WHEEL_URL="${MF_WHEEL_URL_X86_64}"; \
             MC_WHEEL_URL="${MC_WHEEL_URL_X86_64}" ;; \
      *)     echo "Unsupported architecture for MemFabric install: $TARGETARCH" >&2; \
             exit 1 ;; \
    esac; \
    ${PIP_INSTALL} "${MF_WHEEL_URL}"; \
    mfcli kernel install --soc-version "$MF_SOC_VERSION"; \
    ${PIP_INSTALL} "${MC_WHEEL_URL}"

### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router


### Install PyTorch and PTA
RUN (${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cpu) \
    && (${PIP_INSTALL} torch-npu==${TORCH_NPU_VERSION} --extra-index-url ${TORCH_NPU_INDEX_URL})

RUN ${PIP_INSTALL} modelscope evalscope

## Install triton-ascend
RUN ${PIP_INSTALL} pybind11 && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        ${PIP_INSTALL} https://github.com/triton-lang/triton-ascend/releases/download/v3.6.0/triton_ascend-3.6.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        ${PIP_INSTALL} https://github.com/triton-lang/triton-ascend/releases/download/v3.6.0/triton_ascend-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl; \
    else \
        echo "Unsupported architecture: $TARGETARCH"; \
        exit 1; \
    fi

# Install SGLang
RUN git clone https://github.com/sgl-project/sglang --branch ${SGLANG_TAG} /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && \
    ${PIP_INSTALL} -v -e .[all_npu]

ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann-${CANN_VERSION}

ENV LD_LIBRARY_PATH=/usr/local/Ascend/cann-${CANN_VERSION}/lib64:/usr/local/Ascend/cann-${CANN_VERSION}/lib:/usr/local/Ascend/cann-${CANN_VERSION}/x86_64-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH}


RUN mkdir cann-custom-ops && \
    cd cann-custom-ops && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh && \
    wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    chmod +x *.run && \
    ./CANN-custom_ops-none-linux.$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp && \
    ./cann-ops-transformer-custom_linux-$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/customize/bin/set_env.bash && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/custom_transformer/bin/set_env.bash && \
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    ${PIP_INSTALL} custom_ops-1.0-cp312-cp312-linux_$(arch).whl && \
    cd .. && rm -rf cann-custom-ops

# Install Deep-ep
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

CMD ["/bin/bash"]
