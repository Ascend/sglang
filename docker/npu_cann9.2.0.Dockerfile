ARG CANN_VERSION=9.2.0
ARG DEVICE_TYPE=950
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

ARG CANN_VERSION
ARG DEVICE_TYPE

ARG PYTORCH_VERSION="2.13.0"
ARG TORCHVISION_VERSION="0.28.0"
ARG TORCHAUDIO_VERSION="2.11.0"
ARG TORCH_NPU_VERSION="2.13.0"
ARG TORCH_NPU_WHEEL_URL_AARCH64="https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_aarch64.whl"
ARG TORCH_NPU_WHEEL_URL_X86_64="https://gitcode.com/Ascend/pytorch/releases/download/v26.2.0-pytorch2.13.0/torch_npu-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl"
ARG TORCH_NPU_INDEX_URL="https://ascend.devcloud.huaweicloud.com/pypi/simple/"

ARG TRITON_AARCH64="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_34_aarch64.whl"
ARG TRITON_X86_64="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/20261105160244/triton_ascend-3.6.0%2Bdev20261105160244-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_34_x86_64.whl"

ARG SGLANG_TAG=refactor/dsv4-cann-transformer-kernels

ARG SGLANG_KERNEL_NPU_TAG=2026.11.0.post4

ARG MF_VERSION="1.3.0"

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
    ${PIP_INSTALL} memfabric-hybrid==${MF_VERSION}; \
    mfcli kernel install --soc-version "$MF_SOC_VERSION"; \
    ${PIP_INSTALL} memcache-hybrid==${MF_VERSION}

### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router


### Install PyTorch and PTA
RUN set -eux; \
    case "$TARGETARCH" in \
      arm64) TORCH_NPU_WHEEL="${TORCH_NPU_WHEEL_URL_AARCH64}" ;; \
      amd64) TORCH_NPU_WHEEL="${TORCH_NPU_WHEEL_URL_X86_64}" ;; \
      *) echo "Unsupported architecture: $TARGETARCH" >&2; \
         exit 1 ;; \
    esac; \
    ${PIP_INSTALL} torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cpu; \
    ${PIP_INSTALL} "${TORCH_NPU_WHEEL}"

RUN ${PIP_INSTALL} modelscope evalscope pytest

## Install triton-ascend
RUN ${PIP_INSTALL} pybind11 && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        ${PIP_INSTALL} "${TRITON_AARCH64}"; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        ${PIP_INSTALL} "${TRITON_X86_64}"; \
    else \
        echo "Unsupported architecture: $TARGETARCH"; \
        exit 1; \
    fi

# Install SGLang
RUN git clone https://github.com/lilDss/sglang --branch ${SGLANG_TAG} /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && \
    ${PIP_INSTALL} -v -e .[all_npu]

ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann-${CANN_VERSION}

ENV LD_LIBRARY_PATH=/usr/local/Ascend/cann-${CANN_VERSION}/lib64:/usr/local/Ascend/cann-${CANN_VERSION}/lib:/usr/local/Ascend/cann-${CANN_VERSION}/x86_64-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

RUN sed -Ei -e '/^[[:space:]]*"torch\.npu\.random\.manual_seed(_all)?",[[:space:]]*$/d' -e 's/^([[:space:]]*)"torch\.npu\.set_device",/\1"torch.npu.random.manual_seed_all",\n\1"torch.npu.random.manual_seed",\n\1"torch.npu.set_device",/' /usr/local/python3.12.15/lib/python3.12/site-packages/torch_npu/dynamo/trace_rule.py

RUN cp /usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libmmpa.so /usr/lib64/

RUN mkdir cann-custom-ops && \
    cd cann-custom-ops && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh && \
    wget https://github.com/lilDss/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.13.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    wget https://github.com/lilDss/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.13.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.13.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.13.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
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
    && wget https://github.com/lilDss/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.13.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.13.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

CMD ["/bin/bash"]
