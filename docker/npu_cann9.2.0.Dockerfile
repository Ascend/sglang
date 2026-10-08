ARG CANN_VERSION=9.2.0
# quay.io 上目前只有 beta 镜像（9.2.0-<hw>-ubuntu22.04-py3.12 还不存在），所以把
# "镜像 tag / CANN 安装目录名" 和 "kernel 产物版本号" 分开：
#   CANN_IMAGE_TAG -> FROM 的 tag，以及 /usr/local/Ascend/cann-* 的目录名
#   CANN_VERSION   -> kernel 包名里的 cann9.2.0
# 等官方发布正式 9.2.0 镜像后，把 CANN_IMAGE_TAG 改回 9.2.0 即可。
ARG CANN_IMAGE_TAG=9.2.0-beta.1
ARG DEVICE_TYPE=950
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12
ARG arch

FROM quay.io/ascend/cann:$CANN_IMAGE_TAG-$DEVICE_TYPE-$OS-$PYTHON_VERSION

ARG TARGETARCH
ARG CANN_VERSION
ARG CANN_IMAGE_TAG
ARG DEVICE_TYPE
ARG arch
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
# torch_npu 2.10.0.post5 requires torch==2.10.0, so PYTORCH_VERSION stays at 2.10.0
ARG PYTORCH_VERSION="2.10.0"
ARG TORCHVISION_VERSION="0.25.0"
ARG TORCHAUDIO_VERSION="2.10.0"
# CANN 9.2.0 配套的 torch_npu 是 v26.2.0-beta.1-pytorch2.10.0（2.10.0.post5）。
# 华为 devcloud 的 pip 源上 2.10.0 线只有 9.1.X 配套的 post6，所以这里直接从 gitcode release 装 wheel。
ARG TORCH_NPU_VERSION="2.10.0.post5"
ARG TORCH_NPU_RELEASE="v26.2.0-beta.1-pytorch2.10.0"
ARG SGLANG_TAG=main
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
# kernel 产物来源。2026.10.0.post9 目前只在 fork 上；合入 sglang 前需要改成
# sgl-project/sgl-kernel-npu 并确认该 tag 已在官方仓发布，否则 wget 会 404。
ARG SGLANG_KERNEL_NPU_REPO=huangxiaojun15/sgl-kernel-npu
ARG SGLANG_KERNEL_NPU_TAG=2026.10.0.post9
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG DEVICE_TYPE
ARG MODELSCOPE_VERSION=""
ARG EVALSCOPE_VERSION=""

# memfabric-hybrid / memcache-hybrid version, installed from the pip index (no OBS bucket download)
ARG MF_VERSION="1.2.1"

# memfabric-zbal: 950 与 a3 使用不同版本
ARG ZBAL_VERSION_950="1.2.21004.post1"
ARG ZBAL_VERSION_A3="1.1.3"

# Later RUN steps source /etc/environment_new, so make sure it exists
RUN touch /etc/environment_new

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url $PIP_INDEX_URL
RUN if [ -n "$APTMIRROR" ];then sed -i "s|.*.ubuntu.com|$APTMIRROR|g" /etc/apt/sources.list ;fi

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

### Install memfabric-zbal
RUN if [ "$DEVICE_TYPE" = "950" ]; then ZBAL_PKG="memfabric-zbal==${ZBAL_VERSION_950}"; \
    else ZBAL_PKG="memfabric-zbal==${ZBAL_VERSION_A3}"; fi; \
    ${PIP_INSTALL} "$ZBAL_PKG" -i https://pypi.org/simple/
### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router

### Install PyTorch and PTA
RUN . /etc/environment_new && \
    (${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cpu) \
    && (${PIP_INSTALL} "https://gitcode.com/Ascend/pytorch/releases/download/${TORCH_NPU_RELEASE}/torch_npu-${TORCH_NPU_VERSION}-cp312-cp312-manylinux_2_28_$(arch).whl")

### Install ModelScope & EvalScope
# Installed right after torch/torch-npu so their dependencies resolve against the pinned torch.
# MODELSCOPE_VERSION / EVALSCOPE_VERSION are empty by default -> latest release.
RUN . /etc/environment_new && \
    MS_PKG="modelscope" && \
    ES_PKG="evalscope" && \
    if [ -n "${MODELSCOPE_VERSION}" ]; then MS_PKG="modelscope==${MODELSCOPE_VERSION}"; fi && \
    if [ -n "${EVALSCOPE_VERSION}" ]; then ES_PKG="evalscope==${EVALSCOPE_VERSION}"; fi && \
    ${PIP_INSTALL} "${MS_PKG}" "${ES_PKG}"

## Install triton-ascend
RUN . /etc/environment_new && \
    ${PIP_INSTALL} pybind11 && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        ${PIP_INSTALL} "https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/Triton-ascend/3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1806290330&Signature=eRq3VKjwgP/tTkObtsho%2BzIsmJM%3D"; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        ${PIP_INSTALL} https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl; \
    else \
        echo "Unsupported architecture: $TARGETARCH"; \
        exit 1; \
    fi

# Install SGLang
RUN git clone https://github.com/sgl-project/sglang --branch ${SGLANG_TAG} /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && \
    ${PIP_INSTALL} -v -e .[all_npu]

ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann-${CANN_IMAGE_TAG}

ENV LD_LIBRARY_PATH=/usr/local/Ascend/cann-${CANN_IMAGE_TAG}/lib64:/usr/local/Ascend/cann-${CANN_IMAGE_TAG}/lib:/usr/local/Ascend/cann-${CANN_IMAGE_TAG}/x86_64-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

RUN mkdir cann-custom-ops && \
    cd cann-custom-ops && \
    source /usr/local/Ascend/cann-${CANN_IMAGE_TAG}/set_env.sh && \
    wget https://github.com/${SGLANG_KERNEL_NPU_REPO}/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    wget https://github.com/${SGLANG_KERNEL_NPU_REPO}/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    chmod +x *.run && \
    ./CANN-custom_ops-none-linux.$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_IMAGE_TAG}/opp && \
    ./cann-ops-transformer-custom_linux-$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_IMAGE_TAG}/opp && \
    source /usr/local/Ascend/cann-${CANN_IMAGE_TAG}/opp/vendors/customize/bin/set_env.bash && \
    source /usr/local/Ascend/cann-${CANN_IMAGE_TAG}/opp/vendors/custom_transformer/bin/set_env.bash && \
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    ${PIP_INSTALL} custom_ops-1.0-cp312-cp312-linux_$(arch).whl && \
    cd .. && rm -rf cann-custom-ops

# Install Deep-ep
# pin wheel to 0.45.1 ref: https://github.com/pypa/wheel/issues/662
RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/${SGLANG_KERNEL_NPU_REPO}/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

CMD ["/bin/bash"]