ARG CANN_VERSION=8.5.0
ARG DEVICE_TYPE=a3
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.11

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

ARG TARGETARCH
ARG CANN_VERSION
ARG DEVICE_TYPE
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
ARG PYTORCH_VERSION="2.8.0"
ARG TORCHVISION_VERSION="0.23.0"
ARG TORCH_NPU_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/torch_npu-2.8.0.post2%2Bgitdef4a1c-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG SGLANG_ZIP_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/sglang-pri-zyj-dev_20260416.zip"
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=main

ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG DEVICE_TYPE

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url $PIP_INDEX_URL
RUN if [ -n "$APTMIRROR" ]; then sed -i "s|.*.ubuntu.com|$APTMIRROR|g" /etc/apt/sources.list; fi

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


### Install MemFabric
RUN ${PIP_INSTALL} memfabric-hybrid==1.0.5
### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router

RUN ${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu \
    && ${PIP_INSTALL} "${TORCH_NPU_URL}"


RUN ${PIP_INSTALL} pybind11 triton-ascend

RUN wget -O /tmp/sglang.zip "${SGLANG_ZIP_URL}" \
    && unzip -q /tmp/sglang.zip -d /tmp/sglang-src \
    && cd /tmp/sglang-src/* \
    && cd python \
    && rm -f pyproject.toml \
    && mv pyproject_npu.toml pyproject.toml \
    && export SETUPTOOLS_SCM_PRETEND_VERSION=v0.5.10a2 \
    && ${PIP_INSTALL} -v .[all_npu] \
    && rm -rf /tmp/sglang.zip /tmp/sglang-src

RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

ARG CUSTOM_OPS_RUN_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/${DEVICE_TYPE}/CANN-custom_ops--linux.aarch64.run"
ARG CUSTOM_OPS_WHL_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/${DEVICE_TYPE}/custom_ops-1.0-cp311-cp311-linux_aarch64.whl"
ARG TRANSFORMER_RUN_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/${DEVICE_TYPE}/cann-ops-transformer-custom_linux-aarch64.run"

RUN mkdir -p /tmp/ascend_ops && cd /tmp/ascend_ops \
    && wget -O CANN-custom_ops.run "${CUSTOM_OPS_RUN_URL}" \
    && wget -O cann-ops-transformer.run "${TRANSFORMER_RUN_URL}" \
    && chmod +x CANN-custom_ops.run \
    && ./CANN-custom_ops.run --quiet --install-path=${ASCEND_CANN_PATH}/latest/opp \
    && chmod +x cann-ops-transformer.run \
    && ./cann-ops-transformer.run --quiet --install-path=${ASCEND_CANN_PATH}/latest/opp \
    && ${PIP_INSTALL} ${CUSTOM_OPS_WHL_URL} \
    && cd / && rm -rf /tmp/ascend_ops

RUN echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/customize/bin/set_env.bash" >> /etc/profile\
    && echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/custom_transformer/bin/set_env.bash" >> /etc/profile\
    && echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/customize/bin/set_env.bash" >>  ~/.bashrc \
    && echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/custom_transformer/bin/set_env.bash" >>  ~/.bashrc

CMD ["/bin/bash"]
