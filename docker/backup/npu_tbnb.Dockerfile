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
ARG SGLANG_ZIP_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/sglang-pri-final_20260413.zip"
ARG SGLANG_KERNEL_NPU_TAG=main

ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

# 配置 pip 和 apt 镜像
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

### 安装 PyTorch 和通用 torch_npu
RUN ${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu \
    && ${PIP_INSTALL} "${TORCH_NPU_URL}"

### 安装 triton-ascend
RUN ${PIP_INSTALL} pybind11 triton-ascend

### 下载 SGLang 源码 ZIP 并安装（替代 git clone）
RUN wget -O /tmp/sglang.zip "${SGLANG_ZIP_URL}" \
    && unzip -q /tmp/sglang.zip -d /tmp/sglang-src \
    && cd /tmp/sglang-src/* \
    && cd python \
    && rm -f pyproject.toml \
    && mv pyproject_npu.toml pyproject.toml \
    && export SETUPTOOLS_SCM_PRETEND_VERSION=v0.5.10a1 \
    && ${PIP_INSTALL} -v .[all_npu] \
    && rm -rf /tmp/sglang.zip /tmp/sglang-src

### 安装 sgl-kernel-npu 和 deep-ep
RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

### ---------- 新增：A2 / A3 特定算子包安装 ----------
# 定义各包下载 URL（按设备类型）
# 通用 custom_ops whl 名称（A2/A3 不同 URL，但本地文件名统一处理）
ARG A2_CUSTOM_OPS_RUN_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/910b/CANN-custom_ops--linux.aarch64.run"
ARG A2_CUSTOM_OPS_WHL_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/910b/custom_ops-1.0-cp311-cp311-linux_aarch64.whl"
ARG A2_TRANSFORMER_RUN_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/910b/cann-ops-transformer-custom_linux-aarch64.run"

ARG A3_CUSTOM_OPS_RUN_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/a3/CANN-custom_ops--linux.aarch64.run"
ARG A3_CUSTOM_OPS_WHL_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/a3/custom_ops-1.0-cp311-cp311-linux_aarch64.whl"
ARG A3_TRANSFORMER_RUN_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/newmodel/pkg_20260413/a3/cann-ops-transformer-custom_linux-aarch64.run"


RUN if [ "$DEVICE_TYPE" = "910b" ]; then \
      echo "Using 910b dependencies"; \
      echo "CUSTOM_OPS_WHL_URL=$A2_CUSTOM_OPS_WHL_URL" >> /etc/environment_new; \
    elif [ "$DEVICE_TYPE" = "a3" ]; then \
      echo "Using a3 dependencies"; \
      echo "CUSTOM_OPS_WHL_URL=$A3_CUSTOM_OPS_WHL_URL" >> /etc/environment_new; \
    else \
      echo "Unsupported DEVICE_TYPE: $DEVICE_TYPE"; exit 1; \
    fi
# 安装脚本（合并为一个 RUN 层，便于清理）
RUN . /etc/environment_new \
    && mkdir -p /tmp/ascend_ops && cd /tmp/ascend_ops \
    && if [ "$DEVICE_TYPE" = "910b" ]; then \
          echo "Downloading A2 specific packages..." \
          && wget -O CANN-custom_ops.run "$A2_CUSTOM_OPS_RUN_URL" \
          && wget ${CUSTOM_OPS_WHL_URL} \
          && wget -O cann-ops-transformer.run "$A2_TRANSFORMER_RUN_URL"; \
       elif [ "$DEVICE_TYPE" = "a3" ]; then \
          echo "Downloading A3 specific packages..." \
          && wget -O CANN-custom_ops.run "$A3_CUSTOM_OPS_RUN_URL" \
          && wget ${CUSTOM_OPS_WHL_URL} \
          && wget -O cann-ops-transformer.run "$A3_TRANSFORMER_RUN_URL"; \
       else \
          echo "Unsupported DEVICE_TYPE: $DEVICE_TYPE (must be a2 or a3)" && exit 1; \
       fi \
    && chmod +x CANN-custom_ops.run \
    && ./CANN-custom_ops.run --quiet --install-path=${ASCEND_CANN_PATH}/latest/opp \
    && chmod +x cann-ops-transformer.run \
    && ./cann-ops-transformer.run --quiet --install-path=${ASCEND_CANN_PATH}/latest/opp \
    && ${PIP_INSTALL} ${CUSTOM_OPS_WHL_URL} \
    && cd / && rm -rf /tmp/ascend_ops

# 将环境变量 source 命令持久化到 profile.d，确保后续会话可用
RUN echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/customize/bin/set_env.bash" >> /etc/profile\
    && echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/custom_transformer/bin/set_env.bash" >> /etc/profile\
    && echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/customize/bin/set_env.bash" >>  ~/.bashrc \
    && echo "source ${ASCEND_CANN_PATH}/latest/opp/vendors/custom_transformer/bin/set_env.bash" >>  ~/.bashrc

CMD ["/bin/bash"]