ARG CANN_VERSION=9.1.0
ARG DEVICE_TYPE=950
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12
ARG arch

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# 原始 Dockerfile 里用了 `source`, 必须保证 RUN 走 bash, 否则是 dash
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------- 基础参数 ----
ARG TARGETARCH
ARG arch
ARG CANN_VERSION
ARG DEVICE_TYPE

ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit

# ------------------------------------------------------------ torch / NPU ----
ARG PYTORCH_VERSION="2.10.0"
ARG TORCHVISION_VERSION="0.25.0"
ARG TORCHAUDIO_VERSION="2.10.0"
ARG PTA_URL_ARM64="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl"
ARG PTA_URL_AMD64="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_x86_64.whl"
# triton-ascend: 软文里 "Triton" 那条线
ARG TRITON_ASCEND_ARM64="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ta/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
ARG TRITON_ASCEND_AMD64="https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

# --------------------------------------------------------------- tilelang ----
# 注意: 该 URL 带签名, Expires=1789681639 (≈2026-09-18 04:27 UTC+8), 过期后需重新取
ARG TILELANG_WHEEL_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/dsv41/tilelang-0.1.2%2Bubuntu.22.4.npuir-cp312-cp312-linux_aarch64.whl?AccessKeyId=HPUAXT4YM0U8JNTERLST&Expires=1789681639&Signature=n%2FbGuUSIGPa7OGPpkRS%2B54h3lfA%3D"

# ---------------------------------------------------------------- sglang ------
ARG SGLANG_REPO="https://github.com/sgl-project/sglang.git"
# sgl-project/sglang#38950 的 PR head; 置空则回落到 SGLANG_TAG
ARG SGLANG_PR=38950
ARG SGLANG_TAG="main"
ARG SGLANG_WORKDIR="/sgl-workspace/sglang"

# ------------------------------------------------- memfabric-hybrid (源码) ----
ARG MF_REPO="https://gitcode.com/victor7wang/memfabric_hybrid.git"
ARG MF_BRANCH="br_v4.1_a5"
ARG MF_VERSION="1.2.1"
ARG MF_PREFIX="/usr/local/memfabric_hybrid"
# hybrid 由源码 run 包安装, 默认不再 pip 装; zbal 仍是独立 pip 包
ARG MEMFABRIC_ZBAL_PIP_VERSION="1.2.21004.post1"

# -------------------------------------------------- kernel / custom-ops ------
ARG SGLANG_KERNEL_NPU_TAG="2026.9.0.post1"
# 注意: 该 URL 带签名, Expires=1789680150 (≈2026-09-18 03:52 UTC+8), 过期后需重新取
ARG OPS_TRANSFORMER_RUN_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/dsv41/cann-ops-transformer-custom_linux-aarch64.run?AccessKeyId=HPUAXT4YM0U8JNTERLST&Expires=1789680150&Signature=nIu2UpZryzkP4VVHxl6sWWHiTq8%3D"

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# 按 TARGETARCH 落地各架构 URL, 后续 RUN 统一 `. /etc/environment_new`
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64) printf 'export PTA_URL=%s\nexport TRITON_ASCEND_URL=%s\n' "${PTA_URL_AMD64}" "${TRITON_ASCEND_AMD64}" > /etc/environment_new ;; \
      arm64) printf 'export PTA_URL=%s\nexport TRITON_ASCEND_URL=%s\n' "${PTA_URL_ARM64}" "${TRITON_ASCEND_ARM64}" > /etc/environment_new ;; \
      *) echo "Unsupported TARGETARCH: ${TARGETARCH}"; exit 1 ;; \
    esac

RUN pip config set global.index-url "${PIP_INDEX_URL}"
RUN if [ -n "${APTMIRROR}" ]; then sed -i "s|.*.ubuntu.com|${APTMIRROR}|g" /etc/apt/sources.list; fi

# 开发工具 (memfabric 源码编译也需要 cmake/gcc/git)
RUN apt-get update -y && apt-get install -y \
    unzip \
    build-essential \
    cmake \
    vim \
    wget \
    curl \
    git \
    net-tools \
    zlib1g-dev \
    libnuma-dev \
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

### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router

### Install memfabric-zbal (hybrid 走源码编译, 不在这里装)
RUN ${PIP_INSTALL} memfabric-zbal==${MEMFABRIC_ZBAL_PIP_VERSION} -i ${PIP_INDEX_URL}
# 如果 1.2.1 的 run 包没带 python 包, 再打开下面这行
# RUN ${PIP_INSTALL} memfabric-hybrid==${MF_VERSION}

### Install PyTorch and PTA
RUN . /etc/environment_new \
    && (${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cpu) \
    && (${PIP_INSTALL} ${PTA_URL})

### Install triton-ascend
RUN . /etc/environment_new \
    && ${PIP_INSTALL} pybind11 \
    && ${PIP_INSTALL} ${TRITON_ASCEND_URL}

### Install tilelang (NPUIR)
RUN ${PIP_INSTALL} "${TILELANG_WHEEL_URL}"

# -----------------------------------------------------------------------------
# memfabric-hybrid: 源码编译 br_v4.1_a5 -> run 包 -> 安装
# 需要 build-essential / cmake 等已在上面装好
# -----------------------------------------------------------------------------
RUN set -eux; \
    git clone --branch "${MF_BRANCH}" --depth 1 "${MF_REPO}" /tmp/memfabric_hybrid; \
    cd /tmp/memfabric_hybrid; \
    bash script/build.sh; \
    ls -l ./*.run; \
    PKG=$$(ls -1 memfabric_hybrid*.run | head -n1); \
    echo "install pkg: $$PKG"; \
    "./$$PKG" --install; \
    rm -rf /tmp/memfabric_hybrid

# set_env.sh 固化到 shell 环境 (RUN 里则显式 source)
RUN set -eux; \
    test -f ${MF_PREFIX}/set_env.sh; \
    echo "source ${MF_PREFIX}/set_env.sh" > /etc/profile.d/10_memfabric_hybrid.sh; \
    grep -qxF "source ${MF_PREFIX}/set_env.sh" /root/.bashrc || echo "source ${MF_PREFIX}/set_env.sh" >> /root/.bashrc

# -----------------------------------------------------------------------------
# custom ops (1): sgl-kernel-npu release 里的 CANN-custom_ops
# -----------------------------------------------------------------------------
RUN mkdir cann-custom-ops \
    && cd cann-custom-ops \
    && source /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh \
    && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && chmod +x *.run \
    && ./CANN-custom_ops-none-linux.$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp \
    && source /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/customize/bin/set_env.bash \
    && source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    && source /usr/local/Ascend/nnal/atb/set_env.sh \
    && ${PIP_INSTALL} custom_ops-1.0-cp312-cp312-linux_$(arch).whl \
    && cd .. && rm -rf cann-custom-ops

# -----------------------------------------------------------------------------
# custom ops (2): dsv41 版 ops-transformer run 包 (obs 预签名直链)
# -----------------------------------------------------------------------------
RUN set -eux; \
    wget -O /tmp/cann-ops-transformer-custom_linux-$(arch).run "${OPS_TRANSFORMER_RUN_URL}"; \
    chmod +x /tmp/cann-ops-transformer-custom_linux-$(arch).run; \
    /tmp/cann-ops-transformer-custom_linux-$(arch).run --install-path=/usr/local/Ascend/cann-${CANN_VERSION}/opp; \
    rm -f /tmp/cann-ops-transformer-custom_linux-$(arch).run; \
    source /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh; \
    source /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/custom_transformer/bin/set_env.bash

# -----------------------------------------------------------------------------
# SGLang: sgl-project/sglang#38950 (PR head) + NPU 依赖
# -----------------------------------------------------------------------------
RUN set -eux; \
    mkdir -p /sgl-workspace; \
    git clone "${SGLANG_REPO}" "${SGLANG_WORKDIR}"; \
    cd "${SGLANG_WORKDIR}"; \
    if [ -n "${SGLANG_PR}" ]; then \
        git fetch --depth 1 origin refs/pull/${SGLANG_PR}/head:sglang_pr; \
        git checkout sglang_pr; \
    else \
        git checkout "${SGLANG_TAG}"; \
    fi; \
    git log -1 --oneline; \
    cd python; \
    rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml; \
    sed -i '/memfabric-hybrid==/d; /memfabric-zbal==/d' pyproject.toml; \
    ${PIP_INSTALL} -v -e .[all_npu]

# -----------------------------------------------------------------------------
# Deep-ep / sgl-kernel-npu
# -----------------------------------------------------------------------------
RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

# -----------------------------------------------------------------------------
# 统一运行环境 (容器启动 / 后续 RUN 都能用)
# -----------------------------------------------------------------------------
RUN set -eux; \
    cat > /opt/ascend_env.sh <<EOF
#!/bin/bash
# 由 Dockerfile 生成, CANN_VERSION=${CANN_VERSION}
for f in
  /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh
  /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  /usr/local/Ascend/nnal/atb/set_env.sh
  /usr/local/memfabric_hybrid/set_env.sh
  /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/customize/bin/set_env.bash
  /usr/local/Ascend/cann-${CANN_VERSION}/opp/vendors/custom_transformer/bin/set_env.bash
do
  [ -f "\$f" ] && source "\$f"
done
EOF
    chmod +x /opt/ascend_env.sh; \
    grep -qxF 'source /opt/ascend_env.sh' /root/.bashrc || echo 'source /opt/ascend_env.sh' >> /root/.bashrc; \
    echo 'source /opt/ascend_env.sh' > /etc/profile.d/00_ascend_env.sh

ENV ASCEND_CANN_PATH=${ASCEND_CANN_PATH}
ENV MF_HYBRID_PREFIX=${MF_PREFIX}
ENV SGLANG_HOME=${SGLANG_WORKDIR}

CMD ["/bin/bash"]
