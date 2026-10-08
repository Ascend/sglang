ARG CANN_VERSION=9.1.0
ARG DEVICE_TYPE=950
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12
ARG arch

FROM quay.io/ascend/cann:$CANN_VERSION-$DEVICE_TYPE-$OS-$PYTHON_VERSION

# Update pip & apt sources
ARG TARGETARCH
ARG CANN_VERSION
ARG DEVICE_TYPE
ARG arch
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG APTMIRROR=""
ARG PYTORCH_VERSION="2.10.0"
ARG TORCHVISION_VERSION="0.25.0"

ARG TORCHAUDIO_VERSION="2.10.0"
ARG PTA_URL_ARM64="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl"
ARG PTA_URL_AMD64="https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-pytorch2.10.0/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_x86_64.whl"
ARG SGLANG_TAG=ifmn/npu/glm-5-optim_0824
ARG ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit
ARG SGLANG_KERNEL_NPU_TAG=2026.9.14
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG DEVICE_TYPE

# ---------------------------------------------------------------------------
# MemFabric: build from source branch instead of installing released PyPI wheels
# ---------------------------------------------------------------------------
ARG MEMFABRIC_REPO="https://gitcode.com/Ascend/memfabric_hybrid.git"
ARG MEMFABRIC_REF="br_feature_a5_offload"
ARG MEMFABRIC_SRC_DIR="/opt/memfabric_hybrid"
ARG MF_BUILD_JOBS="32"
# OFF keeps the default of script/build_and_pack_run.sh. Set ON (with MF_BUILD_HCOM_UB=ON)
# if the workload uses HOST_RDMA / HOST_URMA data paths (needs libibverbs-dev).
ARG MF_BUILD_HCOM="OFF"
ARG MF_BUILD_HCOM_UB="OFF"
# ON builds and installs the HYBM AICPU kernel run package into the CANN OPP tree,
# so memfabric_hybrid does not have to rebuild the AICPU kernel on first import.
ARG MF_PREINSTALL_AICPU_KERNEL="OFF"
# py: keep memfabric-zbal from PyPI; src: build it from the same branch (see notes)
ARG MEMFABRIC_ZBAL_SOURCE="py"
ARG MEMFABRIC_ZBAL_VERSION="1.2.21004.post1"


RUN if [ "$TARGETARCH" = "amd64" ]; then \
      echo "Using x86_64 dependencies"; \
      echo "export PTA_URL=$PTA_URL_AMD64" >> /etc/environment_new; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      echo "Using aarch64 dependencies"; \
      echo "export PTA_URL=$PTA_URL_ARM64" >> /etc/environment_new; \
    else \
      echo "Unsupported TARGETARCH: $TARGETARCH"; exit 1; \
    fi

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url $PIP_INDEX_URL
RUN if [ -n "$APTMIRROR" ];then sed -i "s|.*.ubuntu.com|$APTMIRROR|g" /etc/apt/sources.list ;fi

# Install development tools and utilities
RUN apt-get update -y && apt upgrade -y && apt-get install -y \
    git \
    unzip \
    build-essential \
    cmake \
    ninja-build \
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


### Build & install MemFabric Hybrid from ${MEMFABRIC_REF}
RUN set -ex; \
    python3 -c 'import os, sysconfig; h = os.path.join(sysconfig.get_path("include"), "Python.h"); assert os.path.exists(h), "python headers not found: " + h'; \
    ${PIP_INSTALL} "setuptools>=68.0" pybind11 "wheel==0.45.1"; \
    python3 -m pip uninstall -y memfabric-hybrid memfabric-zbal || true; \
    git clone --depth 1 --branch "${MEMFABRIC_REF}" "${MEMFABRIC_REPO}" "${MEMFABRIC_SRC_DIR}"; \
    cd "${MEMFABRIC_SRC_DIR}"; \
    . /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh; \
    export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PATH}/latest}"; \
    export PYTHON_HOME="$(python3 -c 'import sys; print(sys.prefix)')"; \
    export MF_BUILD_JOBS="${MF_BUILD_JOBS}"; \
    bash script/build.sh RELEASE OFF OFF ON ON NPU OFF "${MF_BUILD_HCOM}" ON "${MF_BUILD_HCOM_UB}" OFF cmake; \
    ${PIP_INSTALL} "${MEMFABRIC_SRC_DIR}"/output/memfabric_hybrid/wheel/*.whl

### Optional: build & install the HYBM AICPU kernel run package (avoids first-import build)
RUN if [ "${MF_PREINSTALL_AICPU_KERNEL}" = "ON" ]; then \
      set -ex; \
      cd "${MEMFABRIC_SRC_DIR}"; \
      . /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh; \
      export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PATH}/latest}"; \
      bash script/kernel/build_ops_run.sh; \
      chmod +x output/memfabric_hybrid_aicpu_kernel.run; \
      ./output/memfabric_hybrid_aicpu_kernel.run --install --install-for-all --force; \
    else \
      echo "[memfabric] skip AICPU kernel pre-install (MF_PREINSTALL_AICPU_KERNEL=OFF)"; \
    fi

### Install SGLang Model Gateway
RUN ${PIP_INSTALL} sglang-router


### Install PyTorch and PTA
RUN . /etc/environment_new && \
    (${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cpu) \
    && (${PIP_INSTALL} ${PTA_URL})


### Install memfabric-zbal (PyPI wheel by default; MEMFABRIC_ZBAL_SOURCE=src builds it from the branch)
RUN set -ex; \
    if [ "${MEMFABRIC_ZBAL_SOURCE}" = "src" ]; then \
      cd "${MEMFABRIC_SRC_DIR}"; \
      . /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh; \
      export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PATH}/latest}"; \
      export PYTHON_HOME="$(python3 -c 'import sys; print(sys.prefix)')"; \
      bash app/zbal/script/build.sh; \
      ${PIP_INSTALL} "${MEMFABRIC_SRC_DIR}"/app/zbal/output/memfabric_zbal*.whl; \
    else \
      ${PIP_INSTALL} memfabric-zbal==${MEMFABRIC_ZBAL_VERSION} -i https://pypi.org/simple/; \
    fi; \
    rm -rf "${MEMFABRIC_SRC_DIR}"


## Install triton-ascend
RUN . /etc/environment_new && \
    ${PIP_INSTALL} pybind11 && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        ${PIP_INSTALL} https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ta/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        ${PIP_INSTALL} https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl; \
    else \
        echo "Unsupported architecture: $TARGETARCH"; \
        exit 1; \
    fi

# Install SGLang
RUN git clone https://github.com/shengzhaotian/sglang --branch ${SGLANG_TAG} /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && \
    sed -i '/"memfabric-hybrid==1.1.4"/d; /"memfabric-zbal==1.1.2"/d' pyproject.toml && \
    ${PIP_INSTALL} -v -e .[all_npu]

RUN mkdir cann-custom-ops && \
    cd cann-custom-ops && \
    source /usr/local/Ascend/cann-${CANN_VERSION}/set_env.sh && \
    wget https://github.com/Liwansi/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    wget https://github.com/Liwansi/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
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
    && wget https://github.com/Liwansi/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.10.0-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

# Verify the MemFabric that ended up in the image (git commit proves the branch build)
RUN set -ex; \
    python3 -m pip show memfabric-hybrid memfabric-zbal | grep -E "^(Name|Version):"; \
    location="$(python3 -m pip show memfabric-hybrid | awk -F': ' '/^Location:/{print $2}')"; \
    cat "${location}/memfabric_hybrid/VERSION"

CMD ["/bin/bash"]