ARG CANN_VERSION=9.3.0
ARG DEVICE_TYPE=950
ARG arch

ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/sglang/cann9.3.0-openeuler24.03:20261009-py312
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG TARGETARCH
ARG CANN_VERSION
ARG DEVICE_TYPE
ARG arch
ARG BASE_IMAGE
ARG PIP_INDEX_URL="https://pypi.org/simple/"
ARG PYTORCH_VERSION="2.12.0"
ARG TORCHVISION_VERSION="0.27.0"
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
ARG TORCH_NPU_VERSION="2.12.0.post2"
ARG TORCH_NPU_INDEX_URL="https://ascend.devcloud.huaweicloud.com/pypi/simple/"
ARG SGLANG_TAG=main
ARG SGLANG_KERNEL_NPU_REPO=huangxiaojun15/sgl-kernel-npu
ARG SGLANG_KERNEL_NPU_TAG=2026.10.7
ARG PIP_INSTALL="python3 -m pip install --no-cache-dir"
ARG MODELSCOPE_VERSION=""
ARG EVALSCOPE_VERSION=""

ARG MF_VERSION="1.2.1"

ARG ZBAL_VERSION_950="1.2.21004.post1"

WORKDIR /workspace

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip config set global.index-url $PIP_INDEX_URL

RUN dnf makecache && dnf install -y \
        unzip \
        zip \
        which \
        findutils \
        tar \
        xz \
        net-tools \
        zlib-devel \
        pkgconf \
        openssl \
        openssl-devel \
        glibc-all-langpacks \
    && dnf clean all \
    && rm -rf /var/cache/yum

RUN dnf install -y ccache lld clang mesa-libGL \
    || echo "[warn] 部分可选包（ccache/lld/clang/mesa-libGL）没装上，继续构建"

RUN localedef -i en_US -f UTF-8 en_US.UTF-8 || true
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN printf 'source /usr/local/Ascend/cann/set_env.sh\n' > /etc/profile.d/ascend.sh
ENV ASCEND_HOME_PATH=/usr/local/Ascend/cann
ENV ASCEND_OPP_PATH=/usr/local/Ascend/cann/opp
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/lib:${ASCEND_HOME_PATH}/aarch64-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib
ENV PYTHONPATH=${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe

RUN set -eux; \
    if [ "$DEVICE_TYPE" != "950" ]; then \
        echo "Unsupported DEVICE_TYPE (only 950 is built): $DEVICE_TYPE" >&2; \
        exit 1; \
    fi; \
    MF_SOC_VERSION="A5"; \
    ${PIP_INSTALL} memfabric-hybrid==${MF_VERSION}; \
    mfcli kernel install --soc-version "$MF_SOC_VERSION"; \
    ${PIP_INSTALL} memcache-hybrid==${MF_VERSION}

RUN PIP_INDEX="${PIP_INDEX_URL:+--index-url ${PIP_INDEX_URL}}"; \
    ${PIP_INSTALL} "memfabric-zbal==${ZBAL_VERSION_950}" ${PIP_INDEX}

RUN ${PIP_INSTALL} sglang-router

RUN . /usr/local/Ascend/cann/set_env.sh && \
    (${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}) \
    && (${PIP_INSTALL} "torch-npu==${TORCH_NPU_VERSION}" --extra-index-url ${TORCH_NPU_INDEX_URL})

RUN . /usr/local/Ascend/cann/set_env.sh && \
    python3 -c "import torch, torch_npu; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__)"

RUN . /usr/local/Ascend/cann/set_env.sh && \
    MS_PKG="modelscope" && \
    ES_PKG="evalscope" && \
    if [ -n "${MODELSCOPE_VERSION}" ]; then MS_PKG="modelscope==${MODELSCOPE_VERSION}"; fi && \
    if [ -n "${EVALSCOPE_VERSION}" ]; then ES_PKG="evalscope==${EVALSCOPE_VERSION}"; fi && \
    ${PIP_INSTALL} "${MS_PKG}" "${ES_PKG}"

RUN . /usr/local/Ascend/cann/set_env.sh && \
    ${PIP_INSTALL} pybind11 && \
    if [ "$TARGETARCH" != "arm64" ]; then \
        echo "Unsupported architecture (only aarch64 is built): $TARGETARCH" >&2; \
        exit 1; \
    fi && \
    ${PIP_INSTALL} ""https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com:443/Triton-ascend/3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl?AccessKeyId=HPUAAPJN7IAXFCS2GDSQ&Expires=1806290330&Signature=eRq3VKjwgP/tTkObtsho%2BzIsmJM%3D""

RUN git clone https://github.com/sgl-project/sglang --branch ${SGLANG_TAG} /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang/python && rm -rf pyproject.toml && mv pyproject_npu.toml pyproject.toml && \
    ${PIP_INSTALL} -v -e .[all_npu]

RUN mkdir cann-custom-ops && \
    cd cann-custom-ops && \
    . /usr/local/Ascend/cann/set_env.sh && \
    wget https://github.com/${SGLANG_KERNEL_NPU_REPO}/releases/download/${SGLANG_KERNEL_NPU_TAG}/custom-ops-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    wget https://github.com/${SGLANG_KERNEL_NPU_REPO}/releases/download/${SGLANG_KERNEL_NPU_TAG}/ops-transformer-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip && \
    unzip custom-ops-*.zip && \
    unzip ops-transformer-*.zip && \
    chmod +x *.run && \
    ./CANN-custom_ops-none-linux.$(arch).run --install-path=${ASCEND_HOME_PATH}/opp && \
    ./cann-ops-transformer-custom_linux-$(arch).run --install-path=${ASCEND_HOME_PATH}/opp && \
    . ${ASCEND_HOME_PATH}/opp/vendors/customize/bin/set_env.bash && \
    . ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/bin/set_env.bash && \
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then . /usr/local/Ascend/nnal/atb/set_env.sh; fi && \
    ${PIP_INSTALL} custom_ops-*.whl && \
    cd .. && rm -rf cann-custom-ops

RUN ${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil \
    && mkdir sgl-kernel-npu \
    && cd sgl-kernel-npu \
    && wget https://github.com/${SGLANG_KERNEL_NPU_REPO}/releases/download/2026.10.4/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip \
    && unzip sgl-kernel-npu-*.zip \
    && ${PIP_INSTALL} deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl \
    && cd .. && rm -rf sgl-kernel-npu \
    && cd "$(python3 -m pip show deep-ep | awk '/^Location:/ {print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so

CMD ["/bin/bash"]
