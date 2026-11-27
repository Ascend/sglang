ARG CANN_VERSION=9.2.0
ARG DEVICE_TYPE=a3
ARG OS=ubuntu22.04
ARG PYTHON_VERSION=py3.12

FROM quay.io/ascend/cann:${CANN_VERSION}-${DEVICE_TYPE}-${OS}-${PYTHON_VERSION}

ARG DEVICE_TYPE

RUN python3 -m pip install --no-cache-dir cannbot-dsl==0.8.0

RUN python3 -m pip install --no-cache-dir \
        memfabric-hybrid==1.5.0.dev20261124 \
        --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/ \
    && command -v mfcli \
    && case "${DEVICE_TYPE}" in \
         a3)  SOC_VERSION="A3" ;; \
         950) SOC_VERSION="A5" ;; \
         *)   echo "Unsupported DEVICE_TYPE: ${DEVICE_TYPE}" >&2; exit 1 ;; \
       esac \
    && mfcli kernel install --soc-version "${SOC_VERSION}"
