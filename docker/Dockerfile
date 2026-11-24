FROM quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12

RUN python3 -m pip install --no-cache-dir cannbot-dsl==0.8.0

RUN set -eux; \
    case "$DEVICE_TYPE" in \
      950) MF_SOC_VERSION="A5" ;; \
      a3)  MF_SOC_VERSION="A3" ;; \
      *)   echo "Unsupported DEVICE_TYPE for mfcli kernel install: $DEVICE_TYPE" >&2; \
           exit 1 ;; \
    esac; \
    ${PIP_INSTALL} memfabric-hybrid==1.5.0.dev20261119 --trusted-host ascend.devcloud.huaweicloud.com -i https://ascend.devcloud.huaweicloud.com/pypi/simple; \
    mfcli kernel install --soc-version "$MF_SOC_VERSION";

CMD ["/bin/bash"]
