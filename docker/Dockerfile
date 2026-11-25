FROM quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12

RUN python3 -m pip install --no-cache-dir cannbot-dsl==0.8.0

RUN python3 -m pip install --no-cache-dir \
        memfabric-hybrid==1.5.0.dev20261124 \
        --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/ \
    && command -v mfcli \
    && mfcli kernel install --soc-version A5

CMD ["/bin/bash"]
