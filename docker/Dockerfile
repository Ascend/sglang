FROM quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12

RUN python3 -m pip install --no-cache-dir cannbot-dsl==0.8.0

RUN python3 -m pip install --no-cache-dir memfabric-hybrid==1.5.0.dev20261123 --trusted-host ascend.devcloud.huaweicloud.com -i https://ascend.devcloud.huaweicloud.com/artifactory/ascend-pypi/nightly/memfabric-hybrid; \
    mfcli kernel install --soc-version A5

CMD ["/bin/bash"]
