# 环境变量说明

MindIE LLM安装完成后，提供进程级环境变量设置脚本“set\_env.sh“，以自动完成环境变量设置。

## “set\_env.sh“脚本环境变量

**表 1** “set\_env.sh“脚本环境变量说明


|环境变量名|功能描述|取值范围|默认值|
|--|--|--|--|
|**MindIE_LLM相关环境变量**|
|MINDIE_LLM_HOME_PATH|MindIE LLM主目录所在路径。|N/A|N/A|
|MINDIE_LLM_RECOMPUTE_THRESHOLD|MindIE LLM中重计算阈值。|[0,1]|0.5|
|PYTORCH_INSTALL_PATH|torch三方件的安装路径，使用以下方式获取python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))'。|N/A|N/A|
|PYTORCH_NPU_INSTALL_PATH|torch_npu三方件的安装路径，使用以下方式获取python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))'。|N/A|N/A|
|**ATB_Models相关环境变量**|
|ATB_OPERATION_EXECUTE_ASYNC|控制ATB graph的异步调度，默认使用二级流水。当CPU数量不受限时，可尝试开启三级流水进行性能调优。|0：不开启<br>1：开启二级流水<br>2：开启三级流水|1|
|ATB_SPEED_HOME_PATH|ATB模型lib路径的环境变量，必须配置。|必须是ATB模型lib路径|None|
|HCCL_INTRA_PCIE_ENABLE|控制是否开启All2All分层通信及INT8通信特性。“HCCL_INTRA_PCIE_ENABLE”和“HCCL_INTRA_ROCE_ENABLE”必须同时设置为开启，才能开启该功能。这两个环境变量的更多描述请参见《CANN 环境变量参考》中的“集合通信”章节。建议在Atlas 800I A2 推理服务器和Atlas 800I A3 超节点服务器，MoE模型的Combine INT8算子场景下打开 ，提升性能优化。|0：关闭<br>1：开启|N/A|
|HCCL_INTRA_ROCE_ENABLE|0：开启1：关闭|N/A|
|**日志相关环境变量**|
|MINDIE_LOG_TO_FILE|MindIE日志写入文件开关。|0：不开启<br>1：开启|1|
|MINDIE_LOG_LEVEL|MindIE日志级别。|DEBUG<br>INFO<br>WARN<br>ERROR<br>CRITICAL|INFO|
|MINDIE_LOG_TO_STDOUT|MindIE日志写入标准输出流开关。|0：不开启<br>1：开启|0|
|**Ascend Extension for PyTorch相关环境变量**|
|MASTER_IP|多机服务化设置的主机ip|若值非空，则IP应该是合法ip|None|
|MASTER_PORT|多机服务化设置的主机接口|若值非空，则端口号[0,65535]|None|


## 其他可选环境变量

Server相关环境变量请参考**表2**。

**表 2**  Server相关环境变量说明

|参数名称|参数说明|取值范围|缺省值|
|--|--|--|--|
|MINDIE_LLM_HOME_PATH|Server的安装路径。|路径参数。|/usr/local/Ascend/mindie/latest/mindie-service|
|MIES_CONFIG_JSON_PATH|config.json文件的路径。如果该环境变量存在，则读取该值；如果不存在，则读取*${MINDIE_LLM_HOME_PATH}*/conf/config.json文件。|路径参数。|NA|
|MIES_CONTAINER_IP|容器IP地址，容器部署时配置。EndPoint提供的业务面RESTful接口绑定的IP地址和多机推理场景gRPC通信采用的IP地址。多机推理时需要设置该环境变量。|IPv4地址。|NA|
|MIES_CONTAINER_MANAGEMENT_IP|EndPoint提供的内部RESTful接口绑定的IP地址。|IPv4地址。|NA|
|MIES_MEMORY_DETECTOR_MODE|内存状态打点使能开关。|0：关闭<br>1：开启|0|
|MIES_PROFILER_MODE|性能状态打点使能开关。|0：关闭<br>1：开启|0|
|LD_LIBRARY_PATH|lib所在的路径。|路径参数。|${MINDIE_LLM_HOME_PATH}/lib:${LD_LIBRARY_PATH}|
|ASCEND_SLOG_PRINT_TO_STDOUT|CANNDEV日志打印控制开关。|1：打印。<br>0：写入到~/ascend目录。|0|
|ASCEND_GLOBAL_LOG_LEVEL|CANNDEV日志级别。|0：debug<br>1：info<br>2：warn<br>3：error|3|
|ASCEND_GLOBAL_EVENT_ENABLE|设置应用类日志是否开启Event日志。|0：关闭Event<br>日志。1：开启Event日志。|0|
|HCCL_BUFFSIZE|控制两个NPU之间共享数据的缓存区大小。|大于或等于1，单位：MB。|120|
|EP_OPENSSL_PATH|EndPoint开启HTTPS认证后，通过该环境变量来指定openssl加载运行时so文件。该环境变量在EndPoint模块启动时自动设置，不需要用户手动设置。|路径参数。|${MINDIE_LLM_HOME_PATH}/lib|
|HSECEASY_PATH|EndPoint开启HTTPS认证后，使用HSECEASY工具对密钥口令进行加密。该环境变量指定HSECEASY加载运行时so文件路径。|路径参数。|${MINDIE_LLM_HOME_PATH}/lib|
|MIES_CERTS_LOG_TO_FILE|证书管理工具环境变量，日志是否输出到文件。|0：输出到文件。<br>1：不输出。|0|
|MIES_CERTS_LOG_TO_STDOUT|证书管理工具环境变量，日志打印控制开关。|0：不打印日志。<br>1：打印日志。|1|
|MIES_CERTS_LOG_LEVEL|证书管理工具环境变量，日志级别。|DEBUG<br>INFO<br>WARNING<br>ERROR<br>FATAL|INFO|
|MIES_CERTS_LOG_PATH|证书管理工具环境变量，日志路径。|路径参数。|/workspace/log/certs.log|
|DYNAMIC_AVERAGE_WINDOW_SIZE|/metrics-json接口中，动态统计指标平均值的动态窗口大小。|正数|1000|
|MIES_SERVICE_MONITOR_MODE|是否开启推理服务化的在线管控指标，开启时才可以正常请求/metrics接口。|0：关闭<br>1：开启|0|
|LOCAL_CACHE_DIR|收到多模态请求后，通过该环境变量来指定图片的暂存路径。|路径参数。|~/mindie/cache|
|TOKENIZER_ENCODE_TIMEOUT|TOKENIZER Encode截断的超时时间，单位为秒。|[5, 300]|60|
|MINDIE_ASYNC_SCHEDULING_ENABLE|是否开启异步调度。|1：开启其他值：关闭|NA|


MindIE\_LLM相关环境变量请参考**表3**。

**表 3**  MindIE\_LLM相关环境变量说明

|环境变量名|功能描述|取值范围|默认值|
|--|--|--|--|
|HOST_IP|宿主机IP地址。|N/A|N/A|
|LOCAL_RANK|指示Device的本地ID。|[0, ${WORLD_SIZE} - 1]|0|
|MIES_USE_MB_SWAPPER|高性能Swap开关。|0：不开启<br>1：开启|0|
|MINDIE_CHECK_INPUTFILES_PERMISSION|是否需要检验外部文件的权限信息，包括文件所有者和其他人对文件的写权限。|0：不需要检验外部文件的权限信息<br>其他值或None：需要检验外部文件的权限信息。|None|
<<<<<<< HEAD
|MINDIE_LLM_BENCHMARK_ENABLE|是否开启MindIE LLM模块的Benchmark功能，开启后将会输出性能数据到指定文件路径。|0：不开启<br>1：开启|0|
=======
|MINDIE_LLM_BENCHMARK_ENABLE|是否开启MindIE LLM模块的Benchmark功能，开启后将会输出性能数据到请求响应体(指定路径时，还输出到指定文件路径)。|1：开启同步模式<br>2：开启异步模式<br>其他值或None：不开启|None|
>>>>>>> upstream/dev
|MINDIE_LLM_BENCHMARK_FILEPATH|指定MindIE LLM模块的Benchmark功能输出的性能数据文件路径。|N/A|"{MINDIE_LLM_HOME_PATH}/logs/benchmark.jsonl"|
|MINDIE_LLM_BENCHMARK_RESERVING_RATIO|当性能数据文件超过最大文件大小限制时，旧数据会被新数据覆盖。此环境变量指定保留旧数据的比例，默认为0.1。|[0.0, 1.0]|0.1|
|MINDIE_LLM_FRAMEWORK_BACKEND|MindIE LLM框架后端类型，当前可选值为"atb"（ATB，默认值）和"ms"（MindSpore）。|ATB<br>MS（不区分大小写）|ATB|
|NPU_DEVICE_IDS|使用的NPU卡号。|[0,卡号]<br>例：[0, 1, 2,...]|N/A|
|NPU_MEMORY_FRACTION|NPU显存利用率，代表总显存分配给模型权重、kvcache和work space的比例。不包含HCCL和PTA申请的空间。建议将该值设置为可拉起服务的最小值。具体方法是：按照默认配置启动服务，若无法拉起服务，则上调参数至可拉起为止；若拉起服务成功，则下调该参数至刚好拉起服务为止。总之，在服务能正常拉起的前提下，更低的值可以保障更高的服务系统稳定性。|(0.0, 1.0]Kimi K2模型，推荐设置为0.9。|在ATB Models中默认值为1.0在MindIE LLM中默认值为0.8|
|PERFORMANCE_PREFIX_TREE_ENABLE|memory_decoding并行解码高性能前缀树实现开关。|0：不开启<br>1：开启|0|
|POST_PROCESSING_SPEED_MODE_TYPE|指定后处理加速模式|0：不开启加速<br>1：开启top_p近似计算<br>2：开启索引加速<br>3：同时开启上top_p近似计算和索引加速|0|
|RANK|指示device的全局ID。|[0, ${WORLD_SIZE})|0|
|SOURCE_DATE_EPOCH|消除whl包的bep差异。|N/A|N/A|
|WORLD_SIZE|启用几张卡进行推理。|[1,1048576]|N/A|


ATB\_Models相关环境变量请参考**表4**。

**表 4**  ATB\_Models相关环境变量说明

|环境变量名|功能描述|取值范围|默认值|
|--|--|--|--|
|ATB_LLM_BENCHMARK_ENABLE|性能数据获取是否打开。|0：不开启<br>其它值：开启|0|
|ATB_LLM_BENCHMARK_FILEPATH|性能数据保存路径。|所有值|None|
|ATB_LLM_ENABLE_AUTO_TRANSPOSE|是否开启权重右矩阵自动转置寻优。|None或1：开启<br>其他值：不开启|None|
|ATB_LLM_HCCL_ENABLE|华为集合通信计算库选择。|N/A|N/A|
|ATB_LLM_LCOC_ENABLE|通信计算掩盖功能开关。|None或1：开启<br>其他值：不开启|None|
|ATB_LLM_LOGITS_SAVE_ENABLE|是否保存logits信息。|0：否<br>其它值：是|0|
|ATB_LLM_LOGITS_SAVE_FOLDER|保存logits信息的文件夹。|所有值|None|
|ATB_LLM_RAZOR_ATTENTION_ENABLE|开启ra压缩需要。|0：不开启<br>1：开启|0|
|ATB_LLM_RAZOR_ATTENTION_ROPE|在rope旋转编码方式下的Razor attention压缩算法使能开关。|0：不开启<br>1：开启|0|
|ATB_LLM_TOKEN_IDS_SAVE_ENABLE|是否保存token信息。|0：否<br>其它值：是|0|
|ATB_LLM_TOKEN_IDS_SAVE_FOLDER|保存token信息的文件夹。|所有值|None|
|ATB_PROFILING_ENABLE|是否采集性能profiling数据。|1：是<br>其他值或None：否|None|
|ATB_USE_TILING_COPY_STREAM|是否开启双stream功能。|1：开启<br>其他值或None：不开启|None|
|BIND_CPU|是否将NPU上运行的进程基于CPU亲和度绑核。|None或1：开启<br>其他值：不开启|None|
|CPU_BINDING_NUM|每个device上绑定的核数。|[0, cpu核数除以numa上的device个数]|None|
|HCCL_DETERMINISTIC|HCCL通信的确定性计算。多机推理场景下建议开启。|false：关闭<br>true：开启|一般为true，与模型相关。|
|IS_ALIBI_MASK_FREE|是否支持Speculate。|1：开启<br>其他值或None：不开启|None|
|LCCL_DETERMINISTIC|LCCL通信的确定性计算。|0：关闭<br>1：开启|一般为1，与模型相关。|
|LONG_SEQ_ENABLE|判断是否使能长序列特性。|1：是<br>其他值或None：否|None|
|MINDIE_ACLNN_CACHE_GLOBAL_COUNT|Plugin Op中aclExecutor及对应aclTensor的全局Cache个数。|[0, 100)|16|
|PROFILING_FILEPATH|设置profiling文件路径，默认保存在当前路径下profiling文件夹中。|N/A|N/A|
|PROFILING_LEVEL|设置ProfilerLevel。|Level0<br>Level1<br>Level2<br>Level_none|Level0|
|RESERVED_MEMORY_GB|模型运行时动态申请显存池的大小。|[0, 64)|3|
|MINDIE_ENABLE_EXPERT_HOTPOT_GATHER|负载均衡专家热点信息采集开关。|1：开启<br>其他值或None：不开启|None|
|MINDIE_EXPERT_HOTPOT_DUMP_PATH|负载均衡专家热点信息保存路径。|所有值|None|
|REMOVE_GENERATION_CONFIG_DICT|开启后，设置模型后处理参数为默认值（仅LLM类模型生效）|1：开启其他值或None：不开启|None|


日志相关环境变量请参考**表5**。

**表 5**  日志相关环境变量说明

|环境变量名|功能描述|取值范围|默认值|
|--|--|--|--|
|MINDIE_LOG_LEVEL|控制日志级别。|DEBUG<br>INFO<br>WARN<br>ERROR<br>CRITICAL|INFO|
|MINDIE_LOG_PATH|控制日志写入路径。|N/A|"mindie/log/debug"|
|MINDIE_LOG_ROTATE|控制日志轮转。|-fs：每个日志文件的大小，单位MB，取值范围[1, 500]<br>-r：每个进程可写日志文件个数，取值范围[1, 64]<br>例如：export MINDIE_LOG_ROTATE="-fs 40 -r 2"|-fs：20-r：10|
|MINDIE_LOG_TO_FILE|控制日志是否保存到文件，设置为1则开启。|{0, 1, true, false}|true|
|MINDIE_LOG_TO_STDOUT|控制日志是否打印，设置为1则开启。|{0, 1, true, false}|false|
|MINDIE_LOG_VERBOSE|控制日志中是否加入可选日志内容。|{0, 1, true, false}|true|
|PYTHON_LOG_MAXSIZE|ATB Python日志单个文件的最大容量（单位：字节）。|[0, 524288000]|None|


加速库相关环境变量请参考**表6**。

**表 6**  加速库相关环境变量说明

|环境变量名|功能描述|取值范围|默认值|
|--|--|--|--|
|ASCEND_LAUNCH_BLOCKING|算子同步下发功能开关，用于debug场景。|0：不开启<br>1：开启|0|
|ASCEND_RT_VISIBLE_DEVICES|设置卡号。|[0, 卡号]<br>例：[0, 1, 2,...]|N/A|
|ATB_HOME_PATH|ATB加速库路径的环境变量，无默认值，必须配置。|N/A|N/A|
|ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT|全局kernelCache的槽位数。增加槽位数时：增加cache命中率，但降低检索效率。减少槽位数时：提高检索效率，但降低cache命中率。|[1, 1024]|16|
|ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT|本地kernelCache的槽位数。增加槽位数时：增加cache命中率，但降低检索效率。减少槽位数时：提高检索效率，但降低cache命中率。|[1, 1024]|1|
|ATB_WORKSPACE_MEM_ALLOC_GLOBAL|是否使用全局中间tensor内存分配算法。开启后会对中间tensor内存进行大小计算与分配。|0：不开启<br>1：开启|1|


更多加速库相关的环境变量可参考《CANN ATB加速库开发指南》的“环境变量参考”章节。

> [!NOTE]说明
>-  “INF\_NAN\_MODE\_ENABLE”，“TASK\_QUEUE\_ENABLE”和“RANK\_TABLE\_FILE”等更多PyTorch环境变量，请参见《环境变量参考》中的“INF\_NAN\_MODE\_ENABLE”章节。
>-  当BIND\_CPU环境变量开启时，会调用execute\_command方法执行以下命令：
>   **execute\_command\(\["npu-smi", "info", "-i", f"\{npu\_id\}", "-t", "memory"\]\).split\("\\n"\)\[1:\]execute\_command\(\["npu-smi", "info", "-i", f"\{npu\_id\}", "-t", "usages"\]\).split\("\\n"\)\[1:\]execute\_command\(\["npu-smi", "info", "-m"\]\).strip\(\).split\("\\n"\)\[1:\]execute\_command\(\["npu-smi", "info", "-t", "board", "-i", f"\{device\_info.npu\_id\}", -c", f"\{device\_info.chip\_id\}"\]\).strip\(\).split\("\\n"\)execute\_command\(\["lspci", "-s", f"\{pcie\_no\}", "-vvv"\]\).split\("\\n"\)execute\_command\(\["lscpu"\]\).split\("\\n"\)**
