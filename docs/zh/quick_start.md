# 快速入门

## 环境准备

本文档以Atlas 800I A2 推理服务器和Qwen2-7B模型为例，让开发者快速开始使用MindIE进行大模型推理流程。

### 前提条件

物理机部署场景，需要在物理机安装NPU驱动固件以及部署Docker，执行如下步骤判断是否已安装NPU驱动固件和部署Docker。

-   执行以下命令查看NPU驱动固件是否安装。若出现类似如[图1](#figure1)所示，说明已安装。否则请参见[表1](#table1)进行安装。

    ```
    npu-smi info
    ```

    **图 1**  回显信息  <a id="figure1"></a>
    
    ![](./figures/command_output.png "回显信息")

    **表 1** Atlas A2 推理系列产品 <a id="table1"></a>
    
    |产品型号       |参考文档|
    |------------|------------|
    |Atlas 800I A2|《Atlas A2 中心推理和训练硬件 24.1.0 NPU驱动和固件安装指南》中的“物理机安装与卸载”章节|


-   执行以下命令查看Docker是否已安装并启动。Docker的安装可参见[安装Docker](../install/source/docker_installation.md)。

    ```
    docker ps
    ```

    回显以下信息表示Docker已安装并启动。

    ```
    CONTAINER ID        IMAGE        COMMAND         CREATED        STATUS         PORTS           NAMES
    ```

### 获取模型权重

1.  请先下载权重，这里以Qwen2-7B为例，下载链接：[https://huggingface.co/Qwen/Qwen2-7B/tree/main](https://huggingface.co/Qwen/Qwen2-7B/tree/main)，将权重文件上传至服务器任意目录（如/home/weight）。
2.  执行以下命令，修改权重文件权限：

    ```
    chmod -R 755 /home/weight
    ```

### 获取容器镜像

进入[昇腾官方镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)，根据设备型号选择下载对应的MindIE镜像。

该镜像已具备模型运行所需的基础环境，包括：CANN、FrameworkPTAdapter、MindIE与ATB Models，可实现模型快速上手推理。

**表 2**  容器内各组件安装路径

|组件|安装路径|
|--|--|
|CANN|/usr/local/Ascend/cann|
|CANN-NNAL-ATB|/usr/local/Ascend/nnal/atb|
|MindIE|/usr/local/Ascend/mindie|
|ATB Models|/usr/local/Ascend/atb-models|

## 启动容器

1. 下载完成镜像后，执行以下命令启动容器。

    ```
    docker run -it -d --net=host --shm-size=1g \
           --name <container-name> \
           -w /home \
           --device=/dev/davinci0:rwm \
           --device=/dev/davinci1:rwm \
           --device=/dev/davinci2:rwm \
           --device=/dev/davinci3:rwm \
           --device=/dev/davinci_manager:rwm \
           --device=/dev/hisi_hdc:rwm \
           --device=/dev/devmm_svm:rwm \
           -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
           -v /usr/local/dcmi:/usr/local/dcmi:ro \
           -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
           -v /usr/local/sbin/:/usr/local/sbin:ro \
           -v /home/weight:/home/weight:ro \
           mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts bash
    ```

    > [!NOTE]说明 
    > - “mindie:2.2.RC1-800I-A2-py311-openeuler24.03-lts”为镜像名称，可根据实际情况修改。
    > - 对于--device参数，挂载权限设置为rwm，而非权限较小的rw或r，原因如下：
    > - 对于Atlas 800I A2 推理服务器，若设置挂载权限为rw，可以正常进入容器，同时也可以使用npu-smi命令查看npu占用信息，并正常运行MindIE业务；但如果挂载的npu（即对应挂载选项中的davinci_xxx_，如npu0对应davinci0）上有其它任务占用，则使用npu-smi命令会打印报错，且无法运行MindIE任务（此时torch.npu.set\_device\(\)会失败）。
    > - 对于Atlas 800I A3 超节点服务器，若设置挂载权限为rw，进入容器后，使用npu-smi命令会打印报错，且无法运行MindIE任务（此时torch.npu.set\_device\(\)会失败）。

    **表 1**  参数说明

    |参数|参数说明|
    |--|--|
    |--name|表示给容器指定一个名称。<container-name>是容器的标识符，可以自行设置，且在当前系统中具有唯一性。如果不设置，Docker会自动分配一个随机名称。|
    |--device|表示映射的设备，可以挂载一个或者多个设备。需要挂载的设备如下：/dev/davinci*X*：NPU设备，X是ID号，如：davinci0。/dev/davinci_manager：davinci相关的管理设备。/dev/hisi_hdc：hdc相关管理设备。/dev/devmm_svm：内存管理相关设备。可根据以下命令查询device个数及名称方式，根据需要绑定设备，修改上面命令中的"--device=****"。ll /dev/ | grep davinci|
    |-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro|将宿主机目录“/usr/local/Ascend/driver”挂载到容器，请根据驱动所在实际路径修改。|
    |-v /usr/local/sbin:/usr/local/sbin:ro|将宿主机工具“/usr/local/sbin/”以只读模式挂载到容器中，请根据实际情况修改。|
    |-v /home/weight:/home/weight:ro|设定权重挂载的路径，需要根据用户的情况修改。请将权重文件和数据集文件同时放置于该路径下。|


2. 执行以下命令进入容器。

    ```
    docker exec -it <container-name> /bin/bash
    ```

> [!NOTE]说明 
> 更多详细信息，请参考[启动容器](https://gitee.com/ascend/ascend-docker-image/tree/dev/mindie#%E5%90%AF%E5%8A%A8%E5%AE%B9%E5%99%A8)章节。

## 模型推理

1. 执行如下命令，查询安装路径<site-packages>。

    ```bash
    pip show mindie_llm | grep location
    ```

   若python版本是3.11，则查询到的默认安装路径为：`/usr/local/lib/python3.11/site-packages`。

2. 执行如下命令，进入安装路径。
   
   ```
   cd <site-packages>
   ```
 
3. 确认配置文件有640权限。

    ```bash
    chmod 640 <site-packages>/mindie_llm/conf/config.json
    ```

    > [!NOTE]说明
    > 若文件权限不符合要求将会导致服务启动失败。
<!--  chmod 750 mindie-service
    chmod -R 550 mindie-service/bin
    chmod 550 mindie-service/lib
    chmod 440 mindie-service/lib/*
    chmod 550 mindie-service/lib/grpc
    chmod 440 mindie-service/lib/grpc/*
    chmod -R 550 mindie-service/include
    chmod -R 550 mindie-service/scripts
    chmod 750 mindie-service/logs
    chmod 750 mindie-service/conf
    chmod 640 mindie-service/conf/config.json
    chmod 700 mindie-service/security
    chmod -R 700 mindie-service/security/* -->

4. 设置环境变量，开启日志打印。 <a id="step3"></a>

    ```
    export MINDIE_LOG_TO_STDOUT=1
    ```

5. 配置服务化参数。
   
    a. 进入conf目录，打开“config.json“文件。

    ```
    cd mindie_llm/conf
    vim config.json
    ```

    b. 按“i”进入编辑模式，根据实际情况修改“config.json“中的配置参数。（以下已Qwen2-7B为例，需要修改的配置参数已加粗）

    ``` json

    {
        "ServerConfig" : 
            {
            "httpsEnabled" : false
            },
        "BackendConfig" : 
        {
                "npuDeviceIds" : [[0,1,2,3]],
                "ModelDeployConfig" :
            {
                    "ModelConfig" : [
                    {
                        "modelName" : "qwen2-7b",
                        "modelWeightPath" : "/home/weight",  
                        "worldSize" : 4,
                        "trustRemoteCode": false
                    }
                ]
            },
        }
    }
    ```

    如上的参数说明如下，更多“config.json“的参数说明请参考[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)。

    |配置项|取值类型|取值范围|配置说明|
    |--|--|--|--|
    |httpsEnabled|bool|true（开启）false（关闭）|是否开启HTTPS通信安全认证。true：开启HTTPS通信。false：关闭HTTPS通信。如果网络环境不安全，不开启HTTPS通信，即“httpsEnabled”=“false”时，会存在较高的网络安全风险。|
    |npuDeviceIds|std::vector<std::set<size_t>>|根据模型和环境的实际情况来决定。|表示启用哪几张卡。对于每个模型实例分配的npuIds，使用芯片逻辑ID表示。在未配置ASCEND_RT_VISIBLE_DEVICES环境变量时，每张卡对应的逻辑ID可使用"npu-smi info -m"指令进行查询。若配置ASCEND_RT_VISIBLE_DEVICES环境变量时，可见芯片的逻辑ID按照ASCEND_RT_VISIBLE_DEVICES中配置的顺序从0开始计数。例如：ASCEND_RT_VISIBLE_DEVICES=1,2,3,4则以上可见芯片的逻辑ID按顺序依次为0,1,2,3。多机推理场景下该值无效，每个节点上使用的npuDeviceIds根据ranktable计算获得。必填，默认值：[[0,1,2,3]]。|
    |modelName|string|由大写字母、小写字母、数字、中划线、点和下划线组成，且不以中划线、点和下划线作为开头和结尾，字符串长度小于或等于256。|模型名称。必填，默认值："llama_65b"。|
    |modelWeightPath|std::string|文件绝对路径长度的上限与操作系统的设置（Linux为PATH_MAX）有关，最小值为1。|模型权重路径。程序会读取该路径下的config.json中torch_dtype和vocab_size字段的值，需保证路径和相关字段存在。必填，默认值："/data/atb_testdata/weights/llama1-65b-safetensors"。该路径会进行安全校验，需要和执行用户的属组和权限保持一致。|
    |worldSize|uint32_t|根据模型实际情况来决定。每一套模型参数中worldSize必须与使用的NPU数量相等。|启用几张卡推理。必填，默认值：4。|
    |trustRemoteCode|bool|truefalse|是否信任远程代码。false：不信任远程代码。true：信任远程代码。选填，默认值：false。如果设置为true，会存在信任远程代码行为，可能会导致恶意代码注入风险，请自行保障代码注入安全风险。|


    c. 按“Esc”，输入`:wq!`，按“Enter”保存并退出编辑。


6.  启动服务。
   
    a. 执行如下命令，启动服务。

    ```
    mindie_llm_server
    ```
    b. 回显如下则说明启动成功。

    ```
    Daemon start success!
    ```

    
    > [!CAUTION]注意 
    >- 如果安装过老版本的MindIE（默认安装路径为`/usr/local/Ascend/mindie`）,为避免搜索到老版本的库，请执行命令`mv /usr/local/Ascend/mindie /usr/local/Ascend/mindie-bak`，移除老版本安装路径下的文件。
    >- bin目录按照安全要求，目录权限为550，没有写权限，但执行推理过程中，算子会在当前目录生成kernel\_meta文件夹，需要写权限，因此不能直接在bin启动mindieservice\_daemon。
    >- Ascend-cann-toolkit工具会在执行服务启动的目录下生成kernel\_meta\_temp\_xxxx目录，该目录为算子的cce文件保存目录。因此需要在当前用户拥有写权限目录下（例如Ascend-mindie-server\__\{version\}_\_linux-_\{arch\}_目录，或者用户在Ascend-mindie-server\__\{version\}_\_linux-_\{arch\}_目录下自行创建临时目录）启动推理服务。
    >- 如需切换用户，请在切换用户后执行**rm -f /dev/shm/\***命令，删除由之前用户运行创建的共享文件。避免切换用户后，该用户没有之前用户创建的共享文件的读写权限，造成推理失败。
    >- 标准输出流捕获到的文件output.log支持用户自定义文件和路径。

7.  发送请求。

    服务化API接口请参考《MindIE LLM开发指南》中的**RESTFUL API参考**章节。

    用户可使用HTTPS客户端（Linux curl命令，Postman工具等）发送HTTPS请求，此处以Linux curl命令为例进行说明。

    重开一个窗口，使用以下命令发送请求。例如验证服务是否拉起：

    ```
    curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
    "prompt": "My name is Olivier and I ",
    "max_tokens":10
    }' http://127.0.0.1:1025/generate
    ```

    回显如下则表明请求发送成功：

    ```
    {"text":["My name is Olivier and I  25 years old. I am a French student"]}
    ```

## 精度测试


> [!NOTE]说明
>-  精度测试和性能测试前，请先重开一个窗口进入容器，并参见[3](#step3)设置环境变量。
>-  以下精度测试以AISBench工具为例，AISBench工具的详细使用方法请参见[AISBench工具](https://gitee.com/aisbench/benchmark)。

1. 使用以下命令下载并安装AISBench工具。

    ```
    git clone https://gitee.com/aisbench/benchmark.git 
    cd benchmark/ 
    pip3 install -e ./ --use-pep517
    pip3 install -r requirements/api.txt 
    pip3 install -r requirements/extra.txt
    ```

    > [!NOTE]说明 
    > pip安装方式适用于使用AISBench最新功能的场景（镜像安装MindIE方式除外）。AISBench工具已预装在MindIE镜像中，可使用以下命令查看AISBench工具在MindIE镜像中的安装路径。
    >```
    >pip show ais_bench_benchmark
    >```

2.  准备数据集。

    以gsm8k为例，单击[gsm8k数据集](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载数据集，将解压后的gsm8k文件夹部署到工具根路径的ais\_bench/datasets文件夹下。

3.  配置ais\_bench/benchmark/configs/models/vllm\_api/vllm\_api\_stream\_chat.py文件，示例如下所示。

    ```
    from ais_bench.benchmark.models import VLLMCustomAPIChatStream  
    models = [     
        dict(         
            attr="service",         
            type=VLLMCustomAPIChatStream,         
            abbr='vllm-api-stream-chat',         
            path="/home/weight",                    # 指定模型序列化词表文件绝对路径，一般来说就是模型权重文件夹路径        
            model="qwen2-7b",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）        
            request_rate = 0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求        
            retry = 2,         
            host_ip = "127.0.0.1",      # 指定推理服务的IP        
            host_port = 1025,           # 指定推理服务的端口        
            max_out_len = 512,          # 推理服务输出的token的最大数量        
            batch_size=1,               # 请求发送的最大并发数
            trust_remote_code=False,         
            generation_kwargs = dict(             
                temperature = 0.5,             
                top_k = 10,             
                top_p = 0.95,             
                seed = None,             
                repetition_penalty = 1.03,                                
            ) , 
             pred_postprocessor=dict(type=extract_non_reasoning_content)   
        ) 
    ]
    ```

4.  执行以下命令启动服务化精度测试。

    ```
    ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
    ```

    回显如下所示则表示执行成功：

    ```
    dataset                 version  metric   mode  vllm_api_general_chat 
    ----------------------- -------- -------- ----- ---------------------- 
    demo_gsm8k              401e4c   accuracy gen                   62.50
    ```

## 性能测试

> [!NOTE]说明 
> 以下性能测试以AISBench工具为例，AISBench工具的详细使用方法请参见[AISBench工具](https://gitee.com/aisbench/benchmark)。

1.  使用以下命令下载并安装AISBench工具。

    ```
    git clone https://gitee.com/aisbench/benchmark.git 
    cd benchmark/ 
    pip3 install -e ./ --use-pep517
    pip3 install -r requirements/api.txt 
    pip3 install -r requirements/extra.txt
    ```

    > [!NOTE]说明 
    > pip安装方式适用于使用AISBench最新功能的场景（镜像安装MindIE方式除外）。AISBench工具已预装在MindIE镜像中，可使用以下命令查看AISBench工具在MindIE镜像中的安装路径。
    >```
    >pip show ais_bench_benchmark
    >```

2.  准备数据集。

    以gsm8k为例，单击[gsm8k数据集](https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载数据集，将解压后的gsm8k/文件夹部署到工具根路径的ais\_bench/datasets文件夹下。

3.  配置ais\_bench/benchmark/configs/models/vllm\_api/vllm\_api\_stream\_chat.py文件，示例如下所示。

    ```
    from ais_bench.benchmark.models import VLLMCustomAPIChatStream  
    models = [     
        dict(         
            attr="service",         
            type=VLLMCustomAPIChatStream,         
            abbr='vllm-api-stream-chat',         
            path="/home/weight",                    # 指定模型序列化词表文件绝对路径，一般来说就是模型权重文件夹路径        
            model="qwen2-7b",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）        
            request_rate = 0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求        
            retry = 2,         
            host_ip = "127.0.0.1",      # 指定推理服务的IP        
            host_port = 1025,           # 指定推理服务的端口        
            max_out_len = 512,          # 推理服务输出的token的最大数量        
            batch_size=1,               # 请求发送的最大并发数
            trust_remote_code=False,        
            generation_kwargs = dict(             
                temperature = 0.5,             
                top_k = 10,             
                top_p = 0.95,             
                seed = None,             
                repetition_penalty = 1.03,             
                ignore_eos = True,      # 推理服务输出忽略eos（输出长度一定会达到max_out_len）        
            ) , 
             pred_postprocessor=dict(type=extract_non_reasoning_content)    
        ) 
    ]
    ```

4.  执行以下命令启动服务化性能测试。

    ```
    ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --mode perf --debug
    ```

    回显如下所示则表示执行成功：

    ```

    │ Performance Parameters │ Stage  │ Average        │ Min          │ Max        │ Median       │ P75        │ P90          │ P99          │ N │ 
    │ E2EL                   │total   │ 2048.2945  ms  │ 1729.7498 ms │ 3450.96 ms │ 2491.8789 ms │ 2750.85 ms │ 3184.9186 ms │ 3424.4354 ms │ 8 │
    │ TTFT                   │total   │ 50.332 ms      │ 50.6244 ms   │ 52.0585 ms │ 50.3237 ms   │ 50.5872 ms │ 50.7566 ms   │ 50.0551 ms   │ 8 │
    │ TPOT                   │total   │ 10.6965 ms     │ 10.061 ms    │ 10.8805 ms │ 10.7495 ms   │ 10.7818 ms │ 10.808 ms    │ 10.8582 ms   │ 8 │ 
    │ ITL                    │total   │ 10.6965 ms     │ 7.3583 ms    │ 13.7707 ms │ 10.7513 ms   │ 10.8009 ms │ 10.8358 ms   │ 10.9322 ms   │ 8 │ 
    │ InputTokens            │total   │ 1512.5         │ 1481.0       │ 1566.0     │ 1511.5       │ 1520.25    │ 1536.6       │ 1563.06      │ 8 │ 
    │ OutputTokens           │total   │ 287.375        │ 200.0        │ 407.0      │ 280.0        │ 322.75     │ 374.8        │ 403.78       │ 8 │ 
    │ OutputTokenThroughput  │total   │ 115.9216       │ 107.6555     │ 116.5352   │ 117.6448     │ 118.2426   │ 118.3765     │ 118.6388     │ 8 │
    
    ```

    ```
    
    │ Common Metric            │ Stage    │ Value              │ 
    │ Benchmark Duration       │ total    │ 19897.8505 ms      │ 
    │ Total Requests           │ total    │ 8                  │ 
    │ Failed Requests          │ total    │ 0                  │ 
    │ Success Requests         │ total    │ 8                  │ 
    │ Concurrency              │ total    │ 0.9972             │ 
    │ Max Concurrency          │ total    │ 1                  │ 
    │ Request Throughput       │ total    │ 0.4021 req/s       │ 
    │ Total Input Tokens       │ total    │ 12100              │ 
    │ Prefill Token Throughput │ total    │ 17014.3123 token/s │ 
    │ Total generated tokens   │ total    │ 2299               │ 
    │ Input Token Throughput   │ total    │ 608.7438 token/s   │ 
    │ Output Token Throughput  │ total    │ 115.7835 token/s   │ 
    │ Total Token Throughput   │ total    │ 723.5273 token/s   │ 
    
    
    ```

    性能测试结果主要关注TTFT、TPOT、Request Throughput和Output Token Throughput输出参数，参数详情信息请参见《MindIE Motor开发指南》中的“配套工具 \> 性能/精度测试工具”章节的“表2 性能测试结果指标对比”。

    > [!NOTE]说明 
    > 任务执行的过程最终会落盘在默认的输出路径，该输出路径在运行中的打印日志中有提示，日志内容如下所示：
    >```
    > 08/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250828_151326
    >```
    > 命令执行结束后，outputs/default/20250828\_151326中的任务执行的详情如下所示：
    >```
    > 20250828_151326           # 每次实验基于时间戳生成的唯一目录 
    >├── configs               # 自动存储的所有已转储配置文件 
    >├── logs                  # 执行过程中日志，命令中如果加--debug，不会有过程日志落盘（都直接打印出来了） 
    >│   └── performance/      # 推理阶段的日志文件 
    >└── performance           # 性能测评结果 
    >│    └── vllm-api-stream-chat/          # “服务化模型配置”名称，对应模型任务配置文件中models的 abbr参数 
    >│         ├── gsm8kdataset.csv          # 单次请求性能输出（CSV），与性能结果打印中的Performance Parameters表格一致 
    >│         ├── gsm8kdataset.json         # 端到端性能输出（JSON），与性能结果打印中的Common Metric表格一致 
    >│         ├── gsm8kdataset_details.json # 全量打点日志（JSON） 
    >│         └── gsm8kdataset_plot.html    # 请求并发可视化报告（HTML）
    >```