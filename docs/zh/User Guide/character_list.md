# 特性列表

介绍全部特性和特性价值。

MindIE LLM 支持的特性包括基础特性、量化特性、长序列特性、调度特性、加速特性和交互特性，每种特性的开启方式、限制等详细信息请参见`简介`中的链接。

<table>
    <tr>
        <th>大类</th><th>特性</th><th>简介</th><th>价值</th>
    </tr>
    <tr>
        <td rowspan="8">基础特性</td><td>Multi-Lora</td><td>使用不同的 LoRA 权重进行推理。详见 <a href="./multi_loRA.md">Multi-LoRA</a>。</td><td>支持 LoRA 特性，动态加载、卸载权重</td>
    </tr>
    <tr>
        <td>MoE</td><td>通过引入稀疏激活的专家网络，在不显著增加计算成本的前提下大幅扩展模型参数规模，从而提升模型能力。详见 <a href="./moe.md">MoE</a>。</td><td>以万亿级参数量容纳海量知识，性能潜力远超稠密模型。</td>
    </tr>
    <tr>
        <td>MLA</td><td>利用低秩键值联合压缩来消除推理时键值缓存的瓶颈，支持高效推理。详见 <a href="./mla.md">MLA</a>。</td><td>高效处理超长上下文</td>
    </tr>
    <tr>
        <td>负载均衡</td><td>降低 NPU 卡间的不均衡度，从而提升模型推理的性能。详见 <a href="./expert_parallelism_load_balancer.md">负载均衡</a>。</td><td>降低时延</td>
    </tr>
    <tr>
        <td>共享专家外置</td><td>将共享专家独立部署在单独的 NPU 卡上，与路由专家/冗余专家分离。详见 <a href="./mix_shared_routing.md">共享专家外置</a>。</td><td>优化 TPOT</td>
    </tr>
    <tr>
        <td>Expert Parallel</td><td>通过将专家分别部署在不同的设备上，实现专家级别的并行计算。详见 <a href="./expert_parallel.md">Expert Parallel</a>。</td><td>降低时延，提高吞吐</td>
    </tr>
    <tr>
        <td>Data Parallel</td><td>将推理请求划分为多个批次，并将每个批次分配给不同的设备进行并行处理。详见 <a href="./data_parallel.md">Data Parallel</a>。</td><td>提高吞吐</td>
    </tr>
    <tr>
        <td>Tensor Parallel</td><td>通过将张量（如权重矩阵、激活值等）在多个设备（如 NPU）之间进行切分，实现模型的分布式推理。详见 <a href="./tensor_parallel.md">Tensor Parallel</a>。</td><td>降低单卡显存</td>
    </tr>
    <tr>
        <td rowspan="10">量化特性</td><td>离群值抑制</td><td>通过抑制数据中的异常值，来提升大模型量化的精度。详见 <a href="./anti_outlier.md">离群值抑制</a>。</td><td>减少量化精度损失</td>
    </tr>
    <tr>
        <td>PD MIX 量化</td><td>在模型推理的 Prefill 和 Decode 阶段使用不同的量化方式。详见 <a href="./pdmix.md">PD MIX 量化</a>。</td><td>减低显存</td>
    </tr>
    <tr>
        <td>W8A8 量化</td><td>将权重和激活值统一量化为 int8 格式，以减少模型体积并加速推理计算。详见 <a href="./w8a8.md">W8A8 量化</a>。</td><td>降低显存、提高吞吐</td>
    </tr>
    <tr>
        <td>W4A8 混合量化</td><td>对模型不同层级采用不同量化方式，其中权重采用 4 位 / 8 位分级量化，激活统一采用 8 位量化。详见 <a href="./w4a8_mixed_precision_quantization.md">W4A8 混合量化</a>。</td><td>降低显存、提高吞吐</td>
    </tr>
    <tr>
        <td>W8A16 量化</td><td>仅将权重量化为 8 bit。详见 <a href="./w8a16.md">W8A16 量化</a>。</td><td>降低显存、提高吞吐</td>
    </tr>
    <tr>
        <td>Attention 量化</td><td>将 Q、K、V 统一量化为 8 bit，有效压缩 KV Cache 显存，加速解码阶段注意力计算，显著提升模型吞吐。详见 <a href="./attention_quantization.md">Attention 量化</a>。</td><td>降低显存、提高吞吐</td>
    </tr>
    <tr>
        <td>FA3 量化</td><td>采用类似 Attention 量化，区别在于对 k 的非 rope 张量进行 8 位量化，而 k 的 rope 张量不量化，以优化 KV 显存占用和解码速度，提升吞吐。详见 <a href="./fa3_quantization.md">FA3 量化</a>。</td><td>降低显存、提高吞吐</td>
    </tr>
    <tr>
        <td>KV Cache Int8</td><td>通过降低 KV 显存减少重计算来提升吞吐。详见 <a href="./kv_cache_int8.md">KV Cache Int8</a>。</td><td>降低显存、提高吞吐</td>
    </tr>
    <tr>
        <td>W8A8SC 稀疏量化</td><td>通过稀疏化将不重要的权重置零、将高精度数值转为低位宽存储，以及使用压缩算法进一步减小权重体积，从而实现模型的加速。详见 <a href="./w8a8sc.md">W8A8SC 稀疏量化</a>。</td><td>高稀疏率、降低显存、提高最大吞吐</td>
    </tr>
    <tr>
        <td>W16A16SC 量化</td><td>一种先通过算法稀疏化模型权重，再压缩存储的浮点稀疏量化方法。详见 <a href="./w16a16sc.md">W16A16SC 量化</a>。</td><td>高稀疏率、提高吞吐、避免反量化</td>
    </tr>
    <tr>
        <td rowspan="2">长序列特性</td><td>Context Parallel</td><td>通过将长序列在上下文维度进行切分，分配到不同设备并行处理，减少首 token 响应时间。详见 <a href="./context_parallel.md">Context Parallel</a>。</td><td>降低显存、降低首 token 时延</td>
    </tr>
    <tr>
        <td>Sequence Parallel</td><td>通过对 KV Cache 进行切分，使得每张卡保存的 KV Cache 各不相同，达到节省显存，支持长序列的功能。详见 <a href="./sequence_parallel.md">Sequence Parallel</a>。</td><td>降低显存</td>
    </tr>
    <tr>
        <td rowspan="3">调度特性</td><td>异步调度</td><td>对于 maxBatchSize 较大，且输入输出长度较长的场景，该特性使用模型推理阶段的耗时掩盖数据准备阶段和数据返回阶段的耗时，避免 NPU 计算资源和显存资源浪费。详见 <a href="./asynchronous_scheduling.md">异步调度</a>。</td><td>降低时延</td>
    </tr>
    <tr>
        <td>SplitFuse</td><td>将长提示词分解成更小的块，并在多个 forward step 中进行调度，降低 Prefill 时延。详见 <a href="./splitFuse.md">SplitFuse</a>。</td><td>降低显存和时延、提高吞吐</td>
    </tr>
    <tr>
        <td>SLO 调度优化</td><td>确保 SLO 的前提下提升系统吞吐量。详见 <a href="./slo_aware_scheduling_optimization.md">SLO 调度优化</a>。</td><td>提高吞吐</td>
    </tr>
    <tr>
        <td rowspan="5">加速特性</td><td>Micro Batch</td><td>批处理过程中，将数据切分为更小粒度的多个 batch 运行，使得硬件资源得以充分利用，以提高推理吞吐。详见 <a href="./micro_batch.md">Micro Batch</a>。</td><td>提高吞吐</td>
    </tr>
    <tr>
        <td>并行解码</td><td>利用算力优势弥补访存带宽受限的影响，提升算力利用率。详见 <a href="./speculative_decoding.md">并行解码</a>。</td><td>提高吞吐</td>
    </tr>
    <tr>
        <td>MTP</td><td>在推理过程中，模型不仅预测下一个 token，而且会同时预测多个 token，从而显著提升模型生成速度。详见 <a href="./mtp.md">MTP</a>。</td><td>提高吞吐</td>
    </tr>
    <tr>
        <td>Prefix Cache</td><td>复用跨请求的重复 Block 对应的 KV Cache，从而减少 Prefill 的时间。详见 <a href="./prefix_cache.md">Prefix Cache</a>。</td><td>降低首 token 时延</td>
    </tr>
    <tr>
        <td>KV Cache 池化</td><td>将 DRAM 甚至 SSD 等更大容量的存储介质纳入前缀缓存池，从而突破显存的容量限制。详见 <a href="./kv_cache_pool.md">KV Cache 池化</a>。</td><td>提高 Prefix Cache 命中率</td>
    </tr>
    <tr>
        <td rowspan="2">交互特性</td><td>Function Call</td><td>支持 Function Call 函数调用，使大模型具备使用工具能力。详见 <a href="./function_call.md">Function Call</a>。</td><td>能够借助外部工具来扩展应用范围</td>
    </tr>
    <tr>
        <td>思考解析</td><td>对大模型的输出内容进行结构化解析，将思考过程和输出结果进行分离。详见 <a href="./enable_reasoning.md">思考解析</a>。</td><td>提升复杂场景推理性能</td>
    </tr>
    <tr>
        <td rowspan="1">其他</td><td>权重离线切分</td><td>通过预切分权重至tmpfs，优化大规模模型加载效率，减少NPU传输时间。详见 <a href="./offline_weight_partitioning.md">权重离线切分</a>。</td><td>降低权重加载耗时</td>
    </tr>
</table>

# 特性叠加矩阵

若干特性的兼容性通过以下符号表示：

- ✅ = 完全兼容
- ❌ = 不兼容
- ❔ = 待定

> [!NOTE]说明
> - 对于 ❌ 或 ❔ 标注的情况，可以关联 [issues](https://gitcode.com/Ascend/MindIE-LLM/issues) 跟踪。
> - 这里仅列举主流模型 DeepSeek 和 Qwen。

## DeepSeek 模型

| 特性 | 负载均衡 | 共享专家外置 | Expert Parallel | Data Parallel | 离群值抑制 | PD MIX 量化 | W8A8 量化 | W4A8 混合量化 | FA3 量化 | Context Parallel | Sequence Parallel | 异步调度 | SLO 调度优化 | Micro Batch | MTP | Prefix Cache | KV Cache 池化 | Function Call | 思考解析 |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 负载均衡 | ✅ |
| 共享专家外置 | ✅ | ✅ |
| Expert Parallel | ✅ | ✅ | ✅ |
| Data Parallel | ✅ | ✅ | ✅ | ✅ |
| 离群值抑制 | ✅ | ✅ | ✅ | ✅ | ✅ |
| PD MIX 量化 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| W8A8 量化 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| W4A8 混合量化 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FA3 量化 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Context Parallel | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Sequence Parallel | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 异步调度 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SLO 调度优化 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Micro Batch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| MTP | ✅ |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Prefix Cache | ✅ |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| KV Cache 池化 | ✅ |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Function Call | ✅ |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| 思考解析 | ✅ |✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

> [!NOTE]说明
>- 对于 DeepSeek 模型，最大支持 Context Parallel + Sequence Parallel + prefix cache + KV Cache 池化 + MTP + 异步调度 + FA3 量化叠加，并支持 7 种特性自由组合。短序列（上下文长度短于16k）通常无需开启 Context Parallel 和 Sequence Parallel，长序列（上下文长度 128k）不能叠加 MTP 特性。

## Qwen 模型

| 特性 | Multi-Lora | 负载均衡 (仅支持 Qwen-MoE) | Data Parallel | 离群值抑制 | PD MIX 量化 | W8A8 量化 | W8A16 量化 | KV Cache int8 | W8A8SC 稀疏量化 | W16A16SC 稀疏量化 | 异步调度 | SplitFuse | SLO 调度优化 | Micro Batch | 并行解码 | Prefix Cache | KV Cache 池化 | Function Call | 思考解析 |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Multi-Lora | ✅ |
| 负载均衡 | ✅ | ✅ |
| Data Parallel | ✅ | ✅ | ✅ |
| 离群值抑制 | ❌ | ✅ | ✅ | ✅ |
| PD MIX 量化 | ❌ | ✅ | ✅ | ✅ | ✅ |
| W8A8 量化 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| W8A16 量化 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| KV Cache int8 | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| W8A8SC 稀疏量化 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| W16A16SC 稀疏量化 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 异步调度 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SplitFuse | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| SLO 调度优化 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| Micro Batch | ❌ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ❌ | ❌ | ❔ | ❔ | ❔ | ✅ |
| 并行解码 | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❔ | ✅ | ❌ | ❌ | ❌ | ❌ | ❔ | ✅ |
| Prefix Cache | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| KV Cache 池化 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Function Call | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❔ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| 思考解析 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
