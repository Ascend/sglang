# Anti-Outlier 离群值抑制

介绍特性是什么，如何开启/使用

> [!TIP] 同义术语
> 本特性在不同文档或工具日志中可能被称为：**离群值抑制**、**异常值抑制**、**Anti-Outlier** 或 **AntiOutlier**。

## 特性简介

离群值抑制（Anti-Outlier）主要用于解决模型量化中因激活值分布存在异常值（Outlier）而导致的精度损失问题。在大模型量化过程中，如果激活值中存在数值极大的离群点，会拉大量化区间（Scale），导致大部分正常数值的量化分辨率降低，从而严重影响模型精度。该技术通过对离群值进行平滑或抑制处理，有效改善数据分布，确保量化后的模型仍能保持较高的推理精度。

> [!NOTE]说明 
> Anti-Outlier可以配合其他量化方式一起使用，如 W4A8、W8A8、W8A8C8 等。

以下展示了W8A8 + Anti-Outlier + PDMIX量化后权重描述文件 `quant_model_description.json` 中的部分内容：

```json
{
  "model.layers.0.self_attn.q_proj.weight": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.bias": "FLOAT",				//optional
  "model.layers.0.self_attn.q_proj.quant_bias": "W8A8_MIX",
  "model.layers.0.self_attn.q_proj.input_scale": "W8A8_MIX",
  //...
  "model.layers.0.mlp.down_proj.weight_offset": "W8A8_MIX",
  "model.layers.0.input_layernorm.weight": "FLOAT",
  "model.layers.0.input_layernorm.bias": "FLOAT",				//optional
  "model.layers.0.post_attention_layernorm.weight": "FLOAT",
  "model.layers.0.post_attention_layernorm.bias": "FLOAT",		//optional
  //...
}
```

当前主流开源大模型（如LLaMA、Qwen等）通常使用RmsNorm作为`input_layernorm`和`post_attention_layernorm`。当启用非对称离群值抑制算法时，会引入额外的偏置项（以下称为`norm_bias`）。

为了保证计算等价性，该算法的执行逻辑如下：

1.  **Norm层引入**：在执行Norm操作时，将`norm_bias`加到权重中。
2.  **Linear层抵消**：在随后的Linear层计算前，将对应的`norm_bias`减去。

在实际的模型权重中，`norm_bias`在Linear层的抵消张量，会根据量化场景的不同，以不同方式融合：

-   **Per-tensor场景**：直接融合进Linear层的量化偏置`quant_bias`中。
    -   例如上文示例中的`model.layers.0.self_attn.q_proj.quant_bias`。
-   **Per-token场景**：体现为Linear层的普通偏置`bias`。
    -   例如上文示例中的`model.layers.0.self_attn.q_proj.bias`。

> [!NOTE]说明
> *   **若原Linear层已有Bias**（如Qwen2系列）：直接融合进原有的Bias中。
> *   **若原Linear层无Bias**（如Qwen3-32B）：创建一个新的Bias层来存储该值。

**性能优化建议**：
在PDMIX场景（P阶段使用Per-token量化）下，离群值抑制的bias对量化精度的影响通常较小。为提升性能，可考虑在保证等价性的前提下，在Norm层和Linear层同时移除Bias，从而节省一次Add计算开销。

**图 1**  量化权重推理时流程
![](./figures/anti_outlier_quantization.png "量化权重推理时流程-5")

**表 1**  权重量化后部分层的dtype及shape信息（假设原始权重的shape为 `[n]`）

|Tensor信息|input_layernorm.bias|post_attention_layernorm.bias|
|--|--|--|
|dtype|fp32|fp32|
|shape|[n]|[n]|

## 生成权重

您可以使用msModelSlim工具生成量化权重：[msModelSlim](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/README.md)

以Qwen3-14B为例，安装msModelSlim工具后，可以使用如下命令快速生成一份带有离群值抑制的W8A8PDMIX量化权重：

```sh
msmodelslim quant --model_path {浮点权重路径} --save_path {量化权重路径} --device npu --model_type Qwen3-14B --quant_type w8a8 --trust_remote_code True
```

执行上述命令会默认使用msModelSlim工具的最佳实践方式执行量化，如需了解更多量化参数配置，请参考msModelSlim工具文档。

## 执行推理

以Qwen3-14B-W8A8PDMIX权重为例，您可以使用以下指令执行对话测试，推理内容为"What's deep learning?"，最长输出20个token。

```bash
cd ${ATB_SPEED_HOME_PATH}
torchrun --nproc_per_node 2 --master_port 12350 -m examples.run_pa --model_path {量化权重路径}
```
