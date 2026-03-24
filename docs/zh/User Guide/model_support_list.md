# MindIE-LLM 支持模型列表

介绍支持的模型和模型对应支持的硬件和特性。参考模板在下
> **图例说明**  
> - **<span style="font-size: 1.4em; color:green">●</span>**：充分验证支持  
> - **<span style="font-size: 1.4em; color: #FFA500;">●</span>**：仅功能支持  
> - ✅：支持该列的特性 
> - ❌：不支持该列特性

---

## 文本生成模型

### DeepSeek 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| DeepSeek-R1-Distill-Llama-8B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-r1-distill/DeepSeek-R1-Distill-Llama-8B.md) | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| DeepSeek-R1-Distill-Llama-70B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡<br>Atlas 800I A3：4卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-r1-distill/DeepSeek-R1-Distill-Llama-70B.md) | [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| DeepSeek-R1-Distill-Qwen-1.5B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)<br>Atlas 800I A3：1/2卡(推荐1卡)<br>Atlas 300I Duo：1/2卡（推荐1卡） | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-r1-distill/README_DeepSeek-R1-Distill-Qwen-1.5B.md) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| DeepSeek-R1-Distill-Qwen-7B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)<br>Atlas 800I A3：1/2卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-r1-distill/README_DeepSeek-R1-Distill-Qwen-7B.md) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| DeepSeek-R1-Distill-Qwen-14B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐4卡)<br>Atlas 800I A3：1/2/4卡(推荐2卡)<br>Atlas 300I Duo：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-r1-distill/DeepSeek-R1-Distill-Qwen-14B.md) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |
| DeepSeek-R1-Distill-Qwen-32B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐4卡)<br>Atlas 800I A3：1/2/4卡(推荐2卡)<br>Atlas 300I Duo：1/2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-r1-distill/DeepSeek-R1-Distill-Qwen-32B.md) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| DeepSeek-MoE-16B-Chat | <span style="font-size: 1.4em; color: #FFA500;">●</span> | 4k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡（推荐4卡）<br>Atlas 800I A3：不支持<br>Atlas 300I Duo：不支持  | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek/README_deepseek_moe.md) | [deepseek-ai/deepseek-moe-16b-chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) | [deepseek-ai/deepseek-moe-16b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-moe-16b-chat) |
| DeepSeek-V2-Chat-236B | <span style="font-size: 1.4em; color:green">●</span> | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseekv2/README.md) | [deepseek-ai/DeepSeek-V2-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat) | [deepseek-ai/DeepSeek-V2-Chat](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Chat) |
| DeepSeek-V3-0324 | <span style="font-size: 1.4em; color:green">●</span> | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-v3/README.md) | [deepseek-ai/DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) | [deepseek-ai/DeepSeek-V3-0324](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3-0324) |
| DeepSeek-R1-0528 | <span style="font-size: 1.4em; color:green">●</span> | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-r1/README.md) | [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) | [deepseek-ai/DeepSeek-R1-0528](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528) |
| DeepSeek-V3.1 | <span style="font-size: 1.4em; color:green">●</span> | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-v3/README.md) | [deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | [deepseek-ai/DeepSeek-V3.1](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1) |
| DeepSeek-V3.1-Terminus | <span style="font-size: 1.4em; color:green">●</span> | 128k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/deepseek-v3/README.md) | [deepseek-ai/DeepSeek-V3.1-Terminus](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) | [deepseek-ai/DeepSeek-V3.1-Terminus](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1-Terminus) |

---

### Qwen 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Qwen2-1.5B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)<br>Atlas 800I A3：1/2卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) | [Qwen/Qwen2-1.5B](https://modelscope.cn/models/qwen/Qwen2-1.5B) |
| Qwen2-7B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4(推荐2卡)<br>Atlas 800I A3：1/2卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | [Qwen/Qwen2-7B-Instruct](https://modelscope.cn/models/qwen/Qwen2-7B-Instruct) |
| Qwen2-72B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐8卡)<br>Atlas 800I A3：1/2/4卡(推荐4卡)<br>Atlas 300I Duo：1/2/4卡(推荐4卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) | [Qwen/Qwen2-72B-Instruct](https://modelscope.cn/models/qwen/Qwen2-72B-Instruct) |
| Qwen2.5-0.5B-Instruct | <span style="font-size:1.4em;color:#FFA500;">●</span> | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2卡(推荐2卡)<br>Atlas 800I A3：1卡<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct) |
| Qwen2.5-1.5B-Instruct | <span style="font-size:1.4em;color:#FFA500;">●</span> | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4卡(推荐2卡)<br>Atlas 800I A3：1/2卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | [Qwen/Qwen2.5-1.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-1.5B-Instruct) |
| Qwen2.5-3B-Instruct | <span style="font-size:1.4em;color:#FFA500;">●</span> | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | [Qwen/Qwen2.5-3B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-3B-Instruct) |
| Qwen2.5-7B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4(推荐2卡)<br>Atlas 800I A3：1/2卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [Qwen/Qwen2.5-7B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct) |
| Qwen2.5-14B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)<br>Atlas 800I A3：1/2/4卡(推荐2卡)<br>Atlas 300I Duo：1/2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | [Qwen/Qwen2.5-14B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-14B-Instruct) |
| Qwen2.5-32B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)<br>Atlas 800I A3：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | [Qwen/Qwen2.5-32B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-32B-Instruct) |
| Qwen2.5-72B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡<br>Atlas 800I A3：4卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | [Qwen/Qwen2.5-72B-Instruct](https://modelscope.cn/models/qwen/Qwen2.5-72B-Instruct) |
| Qwen3-0.6B | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1卡<br>Atlas 800I A3：1卡<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | [Qwen/Qwen3-0.6B](https://modelscope.cn/models/qwen/Qwen3-0.6B) |
| Qwen3-1.7B | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1卡<br>Atlas 800I A3：1卡<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | [Qwen/Qwen3-1.7B](https://modelscope.cn/models/qwen/Qwen3-1.7B) |
| Qwen3-4B | <span style="font-size:1.4em;color:green">●</span> | 128k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [Qwen/Qwen3-4B](https://modelscope.cn/models/qwen/Qwen3-4B) |
| Qwen3-8B | <span style="font-size:1.4em;color:green">●</span> | 128k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1卡<br>Atlas 800I A3：1卡<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [Qwen/Qwen3-8B](https://modelscope.cn/models/qwen/Qwen3-8B) |
| Qwen3-14B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：2卡<br>Atlas 800I A3：1卡<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | [Qwen/Qwen3-14B](https://modelscope.cn/models/qwen/Qwen3-14B) |
| Qwen3-32B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：4卡<br>Atlas 800I A3：2卡<br>Atlas 300I Duo：2卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | [Qwen/Qwen3-32B](https://modelscope.cn/models/qwen/Qwen3-32B) |
| Qwen3-30B-A3B-Instruct-2507 | <span style="font-size:1.4em;color:green">●</span> | 128k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡（推荐4卡）<br>Atlas 800I A3：2/4卡（推荐2卡）<br>Atlas 300I Duo：2卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://modelscope.cn/models/qwen/Qwen3-30B-A3B-Instruct-2507) |
| Qwen3-235B-A22B-Instruct-2507 | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：4卡8芯 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) | [Qwen/Qwen3-235B-A22B-Instruct-2507](https://modelscope.cn/models/qwen/Qwen3-235B-A22B-Instruct-2507) |
| Qwen3-Coder-30B-A3B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：4卡8芯 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://modelscope.cn/models/qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Qwen3-Coder-480B-A35B-Instruct | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：4卡8芯 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen/README.md) | [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct) | [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://modelscope.cn/models/qwen/Qwen3-Coder-480B-A35B-Instruct) |

---

### Llama 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Llama3-8B | <span style="font-size:1.4em;color:green">●</span> | 8k | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/llama3/README.md) | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | [LLM-Research/Meta-Llama-3-8B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct) |
| Llama3-70B | <span style="font-size:1.4em;color:green">●</span> | 8k | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡<br>Atlas 800I A3：4卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/llama3/README.md) | [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) | [LLM-Research/Meta-Llama-3-70B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-70B) |
| Llama3.1-8B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/llama3/README.md) | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | [LLM-Research/Meta-Llama-3.1-8B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B) |
| Llama3.1-70B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Atlas 800I A2：8卡<br>Atlas 800I A3：4卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/llama3/README.md) | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B) | [LLM-Research/Meta-Llama-3.1-70B/Meta-Llama-3.1-70B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B) |
| Llama3.1-405B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/llama3/README.md) | [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B) | [LLM-Research/Meta-Llama-3.1-405B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-405B-Instruct) |

---

### ERNIE 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| ERNIE-4.5-300B-A47B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：8卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/ernie_moe/README.md) | [baidu/ERNIE-4.5-300B-A47B-PT](https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-PT) | [baidu/ERNIE-4.5-300B-A47B](https://modelscope.cn/models/baidu/ERNIE-4.5-300B-A47B) |


---

### KIMI 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Kimi-K2-Instruct | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：32卡<br>Atlas 800I A3：16卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/kimi_k2/README.md) | [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) | ❌ |
| Kimi-K2-Thinking | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：32卡<br>Atlas 800I A3：16卡<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/kimi_k2/README.md) | [moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | ❌ |

---

### GLM 系列

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| ChatGLM2-6B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 32k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/chatglm/v2_6b/README.md) | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) | [ZhipuAI/chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b) |
| ChatGLM3-6B | <span style="font-size:1.4em;color:green">●</span> | 8k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/chatglm/v3_6b/README.md) | [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) | [ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b) |
| ChatGLM3-6B-32K | <span style="font-size:1.4em;color:#FFA500;">●</span> | 32k | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/chatglm/v3_6b/README.md) | [THUDM/chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k) | [ZhipuAI/chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k) |
| GLM4-9B | <span style="font-size:1.4em;color:green">●</span> | 128k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/chatglm/v4_9b/README.md) | [THUDM/glm-4-9b](https://huggingface.co/THUDM/glm-4-9b) | [ZhipuAI/glm-4-9b](https://modelscope.cn/models/ZhipuAI/glm-4-9b) |
| GLM-4.5 | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：16卡<br>Atlas 800I A3：待测试<br>Atlas 300I Duo：不支持 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/glm/README.md) | [zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5) | ❌ |

---

### Bloom

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Bloom-7B | <span style="font-size:1.4em;color:green">●</span> | 4096 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/bloom/README.md) | [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1) | [bigscience/bloom-7b1](https://modelscope.cn/models/bigscience/bloom-7b1) |

---

### Baichuan

| 模型 | Support | 最大上下文长度 | W8A8 | W4A8 | W8A16 | LoRA | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Baichuan2-7B | <span style="font-size:1.4em;color:green">●</span> | 4096 | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/baichuan/README.md) | [baichuan-inc/Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan-7B) | [baichuan/Baichuan2-7B](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base) |
| Baichuan2-13B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4096 | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/baichuan/README.md) | [baichuan-inc/Baichuan2-13B](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) | [baichuan/Baichuan2-13B](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base) |

---

## 多模态理解模型

### Qwen 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Qwen2-VL-2B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | [Qwen/Qwen2-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct) |
| Qwen2-VL-7B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | [Qwen/Qwen2-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct) |
| Qwen2-VL-72B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)<br>Atlas 800I A3：2/4卡(推荐4卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) | [Qwen/Qwen2-VL-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-72B-Instruct) |
| Qwen2.5-VL-3B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | [Qwen/Qwen2.5-VL-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL-7B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [Qwen/Qwen2.5-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen2.5-VL-32B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)<br>Atlas 800I A3：2/4卡(推荐2卡)<br>Atlas 300I Duo：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) | [Qwen/Qwen2.5-VL-32B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct) |
| Qwen2.5-VL-72B | <span style="font-size:1.4em;color:green">●</span> | 32k | ✅ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)<br>Atlas 800I A3：2/4卡(推荐4卡)<br>Atlas 300I Duo：2/4卡(推荐4卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) | [Qwen/Qwen2.5-VL-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct) |
| QVQ-72B-Preview | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)<br>Atlas 800I A3：2/4卡(推荐4卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/QVQ-72B-Preview](https://huggingface.co/Qwen/QVQ-72B-Preview) | [Qwen/QVQ-72B-Preview](https://modelscope.cn/models/Qwen/QVQ-72B-Preview) |
| Qwen2-Audio-7B | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/qwen2_vl/README.md) | [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | [Qwen/Qwen2-Audio-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct) |
| Qwen3-VL-2B | <span style="font-size:1.4em;color:green">●</span> | 256K | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)<br>Atlas 300I Duo：1/2/4/8卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/qwen3_vl/README.md) | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | [Qwen/Qwen3-VL-2B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct) |
| Qwen3-VL-4B | <span style="font-size:1.4em;color:green">●</span> | 256K | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐1卡)<br>Atlas 300I Duo：1/2/4/8卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/qwen3_vl/README.md) | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | [Qwen/Qwen3-VL-4B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen3-VL-8B | <span style="font-size:1.4em;color:green">●</span> | 256K | ✅ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 300I Duo：1/2/4/8卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/qwen3_vl/README.md) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [Qwen/Qwen3-VL-8B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-32B | <span style="font-size:1.4em;color:green">●</span> | 256K | ✅ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)<br>Atlas 300I Duo：2/4/8卡(推荐4卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/qwen3_vl/README.md) | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | [Qwen/Qwen3-VL-32B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct) |
| Qwen3-VL-30B-A3B | <span style="font-size:1.4em;color:green">●</span> | 256k | ✅ | ✅ | ❌ | Atlas 800I A2：2/4卡(推荐2卡)<br>Atlas 300I Duo：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/dev/examples/atb_models/examples/models/qwen3_vl/README.md) | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct) |

---

### InternVL 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| InternVL2-8B | <span style="font-size:1.4em;color:green">●</span> | 8k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/internvl/README.md) | [OpenGVLab/InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) | [OpenGVLab/InternVL2-8B](https://modelscope.cn/models/OpenGVLab/InternVL2-8B) |
| InternVL2-40B | <span style="font-size:1.4em;color:green">●</span> | 8k | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)<br>Atlas 800I A3：1/2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/internvl/README.md) | [OpenGVLab/InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B) | [OpenGVLab/InternVL2-40B](https://modelscope.cn/models/OpenGVLab/InternVL2-40B) |
| InternVL2.5-8B  | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/internvl/README.md) | [OpenGVLab/InternVL2_5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) | [OpenGVLab/InternVL2_5-8B](https://modelscope.cn/models/OpenGVLab/InternVL2_5-8B) |
| InternVL2.5-78B  | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：8卡<br>Atlas 800I A3：4卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/internvl/README.md) | [OpenGVLab/InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B) | [OpenGVLab/InternVL2_5-78B](https://modelscope.cn/models/OpenGVLab/InternVL2_5-78B) |

---

### GLM 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| GLM-4V-9B | <span style="font-size:1.4em;color:green">●</span> | 128k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/glm4v/README.md) | [zai-org/glm-4v-9b](https://huggingface.co/zai-org/glm-4v-9b) | [ZhipuAI/glm-4v-9b](https://modelscope.cn/models/ZhipuAI/glm-4v-9b) |
| GLM-4.1V-9B-Thinking | <span style="font-size:1.4em;color:green">●</span> | 64k | ✅ | ✅ | ❌ | Atlas 800I A2：1/2卡(推荐2卡)<br>Atlas 800I A3：1卡<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/glm41v/README.md) | [zai-org/GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) | [ZhipuAI/GLM-4.1V-9B-Thinking](https://modelscope.cn/models/ZhipuAI/GLM-4.1V-9B-Thinking) |

---

### MiniCPM-V

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| MiniCPM-V 2.6 | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2卡(推荐2卡)<br>Atlas 800I A3：1卡<br>Atlas 300I Duo：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/minicpm_qwen2_v2/README.md) | [openbmb/MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6) | [OpenBMB/MiniCPM-V-2_6](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6) |

---

### VITA

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| VITA-1.5 | <span style="font-size:1.4em;color:green">●</span> | 32k | ❌ | ✅ | ❌ | Atlas 800I A2：1卡<br>Atlas 800I A3：1卡 | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/vita/README.md) | [VITA-MLLM/VITA-1.5](https://huggingface.co/VITA-MLLM/VITA-1.5) | [VITA-MLLM/VITA-1.5](https://modelscope.cn/models/VITA-MLLM/VITA-1.5) |

### LLaVa 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| LLaVa-1.5-7B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/llava/README.md) | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | [llava-hf/llava-1.5-7b-hf](https://modelscope.cn/models/llava-hf/llava-1.5-7b-hf) |
| LLaVa-1.5-13B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)<br>Atlas 800I A3：1/2/4卡(推荐2卡)<br>Atlas 300I Duo：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/llava/README.md) | [llava-hf/llava-1.5-13b-hf](https://huggingface.co/llava-hf/llava-1.5-13b-hf) | [llava-hf/llava-1.5-7b-hf](https://modelscope.cn/models/llava-hf/llava-1.5-13b-hf) |
| LLaVa-v1.6-7B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/llava/README.md) | [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) | [llava-hf/llava-v1.6-vicuna-7b-hf](https://modelscope.cn/models/llava-hf/llava-v1.6-vicuna-7b-hf) |
| LLaVa-v1.6-13B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：2/4/8卡(推荐4卡)<br>Atlas 800I A3：1/2/4卡(推荐2卡)<br>Atlas 300I Duo：1/2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/llava/README.md) | [llava-hf/llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) | [llava-hf/llava-v1.6-vicuna-13b-hf](https://modelscope.cn/models/llava-hf/llava-v1.6-vicuna-13b-hf) |
| LLaVa-v1.6-34B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)<br>Atlas 800I A3：2/4卡(推荐2卡)<br>Atlas 300I Duo：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/llava/README.md) | [llava-hf/llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) | [llava-hf/llava-v1.6-34b-hf](https://modelscope.cn/models/llava-hf/llava-v1.6-34b-hf) |
| LLaVa-NeXT-Video-7B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/llava/README.md) | [llava-hf/LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) | [llava-hf/LLaVA-NeXT-Video-7B-hf](https://modelscope.cn/models/llava-hf/LLaVA-NeXT-Video-7B-hf) |
| LLaVa-NeXT-Video-34B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)<br>Atlas 800I A3：2/4卡(推荐2卡)<br>Atlas 300I Duo：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/llava/README.md) | [llava-hf/LLaVA-NeXT-Video-34B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-34B-hf) | [llava-hf/LLaVA-NeXT-Video-34B-hf](https://modelscope.cn/models/llava-hf/LLaVA-NeXT-Video-34B-hf) |

---

### Llama 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Llama 3.2-Vision-11B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 128k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/mllama/README.md) | [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | [LLM-Research/Llama-3.2-11B-Vision-Instruct](https://modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct) |
| Llama 3.2-Vision-90B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 128k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐8卡)<br>Atlas 800I A3：2/4卡(推荐4卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/mllama/README.md) | [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct) | [LLM-Research/Llama-3.2-90B-Vision-Instruct](https://modelscope.cn/models/LLM-Research/Llama-3.2-90B-Vision-Instruct) |

---

### Yi-VL 系列

| 模型 | Support | 最大上下文长度 | W8A8 | atbgraph | aclgraph | 硬件规格 | README 链接 | Hugging Face 链接 | ModelScope 链接 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|
| Yi-VL-6B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：1/2/4/8卡(推荐2卡)<br>Atlas 800I A3：1/2/4卡(推荐1卡)<br>Atlas 300I Duo：1/2/4卡(推荐1卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/yivl/README.md) | [01-ai/Yi-VL-6B](https://huggingface.co/01-ai/Yi-VL-6B) | [01ai/Yi-VL-6B](https://www.modelscope.cn/models/01ai/Yi-VL-6B) |
| Yi-VL-34B | <span style="font-size:1.4em;color:#FFA500;">●</span> | 4k | ❌ | ✅ | ❌ | Atlas 800I A2：4/8卡(推荐4卡)<br>Atlas 800I A3：2/4卡(推荐2卡)<br>Atlas 300I Duo：2/4卡(推荐2卡) | [README](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/models/yivl/README.md) | [01-ai/Yi-VL-34B](https://huggingface.co/01-ai/Yi-VL-34B) | [01ai/Yi-VL-34B](https://www.modelscope.cn/models/01ai/Yi-VL-34B) |
