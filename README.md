# SGLang


## 📢 Latest News

- [2025/12] MindIE LLM 正式宣布开源并面向公众开放！ [会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-LLM)

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/verylucky01/MindIE-LLM)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/verylucky01/MindIE-LLM)

</div>

## 🚀 简介

**MindIE LLM**是昇腾的大语言模型推理加速套件，旨在通过深度优化的模型库和推理优化器，专门提升大模型在昇腾硬件上的推理性能和易用性。MindIE LLM基于昇腾硬件，提供业界通用大模型推理能力，多并发请求的调度，包含Continuous Batching、PagedAttention、FlashDecoding等加速特性，使能用户高性能推理需求。

## 🔍 目录结构

``` 
 ├── mindie_llm                                     # Python 推理框架主模块 
 │   ├── connector                                  # 请求接入层 
 │   ├── text_generator                             # 核心推理引擎 
 │   ├── modeling                                   # 模型封装抽象  
 │   ├── runtime                                    # 运行时编译和模型加载
 │   ├── utils                                      # 工具模块：日志/张量/Profiling/验证等 
 ├── examples                                       # 示例代码 
 ├── docs                                           # 项目文档介绍 
 ├── src                                            # C++ 核心引擎 
 │   ├── engine                                     # LLM 引擎的主逻辑（调度/执行） 
 │   ├── scheduler                                  # 调度器（FCFS/PDDS/Layerwise） 
 │   ├── block_manager                              # KV Cache 块管理（LRU/Prefix Cache/CoW） 
 │   ├── llm_manager                                # Python/C++ 桥接 API  
 │   ├── server                                     # 服务端（gRPC/HTTP 接入端点） 
 │   ├── utils                                      # 基础工具（共享内存/加密/日志/ID 生成等） 
 │   ├── include                                    # 对外头文件接口 
 ├── scripts                                        # 构建与部署脚本 
 ├── tools                                          # 辅助工具 
 ├── tests                                          # 测试                                             
 ├── CMakeLists.txt                                 # CMake 构建配置                          
 ├── README.md    
                              
 ```
## 📢 版本说明

| MindIE 软件版本&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| CANN 版本兼容性&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|:----------------------------|:----------------------------|
| 2.3.0 | 8.5.0 |


## ⚡️ 环境部署

- 通过软件包或镜像方式安装MindIE LLM，请参见[安装指南](./docs/zh/user_guide/install/README.md)。
- 通过拉取最新代码编译安装MindIE LLM，请参见[编译安装指南](./docs/zh/developer_guide/build_guide.md)。

## ⚡️ 快速入门

快速体验使用MindIE进行大模型推理的全流程，请参见[快速入门](./docs/zh/user_guide/quick_start/quick_start.md)。

## 📝 学习文档

- 模型支持列表
  - [代码仓模型支持列表](./docs/zh/user_guide/model_support_list.md)：优先使用，提供当前版本经过测试充分验证支持和仅功能支持的模型全集。
  - [昇腾社区模型支持列表](https://www.hiascend.com/software/mindie/modellist)：提供当前版本经过测试充分验证支持的模型。
- [特性介绍](./docs/zh/user_guide/feature/README.md)：MindIE LLM 支持的推理特性。
- [LLM 使用指南](./docs/zh/user_guide/user_manual/README.md)：MindIE LLM 使用指南，包括推理参数配置、在线和离线推理、参数调优等。
  
## 📝贡献声明

1. 提交错误报告：如果您在MindIE LLM中发现了一个不存在安全问题的漏洞，请在MindIE LLm仓库中的Issues搜索，以防该漏洞被重复提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息，可以尝试解决其中的某个问题。
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：
    <br>a. Fork本项目的仓库。
    <br>b. Clone到本地。
    <br>c. 创建开发分支。
    <br>d. 本地自测，提交前请通过所有的单元测试，包括为您要解决的问题新增的单元测试。
    <br>e. 提交代码。
    <br>f. 新建Pull Request。
    <br>g. 代码检视，您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。
    <br>h. 当您的PR获取足够数量的检视者批准后，Committer会进行最终审核。
    <br>i. 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。

更多贡献相关文档请参见[贡献指南](contributing.md)。


## 📝免责声明

版权所有© 2025-2026 MindIE Project.

您对 "本文档" 的复制、使用、修改及分发受知识共享（Creative Commons，CC）署名 —— 相同方式共享 4.0 国际公共许可协议（以下简称 "CC BY-SA 4.0"）的约束。为了方便用户理解，您可以通过访问 [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/) 了解 CC BY-SA 4.0 的概要（但不是替代）。关于 CC BY-SA 4.0 的完整协议内容，您可以访问如下网址获取：[https://creativecommons.org/licenses/by-sa/4.0/legalcode](https://creativecommons.org/licenses/by-sa/4.0/legalcode)。


## 🌟 相关信息

- [安全声明](./security.md)
- [LICENSE](LICENSE.md)
