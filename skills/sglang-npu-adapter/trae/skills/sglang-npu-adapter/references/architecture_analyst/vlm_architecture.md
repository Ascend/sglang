# VLM架构识别知识

## 1. VLM概述

视觉语言模型(Vision-Language Model)结合视觉编码器和语言模型，支持图像、视频等多模态输入。

### 1.1 VLM vs LLM

| 特性 | LLM | VLM |
|------|-----|-----|
| 输入 | 仅文本 | 文本 + 图像/视频 |
| 组件 | 语言模型 | 视觉编码器 + 投影层 + 语言模型 |
| 应用 | 文本生成 | 多模态理解与生成 |

---

## 2. VLM核心组件

### 2.1 Vision Tower（视觉编码器）

处理图像/视频输入，提取视觉特征：
- **类型**：ViT、CLIP、SigLIP等
- **输出**：视觉特征向量序列

**参考实现**：`python/sglang/srt/models/qwen2_vl.py`

### 2.2 Projector（投影层）

将视觉特征投影到语言模型的嵌入空间：
- **类型**：MLP、Q-Former等
- **作用**：维度对齐

### 2.3 Language Model（语言模型）

处理融合后的多模态表示：
- 与标准LLM结构类似
- 支持多模态token

---

## 3. VLM架构变体

### 3.1 独立Vision Tower

视觉编码器独立于语言模型：
- 代表：LLaVA系列
- 特点：视觉编码器权重冻结或微调

### 3.2 集成Vision Tower

视觉编码器与语言模型深度集成：
- 代表：Qwen2-VL
- 特点：端到端训练

### 3.3 多模态输入类型

| 类型 | 说明 | 代表模型 |
|------|------|----------|
| 图像 | 单张或多张图像 | LLaVA, Qwen2-VL |
| 视频 | 视频帧序列 | Qwen2-VL |
| 音频 | 音频输入 | Qwen2-Audio |

---

## 4. 配置文件关键字段

### 4.1 Vision配置

| 字段 | 说明 |
|------|------|
| vision_config | 视觉编码器配置 |
| image_token_index | 图像token索引 |
| video_token_index | 视频token索引 |

### 4.2 Projector配置

| 字段 | 说明 |
|------|------|
| projector_hidden_act | 投影层激活函数 |
| projector_hidden_size | 投影层隐藏维度 |

---

## 5. VLM适配要点

### 5.1 视觉编码器加载

- 检查vision_config是否正确
- 确认视觉编码器权重路径

### 5.2 多模态token处理

- 正确处理图像/视频token
- 确保位置编码正确

### 5.3 权重加载

- 视觉编码器权重
- 投影层权重
- 语言模型权重

---

## 6. 参考代码

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/models/qwen2_vl.py` | Qwen2-VL实现 |
| `python/sglang/srt/models/llava.py` | LLaVA实现 |
| `python/sglang/srt/models/internvl.py` | InternVL实现 |
