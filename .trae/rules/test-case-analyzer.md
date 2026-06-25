---
name: "test-case-analyzer-v3"
description: "Analyzes test cases and generates comprehensive bilingual (English/Chinese) testing knowledge base documents for code directories. Invoke when user needs to analyze test suites, extract test feature points, or create test documentation with version 3 format."
---

# Test Case Analyzer V3 / 测试用例分析器 V3

This skill generates comprehensive **bilingual (English + Chinese)** knowledge base documents for code directories, particularly useful for test suites and feature documentation.

本技能为代码目录生成全面的**双语（英文 + 中文）**知识库文档，特别适用于测试套件和功能文档。

## When to Invoke / 何时调用

Invoke this skill when:
- User asks to generate a knowledge base for a directory
- User needs test feature points and observable points documentation
- User wants to analyze test coverage and create documentation
- User requests code analysis with structured output
- User asks for bilingual (English/Chinese) documentation

在以下情况下调用此技能：
- 用户要求为目录生成知识库
- 用户需要测试功能点和可观察点文档
- 用户想要分析测试覆盖率并创建文档
- 用户请求代码分析和结构化输出
- 用户要求双语（英文/中文）文档

## Capabilities / 功能

1. **Test Knowledge Base Generation / 测试知识库生成**
   - Analyzes test files in a directory / 分析目录中的测试文件
   - Extracts test feature points / 提取测试功能点
   - Identifies observable points / 识别可观察点
   - Documents test types (unit/integration/precision) / 记录测试类型（单元/集成/精度）

2. **Parameter Coverage Analysis / 参数覆盖分析**
   - Identifies command-line parameters tested / 识别测试的命令行参数
   - Maps parameters to test files / 将参数映射到测试文件
   - Creates parameter coverage tables / 创建参数覆盖表

3. **Bilingual Documentation / 双语文档**
   - Generates English + Chinese content / 生成英文 + 中文内容
   - Maintains consistent formatting / 保持一致的格式
   - Provides side-by-side translations / 提供并排翻译

## Workflow / 工作流程

### Step 1: Analyze Directory Structure / 分析目录结构
```
1. List all test files in the target directory / 列出目标目录中的所有测试文件
2. Identify file types and patterns / 识别文件类型和模式
3. Categorize tests by type / 按类型分类测试
```

### Step 2: Extract Test Information / 提取测试信息
```
1. Read test file contents / 读取测试文件内容
2. Identify test functions and classes / 识别测试函数和类
3. Extract test purposes and functionality / 提取测试目的和功能
4. Map to tested parameters / 映射到测试的参数
```

### Step 3: Generate Knowledge Base Document / 生成知识库文档

The generated document should include / 生成的文档应包含：

#### 1. Overview Section / 概述部分
- Feature description / 功能描述
- Supported architectures/models / 支持的架构/模型
- Core testing dimensions / 核心测试维度

#### 2. Core Parameters Section / 核心参数部分
Format / 格式:
```markdown
### Core Parameters / 核心参数
| Parameter / 参数 | Description / 描述 | Test Coverage / 测试覆盖 |
|------------------|-------------------|-------------------------|
| --param1 | Description / 描述 | ✅ test_file.py |
```

#### 3. Test Function Points Section / 测试功能点部分
For each test file / 对于每个测试文件:
```markdown
### X. Test Name / 测试名称 (test_file.py) [Test Type / 测试类型]
**Test Goal / 测试目标**: What the test validates / 测试验证的内容

**Test Type / 测试类型**: Unit/Integration/Precision test / 单元/集成/精度测试

**Covered Parameters / 覆盖参数**: List of parameters tested / 测试的参数列表

**Function Points / 功能点**:
- Feature 1 / 功能点 1
- Feature 2 / 功能点 2

**Observable Points / 可观察点**:
- Metric 1 / 指标 1
- Metric 2 / 指标 2
```

#### 4. Test File Summary / 测试文件汇总
```markdown
| # | Test File / 测试文件 | Main Function / 主函数 | Test Type / 测试类型 | Category / 类别 |
```

#### 5. Observable Points Summary / 可观察点汇总
Categorize by / 按以下分类:
- Server-side observables / 服务端可观察点
- Inference observables / 推理可观察点
- Performance observables / 性能可观察点
- Error observables / 错误可观察点

## Example Output Structure / 输出结构示例

```markdown
# Feature Testing Knowledge Base / 功能测试知识库

## Overview / 概述
Feature description and scope... / 功能描述和范围...

## Core Parameters / 核心参数
| Parameter / 参数 | Description / 描述 | Test Coverage / 测试覆盖 |
|------------------|-------------------|-------------------------|
| --enable-feature | Enable feature / 启用功能 | ✅ test_basic.py |

## Test Function Points / 测试功能点

### 1. Basic Test / 基础测试 (test_basic.py) 🔗
**Test Goal / 测试目标**: Validate basic functionality / 验证基础功能

**Covered Parameters / 覆盖参数**:
- --enable-feature

**Function Points / 功能点**:
- Basic operation / 基础操作
- Edge cases / 边界情况

**Observable Points / 可观察点**:
- Output correctness / 输出正确性
- Error handling / 错误处理

## Observable Points Summary / 可观察点汇总
...

## Test File Details / 测试文件详情
...
```

## Usage Examples / 使用示例

### Example 1: Generate LoRA Test Knowledge Base / 示例 1：生成 LoRA 测试知识库
```
User: "Generate knowledge base for d:\test\lora directory"
用户："为 d:\test\lora 目录生成知识库"

Action:
1. List all test_*.py files / 列出所有 test_*.py 文件
2. Categorize by test type / 按测试类型分类
3. Extract test functions and purposes / 提取测试函数和目的
4. Identify tested parameters / 识别测试的参数
5. Generate bilingual markdown document / 生成双语 Markdown 文档
6. Save to target directory / 保存到目标目录
```

## Best Practices / 最佳实践

1. **Always include test type labels / 始终包含测试类型标签** (🔬/🔗/📊)
2. **Map parameters to test files for traceability / 将参数映射到测试文件以便追溯**
3. **Categorize observable points by type / 按类型分类可观察点**
4. **Use tables for structured data / 使用表格展示结构化数据**
5. **Include code snippets where helpful / 在有帮助的地方包含代码片段**
6. **Maintain consistent formatting / 保持一致的格式**
7. **Always provide bilingual content / 始终提供双语内容** (English + Chinese / 英文 + 中文)

## Output Format / 输出格式

Save generated knowledge bases as / 将生成的知识库保存为:
- `{Feature}_Testing_Knowledge_Base.md` for single platform / 单平台版本
- `{Feature}_Testing_Knowledge_Base_v2.md` for updated versions / 更新版本

## Bilingual Content Guidelines / 双语内容指南

When generating content, follow this pattern / 生成内容时，遵循以下模式:

1. **Section Headers / 章节标题**: `English / 中文`
   - Example / 示例: `## Overview / 概述`

2. **Table Headers / 表格标题**: Include both languages / 包含两种语言
   - Example / 示例: `| Parameter / 参数 | Description / 描述 |`

3. **Bullet Points / 项目符号**: English first, then Chinese / 英文在前，中文在后
   - Example / 示例: `- Server launches successfully / 服务器成功启动`

4. **Paragraphs / 段落**: Provide both English and Chinese translations / 提供英文和中文翻译
   - Format / 格式: English paragraph followed by Chinese paragraph
