# 常见错误模式与处理

## 1. 环境 / 依赖错

### ImportError / ModuleNotFoundError

```
ModuleNotFoundError: No module named 'sglang.srt.models.xxx'
```

**根因排序**:
1. `PYTHONPATH` 未设 → 检查 `echo $PYTHONPATH | grep "$(pwd)/python"`,未命中则 `export PYTHONPATH=${PWD}/python:$PYTHONPATH`
2. 模型文件名拼写错 / 注册表未更新 → Grep `__init__.py` / `ModelRegistry`
3. 真的没装某依赖 → `pip list | grep <pkg>`;若 `transformers` 缺,**不要 pip install upgrade**,确认版本号后用 `pip install transformers==<pinned>` 装回

### Version mismatch

```
ImportError: cannot import name 'Cache' from 'transformers'
```

不要升级 transformers——`fix_instructions.status=needs_human`,在 diagnosis 写"transformers 版本与 SGLang 不兼容,等用户决定回退方案"。

### torch_npu 缺失

```
ModuleNotFoundError: No module named 'torch_npu'
```

`status=needs_human`,这是基础环境问题,不在 skill 修复范围。

## 2. AttributeError(GPU-only API 被调)

```
AttributeError: module 'torch' has no attribute 'cuda.empty_cache'  # 或类似
AttributeError: 'NoneType' object has no attribute 'is_initialized'
```

**典型场景**:模型文件里直接用了 `torch.cuda.X`,在 NPU 上 attribute 不存在。

**修复**:
```python
# Bad
torch.cuda.empty_cache()

# Good (option 1: device-agnostic)
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif hasattr(torch, "npu") and torch.npu.is_available():
    torch.npu.empty_cache()

# Good (option 2: SGLang 内部已封装)
from sglang.srt.utils import is_npu
if is_npu():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()
```

或直接用 torch-native 等价:`torch.empty_like(x)` 替 `torch.cuda.empty_like(x)`(后者通常不存在;前者跨 device 通用)。

## 3. shape / dtype mismatch

```
RuntimeError: Expected size [4096, 4096] but got [4096, 11008]
RuntimeError: Expected scalar type BFloat16 but found Float
```

**根因排序**:
1. **Weight loading 顺序错** → Grep 模型文件的 `load_weights`,看 q/k/v 是 split 还是 merged,与 HF checkpoint 的存储格式是否匹配
2. **tp split 配置错** → 看 `num_heads % tp == 0`、`num_kv_heads % tp == 0` 是否成立
3. **dtype 默认值不一致** → SGLang 启动时 `--dtype bfloat16` 必填,若用户没显式指定可能默认 fp16

## 4. OOM

```
RuntimeError: NPU out of memory
ACL Error: E39999
```

**降一档**(按顺序尝试):
1. `--mem-fraction-static 0.85` → 0.8 → 0.75
2. `--max-running-requests 256` → 128 → 64
3. `--max-prefill-tokens 8192` → 4096
4. `--context-length 131072` → 32768
5. 降 tp(但通常 tp 越小越 OOM)
6. 实在不行 → `status=needs_human`,可能是模型本身需要更多卡

不要乱开 chunked prefill 当 workaround——可能引入正交问题。

## 5. 启动卡死(健康检查超时)

```
curl -s localhost:8000/v1/models 长时间无响应
```

**根因排序**:
1. **Weight loading 慢** → 大模型首次加载几分钟正常;看 `dummy_run.log` 是否有 "Loaded X% weights" 进度
2. **ACLGraph 编译** → `--enable-aclgraph` 首次启动会卡 30-60s
3. **端口冲突** → `ss -tnlp | grep 8000` 看是否被其他 server 占
4. **server crash 但 stderr 没写到 log** → `kill -0 $(cat server.pid)` 看 PID 还在不在

5 分钟仍无响应 + log 无进展 → 多半是真卡死了,`kill` PID,看是否 stderr 有遗漏的 traceback。
