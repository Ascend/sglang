# Memory Strategy

NPU 上 SGLang 与 HF 同时加载会 OOM(尤其 dsv4 等大模型)。**串行加载策略**保证不抢显存。

## 串行加载顺序(Phase 2 → Phase 3)

```
Phase 2 起点:SGLang server 已经在跑(P1 用它复现)
  ├ 先 dump SGLang 输出到 sgl_layer_outputs/(可选,可在 P3 再做)
  ├ 关闭 SGLang server,等待 30s 让显存释放
  ├ 验证显存释放:npu-smi info | grep <PID> 确认 PID 已消失
  ├
  ├ 启动独立 python:加载 HF 模型(eager mode,同 dtype)
  ├ 跑 dump_hf_layer_outputs.py
  ├ 保存 hf_layer_outputs/
  ├ 关 HF python(显式 del model + torch.npu.empty_cache + sys.exit)
  ├
Phase 3:
  ├ 启动独立 python:加载 SGLang 模型(同 server 代码路径,但不起 server)
  ├ 跑 dump_sgl_layer_outputs.py(若 Phase 2 起点没做)
  ├ 保存 sgl_layer_outputs/
  ├ 此 python 进程可继续用于 Phase 4 实验,或关掉重起
```

## NPU 资源探测(沿用 npu-adapter feedback memory)

启动任何 python(HF 或 SGLang 加载)之前:

```bash
# 数空闲 die
busy_dies=$(npu-smi info | tail -n +<some_offset> | awk '$NF != "0" { count++ } END { print count }')
free_dies=$((8 - busy_dies))

if [ "$free_dies" -lt "$required_dies" ]; then
    # 指数回避:60 → 120 → 240 → 480 → 960 → 1800s 封顶
    sleep $backoff
    # 重试
fi
```

**强制约定**:不 kill 别人的进程。只 kill 自己 PID 文件里的进程。

## 显式释放显存

每次关进程后都做:

```python
import torch
import torch_npu
del model
import gc; gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
```

进程退出 30s 后用 `npu-smi info` 二次确认显存归零再启动下一个进程。
