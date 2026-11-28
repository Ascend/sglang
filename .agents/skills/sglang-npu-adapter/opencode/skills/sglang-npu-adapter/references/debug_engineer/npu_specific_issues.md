# NPU 专属错误

## ACL 错误码

`E39999` 通常是显存不足或 stream sync 失败。`EE1001` / `EE9999` 类是 NPU runtime 报的,通常意味着算子调用失败,需要看完整 traceback 才能定性。

格式:`RuntimeError: ACL stream synchronize failed, error code: E39999`

## 算子级失败模式

### 1. 算子不支持(operator not found)

```
RuntimeError: aclnnSomething is not supported
torch.ops.npu.something not found
```

**判定**:
- 看 traceback 指向哪个 op
- 看 SGLang 仓内是否已有该 op 的 NPU 分支(可能是 dispatch 没走对)
- 若仓内确认无对应实现 → `status=needs_human`,标 `operator_side_bug`,在 diagnosis 写"NPU 算子库缺 X,需算子团队补"

**典型例子**:
- flash_attn 的 `flash_attn_varlen_func` 在 NPU 上无对应,需替换为 torch-native 或 NPU 专用 PFA(Prompt Flash Attention)
- 某些 fused MoE kernel 无 NPU 实现,需 fallback 到拆分的 GEMM

### 2. 算子精度差异

```
现象:dummy 跑通,real-weight 跑通但输出全是同一个 token / 乱码 / NaN
```

这是**精度问题**,不在 debug_engineer 范围。

处理:在 fix_instructions 写 `status=needs_human` + diagnosis 标 `precision_suspect`,主流程会把 `adapter_state.precision_suspect` 置 true,Step 6.5 会触发 precision-rca。

### 3. Stream / device 错乱

```
RuntimeError: NPU stream is not synchronized
RuntimeError: tensor on cuda:0 but expected npu:0
```

**根因**:代码里 hardcode 了 `"cuda:0"` 或 `tensor.cuda()` 而非 `tensor.to(device)`。

**修复**:全文 Grep `\.cuda\(\)` 和 `"cuda` 字符串,改为 device-agnostic 写法。

## ACLGraph 相关

### 启动时报 graph capture 失败

```
ACL graph capture failed
某个 op 不支持 graph mode
```

**临时修复**:`--enable-aclgraph` 改成不开;在 fix_instructions 标"该模型暂不支持 ACLGraph,建议 feature_compatibility.aclgraph=unsupported"

### Graph mode 下 OOM

ACLGraph 会预分配显存,比 eager 用得多。降 `--mem-fraction-static`。

## 多卡 / 分布式

### NCCL 风格的初始化失败(HCCL 在 NPU 上的对应)

```
HCCL error: ...
torch.distributed initialization failed
```

**根因排序**:
1. 卡的拓扑问题(`HCCL_DETERMINISTIC` 之类环境变量)
2. 端口冲突(`--dist-init-port`)
3. 卡数 != world_size(`--tp` 大于实际可见卡数)

**检查命令**:
```bash
echo $ASCEND_RT_VISIBLE_DEVICES  # 应该列出实际可用 NPU
npu-smi info | grep "Ascend"     # 数应可见 NPU
```

### tp 拓扑不均

NPU 上某些拓扑(如跨 P2P 域)通信会慢很多。**这不是 bug**,可以正常跑,只是 throughput 受影响。在 fix_instructions 不必当 error 处理,可在 debug_report.md 标注。

## DeepEP 相关

```
DeepEP not built / missing
```

如果用户开了 `--enable-deepep` 但环境上 DeepEP 没编译,会报这个。

**修复**:
1. 关 flag(若不必须)
2. 或让用户先 build DeepEP(不在 skill 范围,`status=needs_human`)

## MTP 相关

```
MTP 状态不同步 / MTP token 偏移错
```

参考 commit `6fc07c09a` 的 `causal_conv1d_update_mtp_npu`——这是 PoC 分支已知的修复,确认相关代码已合入。

## 已知的非 bug 现象

- **首次启动慢**:torch_npu 首次加载 + 编译 cache 预热,30-60s 正常
- **ACLGraph 编译**:首次启动会卡几十秒,不是死锁
- **NPU 推理 throughput 比 GPU 低**:正常,不要当 bug
