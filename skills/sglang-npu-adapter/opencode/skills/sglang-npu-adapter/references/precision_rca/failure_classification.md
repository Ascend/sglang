# Failure Classification

P1 阶段在确认 drift 真实存在之后,必须把它归到下面 4 个 `failure_class` 中的**唯一一个**,并写回 `precision_context.json.failure_class`。

这一步是后续 dump / 二分策略的分流开关 —— **错分会让你浪费一整轮 P2/P3 在错误路径上**。

---

## 判别决策树

```
跑 failing_prompts 至少 2 次(同 prompt、同 seed、同 server)
  ├ 两次输出都对(无 drift)? → cannot_reproduce
  ├ 两次输出不一致(即使都"看起来错")? → random_undefined
  ├ 两次输出一致且错:
  │   ├ 第 1 个 token 就跟 HF 不一样? → prefill_first_token
  │   └ 第 1 个 token 一样,从第 N 个 token 起发散? → decode_after_first
```

判别用工具:`probe_model_arch.py` 输出的 dtype + 同样 dtype 跑一遍 HF NPU eager,对照 SGLang server 的输出 token id 序列。**不是看文字相似度,是看 token id 序列。**

---

## prefill_first_token

### 现象判据
- failing prompt 喂入后,SGLang 的**第 1 个 sampled token id ≠ HF NPU eager 的第 1 个 sampled token id**
- 多次重跑结果一致(确定性错误,排除 `random_undefined`)

### 典型怀疑点(按出现频次排序)
1. **embedding** —— word_embedding 或 token_type_embedding 路径(权重 layout / scale)
2. **position_ids / rope** —— sin-cos 表 dtype、rope_theta、neox vs gptj 模式、`position_ids` 起点 0 还是 1
3. **attention mask** —— causal mask 形状错、prefix-LM mask 在 SGLang 退化成全因果
4. **prefill 路径的 fused attention** —— flash-attn 在 NPU 上的 prefill 实现
5. **layer norm eps / weight loading** —— 第 1 层 LN 的 weight 没加载

### Dump 策略
- P2/P3 只 dump prefill 阶段(input → 每个 transformer block 出口 → lm_head)
- decode 路径不用 dump(prefill 错的话 decode 也错,但二分先聚焦根源)
- 多次重跑取 mean 没意义(确定性错误,跑 1 次就够)

### 推荐 hypothesis(连接 `hypothesis_library.md`)
按 first_bad_module 类型查:embedding → `weight_loading`;rope/position → `rope_implementation`;attention/mask → `fused_kernel`;LN/eps → `dtype`。

---

## decode_after_first

### 现象判据
- 第 1 个 token id 与 HF 一致
- 第 N(N ≥ 2)个 token 开始与 HF 序列发散
- 多次重跑发散点一致或非常接近(确定性错误)

### 典型怀疑点(按出现频次排序)
1. **KV cache 读写** —— KV layout 错、page table 越界、KV cache dtype 截断
2. **decode 路径的 fused attention** —— prefill 用一套实现、decode 用另一套(常见于 NPU)
3. **rope 在 decode 的累积** —— prefill 算对了 sin-cos,decode 时 position_id 没正确推进
4. **scheduler / batch 边界** —— batch_size=1 时对、>1 时错的 batched decoder 问题
5. **sampling 的数值差** —— logits dtype 截断、softmax 温度路径

### Dump 策略
- P2/P3 需要 dump **decode 阶段的逐 step 输出**(不只 prefill)
- 一般 dump 前 8 step 的每层输出就够;`failing_prompts` 不要选太长的(decode step 太多浪费)
- 若发散点固定在 step k,只 dump step k-1 / k / k+1 三个时刻的层级输出

### 推荐 hypothesis
按 first_bad_module 类型查:KV cache → `weight_loading` + `fused_kernel`;attention → `fused_kernel`;rope → `rope_implementation`;batched 边界 → `communication`(TP) 或 `fused_kernel`。

---

## random_undefined

### 现象判据
- 同 prompt、同 seed、同 server 多次重跑,**输出 token 序列不一致**
- 排除 sampling 随机性(把 `temperature=0` / `top_k=1` 后仍然不一致)

### 典型怀疑点(按出现频次排序)
1. **未初始化 buffer** —— 算子内部 workspace 没 memset,残留数据影响下一次
2. **越界访问** —— attention page table、KV cache 索引越界,读到隔壁请求的数据
3. **race condition** —— 多 stream 的同步缺失、reduction 顺序非确定
4. **HBM ECC / 硬件抖动** —— 罕见,先排除前 3 项再考虑

### Dump 策略
- P2 改成同 prompt **跑 2 次**,各自 dump 完整一遍(`dump_hf_layer_outputs.py` 默认就是 `x_run0` / `x_run1`)
- P3 同样 2 次,先看 `run0 vs run1` 自漂移,若 SGLang 内部 run0 vs run1 就有 drift,基本确认是 random;再跟 HF 比
- `find_first_bad_module.py` 的二分目标变成"找首个 run0 ≠ run1 的层"

### 推荐 hypothesis
此类问题在 sglang python 层很少能修;通常落到 NPU runtime / 算子仓 → 直接 `located_needs_human_fix`,出 op_dumps 上报算子团队(参考 `methodology.md` 的 operator escalation)。

---

## cannot_reproduce

### 现象判据
- failing_prompts 跑 ≥ 2 次都与 HF 一致,无 drift
- 用户提供的 `failing_evidence` 可能过期、配置不一致、或 prompt 选错

### 处置
- **不进入 P2**,直接出 `output/precision_fix_report.md`,status = `cannot_reproduce`
- 报告里列出实际跑出的输出 token 序列与 HF 的对比,要求用户补充:
  - 更多 failing prompt(尤其长度、模板差异较大的)
  - 当时复现的 SGLang 启动参数(dtype / quant / tp_size / 任何 flag)
  - 是否 server 状态特殊(例如长跑后才出错,初始状态正常)

---

## 反模式(不要这么分)

- **"看起来词不对"≠ `prefill_first_token`**:必须按 token id 序列判,不是字面相似度。BPE tokenizer 同 id 段可能渲染不同。
- **"温度高随机"≠ `random_undefined`**:`random_undefined` 的判别前提是已经把 sampling 锁成贪心(temperature=0 / top_k=1)
- **"只在某些 prompt 上错"不是 4 类之外的第 5 类**:仍按上面 4 类归;prompt 选择性属于"触发条件",写在 `failing_evidence.details` 里
- **prefill 错很可能 decode 也错,但归 `prefill_first_token`**:取最早出错点,不取累积错点

---

## 字段与产物

| 项 | 值 |
|---|---|
| 写到 | `precision_context.json.failure_class`(P1 自动探测填回) |
| 枚举 | `prefill_first_token` / `decode_after_first` / `random_undefined` / `cannot_reproduce` |
| 影响 | P2/P3 的 dump 范围(prefill-only vs decode-also)、P3 二分目标、P4 hypothesis 优先级 |
| 报告 | `precision_fix_report.md` 章节"复现 + 错误分类"必须明确写出来 |
