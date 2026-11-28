# Precision RCA + Fix — Methodology & Design Rationale

本文件解释 `precision-rca` 子 agent 的**设计依据** —— 为什么这么做。
具体的执行流程(Phase 1/2/3 + P4 native 替换 + P5 修复报告 / 工具调用)在 `../../prompts/precision_rca.md` 中;主流程触发条件见 npu-adapter 的 `SKILL.md` Step 6.5。

> 历史说明:本文件源于早期独立的 sglang-precision-fix skill。合并到 sglang-npu-adapter 后,precision-rca 子 agent 同时具备"定位 + native 替换修复 + drift 归零验证"能力。文中部分名词(如 status 枚举、产物 schema)请以 prompts/precision_rca.md 为准 —— prompt 是入口,本文件是方法学背景。

---

## Reference 选择:为什么是 HF NPU eager

金标准 = **HF transformers eager mode + 同一 dtype + 同一台 NPU**。

理由:
- **可行性**:用户场景以小模型为主(单机能装),HF 在 NPU 上 eager 加载在内存上可行
- **运维简化**:同机部署消除 GPU/HF 机器的运维成本
- **工程性**:串行加载策略(P2 关 SGLang → 加载 HF → dump → 关 HF → 加载 SGLang),不需要分 die 同时跑

### 逻辑边界(承认而非掩盖)

> HF NPU eager 与 SGLang 共享底层 `aten::*` 原语。如果某个原语自身在 NPU 上有 bug,两边会**同样**错误,agent 报告"无漂移"。这种情况不在本 agent 的检测范围内 —— **设计如此,不是缺陷**。

真要查 NPU runtime 正确性,需要 GPU/CPU 端的 reference,属于另一档问题(out-of-scope)。

---

## 7 大 Category(实验序列分类标准)

`hypothesis_library.md` 按这 7 类组织实验配方;`root_cause.json.candidates[].category` 必须是这 7 个之一:

1. **dtype** —— 强制 fp32 cast 类。覆盖 dtype 截断、layer_norm eps、residual 累加顺序等通用数值稳定性问题
2. **fused_kernel** —— 关 fused 走 fallback。覆盖 attention / MoE 的 fused 实现数值差(NPU flash-attn 等)
3. **quantization** —— force quant=None。覆盖 W8A8 scale dtype/shape 错
4. **weight_loading** —— bit-by-bit weight diff。覆盖 shard split / layout / scale 维度等加载错
5. **communication** —— tp_size=1 / 关 ep。覆盖 allreduce / all2all 的 dtype 与排序差
6. **graph_capture** —— `--disable-cuda-graph`。覆盖 ACLGraph 捕获引入的偏差
7. **rope_implementation** —— RoPE theta / sin-cos table dtype / neox-vs-gptj 路径

**operator-side issue 不是独立 category** —— 它通过 entry 的 `source: "operator"` 字段标识,内含的具体类别仍取自这 7 类(如 `category=fused_kernel + source=operator` 表示"fused kernel 的算子侧 bug")。

---

## Operator escalation:何时升级给厂商

### 判定条件(同时满足)

1. 8 个 sglang-side 实验全部 `inconclusive` 或 `rejects`(找不到任何 sglang 改动能压住 drift)
2. `first_bad_module` 是 fused / quant / communication 类
3. `env_fingerprint`(chip + CANN + torch_npu version)命中 `npu_numerical_behavior.md` 已知算子侧问题清单中的某条

### 触发后

`candidates` 增加一条 `source="operator"` 项,附:
- env_fingerprint
- workaround(sglang 侧绕开方法)
- escalate_to(目标团队)
- 复现步骤

`status` = `located_inconclusive_operator_side`。

### 为什么 agent 不试图修

- agent 不能改 `torch_npu` / CANN 内部代码 —— 这些代码不在 sglang 仓
- 算子侧 bug 通常通过 wheel / driver 升级修,不是 patch
- 即使能找到 workaround,选择"修真根因"还是"workaround"是人的决策,不是 agent 的

---

## 不设时间预算:正确性优先

precision RCA **不设墙钟时间限制**。

### 理由

定位精度 bug 是确定性问题:实验序列跑完要么找到根因,要么没找到。中间被时间预算切断会:
- 漏掉本来能找到的根因(实验在 timeout 时还没跑到)
- 强行报 `max_budget_reached` 这种 status,把"没跑完"当成"无法定位"

正确性优先于墙钟时间。

### 实际的停止条件(语义,不是时间)

- `cannot_reproduce` —— P1 复现不出来 → 缺前提,STOP
- `verdict=supports` —— 找到强候选 → 找到了,STOP
- 实验列表跑完(默认 ≤ 8 个) —— 列表内没找到 → STOP,出 `located_inconclusive`

### NPU 资源占用注意

无时间限制 ≠ 无资源责任。共享 NPU 上仍要遵守 `memory_strategy.md` 的串行加载 + 显存释放规则。RCA 跑得久 ≠ 抢卡:每次实验之间释放显存,允许其他人在 die 间隙使用。
