# DSV4-C128 池耗尽崩溃 —— 修复说明（评审用）

- **分支**：`fix/dsv4-c128-exhaustion`（基于 `ascend/release/2026930`，未提交）
- **代码基线**：`D:\coderepo\project\out_memory\sglang-ascend\sglang`（与报错日志中 `allocation.py:238 / :403` 行号完全一致的版本）
- **触发证据**：`logs/1.log.txt`（2026-10-16 05:35:32~49，跑了约 15 小时后整机崩）
- **改动规模**：6 个文件，+314 / −10
- **本地验证**：`python -m py_compile` 全通过；**无 NPU 环境，功能验证待上板**（见 §6）

---

## 1. TL;DR

日志给出的结论只有一句是确定的：

> 崩的是 **DSV4 C128 压缩 KV 页池**，不是 FULL/SWA 显存不足。而 C128 的可用量**从来没被打进任何日志**，所以故障链被完全掩盖。

围绕这个结论，代码里有 **4 个相互叠加的缺陷**，修法对应 7 处改动：

| 缺陷 | 性质 | 对应改动 |
|---|---|---|
| A. C128 容量不足时 `alloc_extend` 静默 `return None`，被上层翻译成 FULL/SWA 的 "out of memory" | 误诊（排查成本最高的那条） | 改动 2、3 |
| B. OOM 诊断函数对非 SWA/MAMBA 组件直接 `continue`，结构性排除 C128 | 可观测性缺失（"c128 没检查什么"的字面答案） | 改动 1 |
| C. C128 页的 release 有两条路径会被跳过 → 页 id 从所有表中消失、refcount 永久 ≥1 | 真·内存泄漏（与"跑 15 小时才炸"形态吻合） | 改动 4、5 |
| D. C128 池按 `full_token // 128` 分配，**双重 floor、零余量**；且 prefill 准入完全不计 C128 | 容量模型错误 + 缺少背压 | 改动 6、7 |

**改动 1~5 是"把真相暴露出来 + 堵住两条确证的滞留路径"，改动 6~7 是"让 C128 不再先于 FULL 枯竭、并在枯竭前优雅退让而不是带崩整机"。**

---

## 2. 为什么会定位到 C128（证据链）

### 2.1 日志里的关键三行

```
[DP12] KV cache pool is full. Retract requests. #retracted_reqs: 1, ...      (05:35:32)
[DP12] Prefill out of memory. Try to lower your batch size.                  (05:35:33)
Try to allocate 1224 tokens.
Available full tokens: 7296 (full_available_size=7296 + full_evictable_size_=0)
Available swa: 5120 (available_size=3968 + component_evictable_size_=1152)
```

注意第 3~5 行：**FULL 有 7296 token 可用、SWA 有 5120 可用，而本次只需要 1224**。这段文字是 `available_and_evictable_str()` 生成的，它**只报 FULL 和 SWA**。

### 2.2 抛出点与真正的返回点

```
scheduler.py:4097 _get_new_batch_prefill_raw → new_batch.prepare_for_extend()
schedule_batch.py:2763 prepare_for_extend → alloc_for_extend()
allocation.py:403      → alloc_paged_token_slots_extend(...)
allocation.py:238      → raise RuntimeError("Prefill out of memory...")   ← 背锅者
```

`allocation.py:229` 只在 `out_cache_loc is None` 时抛。DSV4 路径下 `out_cache_loc` 取自 `DSV4OutCacheLoc.out_full_loc`，而 bundle 是 `DSV4NPUTokenToKVPoolAllocator.alloc_extend()` 的返回值。该函数**只有两个返回 None 的出口**：

```python
# python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py  (改动前)
def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                 last_loc, extend_num_tokens, *, req_pool_indices=None, req_to_token_pool=None):
    self._cur_req_to_token_pool = req_to_token_pool
    if not self._has_c128_sidecar_capacity(prefix_lens_cpu, seq_lens_cpu):
        return None                                   # 出口 A：C128 页不足
    out_full_loc = super().alloc_extend(...)
    return self._wrap_full_alloc(...)                 # 出口 B：FULL/SWA 页不足（out_full_loc is None）
```

`_has_c128_sidecar_capacity` 的判据：

```python
def _has_c128_sidecar_capacity(self, prefix_lens_cpu, seq_lens_cpu) -> bool:
    need = self.c128_num_pages_needed(prefix_lens_cpu, seq_lens_cpu)
    page_size = self.c128_attn_allocator.page_size
    return need <= self.c128_attn_allocator.available_size() // page_size
```

**出口 B 被日志直接排除**：FULL 侧 `full_available_size=7296`，而这次只需要 1224 token 对应的约 10 个页；`super().alloc_extend` 的失败条件是
`num_new_pages > len(free_pages)`（`allocator_npu.py:54`），在 7296 个空闲 token 下不可能成立。所以：

> **唯一的 None 来源是出口 A —— C128 页池没有可分配的物理页。**

### 2.3 为什么 C128 "回收不动"

C128 的设备页**没有自己的驱逐杠杆**：

- `C128SidecarComponent._evict_device_start / _evict_device_next_node / _evict_device_end` 全是空实现（`c128_sidecar_component.py:359-366`），C128 永远不会自己发起 eviction；
- 页只能作为 **FULL 叶被驱逐时的附带载荷**被回收（`unified_tree_core.py:_release_all_component_layers` → `FreeComponentDeviceSlot` → `release_c128_pages`）；
- 而日志的 radix 树 dump 正好说明**没有任何 FULL 叶可驱逐**：
  - 链 A 全是 `full=yes full_lock=4 / 1` —— `_is_device_leaf` 要求 `any(cd.lock_ref > 0) == False`（`unified_tree_core.py:2093`），被运行中的请求路径锁住 → 不是 D-leaf；
  - 链 B 全是 `full=no` —— 已驱逐，无 device 载荷；
  - 两者相加就是日志里的 `full_evictable_size_=0`。

于是 `ensure_c128_capacity()` 的循环第一轮拿到 `num_tokens_evicted == 0` 就 `break` 返回 `False` —— **而这个返回值被调用方丢掉了**（改动前的 `allocation.py:205`）。

### 2.4 一个必须澄清的诱导项

日志里紧邻的 `#new_token_ratio: 0.0294 -> 0.9401` **不是内存水位**，是 `NewTokenRatioTracker.estimate_new_token_ratio_after_retract()` 估的"还能生成多少 token"（按 `output_ids / max_new_tokens` 算，见 `new_token_ratio_tracker.py:41`）。不要拿它当容量证据。

### 2.5 DP 级联失败与本故障无关

DP11/13/10 的 `gloo Connection closed by peer`（都卡在 `dp_attn.py:474` 的 `mlp_sync_info.all_gather`）以及 `scheduler_0 crashed with exit code -3 (SIGQUIT)`，是 **DP12 抛异常后同组 rank 死在通信点上**的级联结果。查这些是浪费时间。

---

## 3. 缺陷清单（每个缺陷 = 现象 → 代码 → 机制 → 影响）

### 缺陷 A：C128 短路返回 None，被翻译成"FULL/SWA 显存不足"

- **代码**：`dsv4_allocator.py` 的 `alloc_extend` / `alloc_decode` / `alloc_extend_swa_tail`
- **机制**：三个方法都是 `if not self._has_c128_sidecar_capacity(...): return None`，无任何日志。调用方 `alloc_paged_token_slots_extend` 无法区分"None 来自 C128 还是 FULL"，于是统一报 `Prefill out of memory` + FULL/SWA 数字。
- **讽刺点**：作者**已经写好**一条精确文案 `_pool_exhausted()`（"DSV4 c{ratio} {kind} pool exhausted: need N …"），但它只在 `_alloc_c_extend`（真正做 c128 分配的地方）里 raise；而前置检查抢在前面返回 None，**这条信息永远不可达**。
- **影响**：排查成本（本次 15 小时 + 一次误判）；`_pool_exhausted` 形同死代码。

### 缺陷 B：OOM 诊断结构性排除 C128

```python
# unified_radix_cache.py.available_and_evictable_str()  (改动前)
for ct in self.tree_components:
    if ct == BASE_COMPONENT_TYPE: continue
    if ct.is_swa:      available_size = ...swa_available_size()
    elif ct.is_mamba:  available_size = ...mamba_allocator.available_size()
    else:
        continue          # ← C128 落到这里，永远不打印
```

- **影响**：这是"是不是 c128 没检查什么"的字面答案 —— 诊断输出里**结构性地不可能**出现 C128，所以看到 `full=7296 / swa=5120` 会误判为"明明有空间"。

### 缺陷 C：C128 页的两条滞留（泄漏）路径

C128 页用引用计数管理（`c128_page_refcount`），而 **req 的 sidecar 行是页 id 的唯一记录**，计数本身没有反向引用。因此"行被清空但没走 release"= 页 id 从所有表中消失、refcount 永久 ≥1、再也无法回收。

**C-1（`free()` 提前 return）**

```python
# dsv4_allocator.free()  (改动前)
kv_len = max(req.kv.kv_committed_len, req.kv.kv_allocated_len)
req_pool_idx = req.kv.req_pool_idx
if kv_len <= 0 or req_pool_idx is None:
    return                       # ← 直接返回：行没 release、也没 zero
row = req_to_token_pool.req_to_c128_sidecar[int(req_pool_idx)]
self.release_c128_pages(row[row > 0])
row.zero_()
```

**C-2（行复用时的 `zero_()`）**

```python
# dsv4_req_to_token_pool.alloc()  (改动前)
for is_fresh, req, req_pool_idx in zip(fresh, reqs, indices):
    if is_fresh:
        self.req_to_c128_sidecar[int(req_pool_idx)].zero_()   # ← 只清零，不 release
```

- **触发场景**：retract（`reset_for_retract` 前把 KV 释放/长度归零）、abort、以及 C-1 留下的"脏行"被下一个 fresh 请求复用。日志里崩溃**前一步正好发生了一次 retract**（`#retracted_reqs: 1`），与这两条路径的形态吻合。
- **影响**：页随长时间运行单调泄漏 → 正是"跑 15 小时才炸"的形态。**注意：这两条是代码层面确证的滞留，但是否构成本次崩溃的直接原因，需要改动 1 的诊断数据来最终判定（见 §5）。**

### 缺陷 D：容量模型 + 准入

**D-1 池大小双重 floor，零余量**

```python
# pool_configurator._compute_dsv4_sizes()  (改动前)
c128_max_total_num_tokens = full_token // 128
```

再用它构造页级分配器（`dsv4_allocator.mk()` → `NPUPagedTokenToKVPoolAllocator`，`num_pages = size // page_size`），于是

```
池页数 = floor(floor(full_token / 128) / ps)
```

而"表示满池 FULL 所需页数"是 `ceil(full_token / (128 * ps))`。**两者天然可能差 1 页**（例如 full_token=65641、`c128_page_size=16` → 需要 33 页，池只有 32 页）。再叠加**每条活跃链独占一个尾页**（`c128_page_size` 是页级单位，链尾不满页也占整页），C128 就会在 FULL 还有余量时先枯竭。

**D-2 prefill 准入完全不计 C128**

- decode 侧是完整的：`check_decode_capacity()`（`dsv4_allocator.py:307-337`）显式检查 C128，`check_decode_mem` 的注释更写明"allocator 拥有容量闸门，让 allocator 侧不足**优雅 retract** 而不是触碰 fail-loud 报错"，日志第 3 行的 retract 就是它在工作；
- prefill 侧相反：`SWAPrefillBudget._available_and_evictable()` 只算 `full_available_size() + full_evictable_size()`，`prefill_budget.py` 全文 **0 处** `c128`。
- **影响**：这个不对称 = "准入 eager 放行 → allocator 硬崩 → DP 通信点级联 → 整机退出"。这是本次从"请求失败"升级为"引擎崩溃"的关键。

---

## 4. 改动逐条讲解

### 改动 1｜诊断补上 C128（缺陷 B）

**文件**：`python/sglang/srt/mem_cache/unified_radix_cache.py`（`available_and_evictable_str` + 新增 `_c128_capacity_line`）

```python
             elif ct.is_mamba:
                 available_size = self.req_to_token_pool.mamba_allocator.available_size()
             else:
+                # Auxiliary pools that are not full-token addressed (the DSV4
+                # C128 compressed-KV pool) own a page-granular allocator, so
+                # the token arithmetic above does not apply to them. ...
+                c128_line = self._c128_capacity_line(ct)
+                if c128_line is not None:
+                    lines.append(c128_line)
                 continue
```

新增方法按**页**输出（C128 的单位和 FULL token 不可比，所以必须换单位）：

```python
def _c128_capacity_line(self, component_type: ComponentType) -> Optional[str]:
    if component_type != ComponentType.C128:
        return None
    c128_allocator = getattr(self.token_to_kv_pool_allocator, "c128_attn_allocator", None)
    if c128_allocator is None:
        return None
    page_size = c128_allocator.page_size
    free_pages = c128_allocator.available_size() // page_size
    evictable_pages = self.tree_core.component_evictable_size(component_type) // page_size
    return (f"Available {component_type} pages: {free_pages + evictable_pages} "
            f"(free_pages={free_pages} + component_evictable_pages_={evictable_pages}) "
            f"of {c128_allocator.num_pages} ({page_size} compressed tokens/page, "
            f"1 compressed token per 128 full tokens)")
```

修复后，同样的故障会多出这样一行（这就是下次一条日志定性的依据）：

```
Available C128 pages: 0 (free_pages=0 + component_evictable_pages_=0) of 32 (16 compressed tokens/page, 1 compressed token per 128 full tokens)
```

**为什么用 `getattr` + 类型判断**：`unified_radix_cache` 是通用缓存，不能硬依赖 NPU 的 DSV4 分配器；非 DSV4/CUDA 路径返回 `None`，行为不变。

---

### 改动 2｜DSV4 分配器：不再静默 + 提供精确文案

**文件**：`python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py`

新增（放在 `_pool_exhausted` 之后）：

```python
logger = logging.getLogger(__name__)          # 模块头
C128_RATIO = 128                              # 模块头常量

def c128_pages_available(self) -> int:
    allocator = self.c128_attn_allocator
    return allocator.available_size() // allocator.page_size

def c128_pages_total(self) -> int:
    return self.c128_attn_allocator.num_pages

def c128_pool_exhausted_msg(self, need_pages: int, phase: str) -> str:
    """精确的 C128 不足文案，交给调用方拼进既有的 OOM 报告（含树 dump）。"""
    page_size = self.c128_attn_allocator.page_size
    return (f"DSV4 C128 compressed-KV pool exhausted during {phase}: need "
            f"{need_pages} free page(s) of {page_size} compressed tokens, have "
            f"{self.c128_pages_available()}/{self.c128_pages_total()} pages. "
            f"C128 is a separate page-granular pool (1 compressed token per "
            f"{C128_RATIO} full tokens) whose only reclaim path is FULL-leaf "
            f"eviction, so it can run dry while FULL still has free tokens.")

def _c128_sidecar_short(self, prefix_lens_cpu, seq_lens_cpu, where: str) -> bool:
    if self._has_c128_sidecar_capacity(prefix_lens_cpu, seq_lens_cpu):
        return False
    logger.warning("DSV4 c128 KV pool short in %s: need %d page(s), available %d/%d ...",
                   where, self.c128_num_pages_needed(prefix_lens_cpu, seq_lens_cpu),
                   self.c128_pages_available(), self.c128_pages_total())
    return True
```

三个 `alloc_*` 的短路点改成等价但不再静默：

```python
-        if not self._has_c128_sidecar_capacity(prefix_lens_cpu, seq_lens_cpu):
+        if self._c128_sidecar_short(prefix_lens_cpu, seq_lens_cpu, "alloc_extend"):
             return None
```

**为什么这里仍然保留 `return None` 而不直接 raise**：`alloc_extend` 有一条调用方（disagg decode 的 prealloc 路径 `dsv4_unwrap_prealloc`）会容忍 `None`，直接 raise 会改掉那条路径的语义。真正的"精确报错"放在改动 3 的调用侧，那里能同时打印 radix 树 dump。

---

### 改动 3｜`allocation.py`：把被丢弃的回收结果接住，报真因

**文件**：`python/sglang/srt/mem_cache/allocation.py`（prefill 与 decode 两个 DSV4 站点）

```python
     if is_dsv4:
         c128_num_pages = allocator.c128_num_pages_needed(prefix_lens_cpu, seq_lens_cpu)
-        allocator.ensure_c128_capacity(tree_cache, c128_num_pages)
+        if not allocator.ensure_c128_capacity(tree_cache, c128_num_pages):
+            error_msg = (
+                f"Prefill out of memory. "
+                f"{allocator.c128_pool_exhausted_msg(c128_num_pages, 'prefill')}\n"
+                f"Try to allocate {extend_num_tokens} tokens.\n"
+                f"{available_and_evictable_str(tree_cache)}"
+            )
+            logger.error(error_msg)
+            if tree_cache is not None:
+                tree_cache.pretty_print()
+            raise RuntimeError(error_msg)
```

decode 站点同构（`phase='decode'`，token 数用 `len(seq_lens) * token_per_req`）。

**关键点**：
- 保留 `"Prefill out of memory"` 前缀，不破坏任何基于该字符串的 grep/告警；
- **不会把原本能成功的分配变成失败**（方向性论证，这是本改动唯一的安全性要求）：
  - 前置门槛（`_has_c128_sidecar_capacity` 与 `ensure_c128_capacity`）用的是 `available_size() // page_size`，而 `NPUPagedTokenToKVPoolAllocator.available_size() = (len(free_pages) + num_staged_pages) * page_size`；
  - 真正做分配的 `_alloc_c_extend` 要求 `num_new_pages <= len(free_pages)`；
  - 由于 `available_size()//ps >= len(free_pages)`，**前置门槛永远不会比实际分配更严格** → 新增的提前 raise 不可能误杀一个本来会成功的分配。
  - 页数公式也一致：`c128_num_pages_needed()` 的 `ceil((lens//128)/ps)` 与页级分配器的 `(c_seq + ps - 1)//ps - (c_prefix + ps - 1)//ps`（`c_seq = seq//128`）代数相同。
- 已知边界 1：`ensure_c128_capacity` 在 `tree_cache is None or is_chunk_cache()` 时**不做驱逐**直接返回 False；这条路径下改动前也是失败（只是文案不同），所以无回归。
- 已知边界 2：若 `num_staged_pages > 0`，前置门槛可能通过、而 `_alloc_c_extend` 的内部检查失败 —— 此时抛的是同族精确文案 `_pool_exhausted`，同样无回归。

---

### 改动 4｜`free()`：行必须无条件 release + zero（缺陷 C-1）

```python
     if req is None or req_to_token_pool is None:
         return
-    kv_len = max(req.kv.kv_committed_len, req.kv.kv_allocated_len)
     req_pool_idx = req.kv.req_pool_idx
-    if kv_len <= 0 or req_pool_idx is None:
+    if req_pool_idx is None:
         return
+
+    # Release + clear the sidecar row unconditionally. ...
     row = req_to_token_pool.req_to_c128_sidecar[int(req_pool_idx)]
     self.release_c128_pages(row[row > 0])
     row.zero_()
+
+    kv_len = max(req.kv.kv_committed_len, req.kv.kv_allocated_len)
+    if kv_len <= 0:
+        return
     self.get_kvcache().clear_request_scoped_state(int(req_pool_idx))
```

**只改了 release/zero 的门槛**；`clear_request_scoped_state` 仍然保持原来的 `kv_len > 0` 门槛（最小改动，避免影响 request-scoped state 的语义）。

---

### 改动 5｜行复用前先 release（缺陷 C-2）

**文件**：`python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py`

```python
     for is_fresh, req, req_pool_idx in zip(fresh, reqs, indices):
         if is_fresh:
-            self.req_to_c128_sidecar[int(req_pool_idx)].zero_()
+            self._release_stale_c128_row(int(req_pool_idx))

+def _release_stale_c128_row(self, req_pool_idx: int) -> None:
+    row = self.req_to_c128_sidecar[req_pool_idx]
+    if self._dsv4_allocator is not None:
+        self._dsv4_allocator.release_c128_pages(row)
+    row.zero_()
```

`release_c128_pages` 内部自己做 `unique` + 过滤 `> 0`，所以"整行传进去"是安全的，且**多做一次 `unique` 的开销只在每个新请求分配时发生一次**（分配路径本来就有同步点）。

---

### 改动 6｜C128 池按整页 sizing，并保持内存账本诚实（缺陷 D-1）

**文件**：`python/sglang/srt/model_executor/pool_configurator.py`

```python
+    _c128_tail_pages: int = 1          # 类属性（object.__new__ 的测试 stub 也能用）

+    def _c128_page_size(self) -> int:
+        return get_schedule().c128_page_size

+    def _c128_pool_pages(self, full_token: int) -> int:
+        group_tokens = 128 * self._c128_page_size()
+        return ceil_div(full_token, group_tokens) + self._c128_tail_pages

+    def _c128_tail_headroom_bytes(self) -> int:
+        if self.num_layers_ca128 == 0: return 0
+        bytes_per_compressed_token = self.kv_bytes * self.num_layers_ca128
+        return self._c128_tail_pages * self._c128_page_size() * bytes_per_compressed_token
```

```python
-            c128_max_total_num_tokens=full_token // 128,
+            c128_max_total_num_tokens=(
+                self._c128_pool_pages(full_token) * self._c128_page_size()
+            ),
```

并在 `calculate_pool_sizes` 里把 headroom 记入 fixed bytes（**保证内存账本不失真**）：

```python
+        c128_tail_fixed_bytes = self._c128_tail_headroom_bytes()
         fixed_bytes = (
             c128_state_fixed_bytes + swa_fixed_bytes
-            + swa_ring_fixed_bytes + c4_state_fixed_bytes
+            + swa_ring_fixed_bytes + c4_state_fixed_bytes + c128_tail_fixed_bytes
         )
```

`_get_bytes_per_full_token()` 里 C128 的字节系数是 `1/128 * kv_bytes * num_layers_ca128`（每个 full token 摊到的 C128 成本），镜像它可得"每个压缩 token = `kv_bytes * num_layers_ca128` 字节"，两边口径一致。

启动日志新增可核对项：

```
DSV4 memory calculation: ... c128_tail_fixed=0.00 GB, c128_tail_pages=1, full_token=...
DSV4 pool sizes: full=..., swa=..., c4=..., c128=...      ← c128 应比改动前多 1 页(16 压缩 token)
```

**⚠️ 这里有一个我自己踩过并撤掉的坑（请重点复看）**：
第一版我写的是 `self._c128_tail_pages = max_running_requests_per_worker + 1`（每条活跃链一个尾页），逻辑上"更完整"，但
`_resolve_max_running_requests_per_worker()` 是**故意高估**的（注释原话："Over-estimating is safe: a larger fixed bias yields a smaller full_token"，返回值区间 2048~4096，而实际并发常常只有个位数~几十）。
用它算 headroom 会让 C128 池变成原来的 ~129 倍，直接把内存预算吃穿。**已回退为 `+1` 页**，每链尾页改由改动 7 在准入处按精确链长计价。这一段的设计理由也写进了代码注释，防止后人"好心改回去"。

---

### 改动 7｜C128 准入闸门 `DSV4PrefillBudget`（缺陷 D-2）

**文件**：`python/sglang/srt/mem_cache/prefill_budget.py`（新增类）+ `dsv4_allocator.create_prefill_budget()`（改用它）

```python
    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        # DSV4 additionally prices the C128 page pool at admission; see
        # DSV4PrefillBudget. ...
        from sglang.srt.mem_cache.prefill_budget import DSV4PrefillBudget
        return DSV4PrefillBudget(self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens)
```

新类只做三件事：按页计算链 footprint、检查剩余页、累计已准入的页预算。

```python
class DSV4PrefillBudget(SWAPrefillBudget):
    def c128_pages_needed(self, chain_tokens: int) -> int:
        group_tokens = 128 * allocator.page_size
        return -(-max(0, int(chain_tokens)) // group_tokens)      # ceil 到整页

    def c128_pages_free(self) -> int:
        return allocator.available_size() // allocator.page_size

    def _c128_fits(self, chain_tokens: int) -> bool:
        return self.c128_pages_needed(chain_tokens) <= self.c128_pages_free() - self.c128_offset

    # can_allocate_prefill(...)  → super() and self._c128_fits(paged_input)
    # check_prefill(...)         → ok 且 c128 不足时返回 (False, None)
    # reserve(...)               → 非 chunked 续投时 self.c128_offset += pages(extend_input_len)
```

**为什么这样不会活锁**：请求结束时 `free(req=...)` 会把它的页还给池（改动 4/5 之后是**无条件**归还），所以被推迟的请求一定会重新变得可准入。这与 SGLang 处理"FULL 装不下"的既有行为同构。

**为什么 C128 不足时不返回更小的 chunk**：页是分配粒度，缩小 chunk 最多只省一页，而过度切 chunk 会实质拖慢长 prompt。所以按"本轮不可准入"处理。

**为什么用子类而不改通用预算**：`create_prefill_budget` 是 allocator 的方法，DSV4-NPU 的 allocator（`DSV4NPUTokenToKVPoolAllocator`）直接持有它 → 零入侵。另外确认过：
- 代码中**没有任何** `isinstance/type` 检查预算对象（已 grep），子类安全；
- NPU 路径上 HiSparse 包装类不参与（`dsv4_memory_pool.py:298` 明确 "enable_hisparse is not supported on the NPU DSV4 KV pool"），所以这个 override 一定被走到。

---

## 5. 行为对比

| 场景 | 改动前 | 改动后 |
|---|---|---|
| C128 页用尽、FULL 仍有余量（本次现场） | `Prefill out of memory` + 只列 FULL/SWA 数字 → 误判为"明明有空间"；随后 DP 级联、整机 SIGQUIT | 准入阶段按页计价 → **推迟/切 chunk，引擎继续服务**；若仍触发分配失败，报 `DSV4 C128 compressed-KV pool exhausted ... need N page(s), have F/T` 并**附带 C128 页数行** |
| retract / abort / 行复用 | sidecar 行被清空但未 release → 页 id 永久消失、refcount 永久 ≥1 → 长跑单调泄漏 | `free()` 与行复用前**无条件** release + zero |
| 启动池划分 | `c128 = full_token // 128`（双重 floor，零余量） | `c128 = (ceil(full_token / (128·ps)) + 1) · ps`，headroom 字节计入 fixed budget |
| 诊断可观测性 | 结构上不可能看到 C128 | 每份 OOM 报告都带 `Available C128 pages: X (free_pages=.. + evictable..) of T` |

---

## 6. 上板验证清单

> 说明：本机是 Windows、无 NPU，我只能做 `py_compile` + 与日志逐行对照。以下必须在你的机器上跑。

1. **启动自检（不跑业务即可验）**
   - `DSV4 pool sizes: ... c128=` 比改动前**多 1 页**（`16` 压缩 token，`c128_page_size` 默认 16）；
   - `DSV4 memory calculation:` 里出现 `c128_tail_fixed=… c128_tail_pages=1`，且 `full_token` 相比改动前**略微下降**（headroom 进了账本，属预期，量级 ~1 页的字节）。

2. **故障形态对比（关键词搜 `c128`）**
   - 期望：出现 `DSV4 c128 KV pool short in alloc_extend/alloc_decode: need … available …`（warning，表示准入未拦住但被记录），或直接是精确的 `pool exhausted during prefill/decode`；
   - 任何 OOM 报告里都应带 `Available C128 pages:` 行。**这一行是判定"容量不足 vs 泄漏"的唯一依据**：
     - `free_pages` 长期为 0 且 `X/T` 稳定 → 容量模型问题（关注改动 6 的页数、以及 `--max-running-requests` 与长 prompt 的并发组合）；
     - `free_pages` 随时间单调下降 → 仍有泄漏（改动 4/5 已堵两条，若还出现就要按 §7 加不变量探针）。

3. **回归重点场景**（就是日志那个形态）
   - 2 条以上长链（单请求 5 万 token 级）并发 + 触发 retraction；
   - 期望表现：**排队/切 chunk**，不再整机退出；`scheduler_0 crashed with exit code -3` 不应再出现。

4. **内存不回归**
   - `npu-smi info` / HBM 占用：C128 headroom 只多 1 页，占用增量应在噪声量级；若 `full_token` 掉得明显超出预期，说明 `kv_bytes/num_layers_ca128` 的口径与 `_get_bytes_per_full_token` 不一致，回来找我调。

---

## 7. 未完成 / 待确认 / 已知局限

1. **"容量不足 vs 泄漏"尚未在数据上定死**。日志（含你给的这份）没有 C128 数字，这是缺陷 B 的直接后果。改动 1 补上了这个数字，但**需要在你的机器上再复现一次才能定性**。改动 4/5 修的是两条**代码可证**的滞留路径；改动 6/7 修的是两个**代码可证**的容量/准入缺陷。若上板后仍出现 `free_pages` 单调下降，下一步是加探针：
   ```python
   # 建议（尚未实现）：每个 DP rank 每 N step
   assert free_pages + retained <= num_pages
   logger.info("[c128] free=%d retained=%d total=%d", free_pages, retained, num_pages)
   ```
2. **改动 3 用的是"检查 = 提前报错"而非"检查 = 换策略重试"**。也就是说 C128 真枯竭时请求仍会失败（带确切原因），只是不再由误导性信息掩盖。要把它变成"自动降级重试"需要动调度策略，超出本次范围。
3. **`_c128_tail_pages` 固定 1 页**：这是刻意选择（见改动 6 的坑）。并发驱动的尾页由准入兜底；代价是"准入路径被绕过时仍可能耗尽"（例如未来的 disagg prealloc 新路径），此时至少报错是精确的。
4. **未跑单测/UT**：仓库里 DSV4/SWA 的预算与配置器有测试（代码提到 "SWA floor tests" 用 `object.__new__` stub 构造），建议在上板前先跑一遍相关的 mem_cache / pool_configurator 测试。

---

## 8. 附：如何自己复看

```bash
# 当前分支与改动清单
git -C <repo> status --short --branch
git -C <repo> diff --stat

# 逐文件看（推荐顺序：先看"为什么"再看"怎么改"）
git -C <repo> diff -- python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py
git -C <repo> diff -- python/sglang/srt/mem_cache/allocation.py
git -C <repo> diff -- python/sglang/srt/mem_cache/unified_radix_cache.py
git -C <repo> diff -- python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py
git -C <repo> diff -- python/sglang/srt/model_executor/pool_configurator.py
git -C <repo> diff -- python/sglang/srt/mem_cache/prefill_budget.py
```

**评审重点（我最希望被挑战的 3 处）**
1. 改动 3 的等价性论证：`ensure_c128_capacity` 的失败判据是否真的与 `alloc_extend` 内部的 `num_new_pages > len(free_pages)` 一致（若有反例，会出现"本可成功却报错"）。
2. 改动 6 的字节口径：`kv_bytes * num_layers_ca128` 与 `_get_bytes_per_full_token` 的 `1/128 * kv_bytes * num_layers_ca128` 是否严格互逆。
3. 改动 7 的准入语义：`check_prefill` 返回 `(False, None)` 在 `PrefillAdder` 里是否只导致"本轮不选它"，而不会永久踢出队列。

**回退**：改动全在工作区，`git stash` / `git checkout -- <file>` 即可；`release/2026930` 分支本身未被污染。
**移植到另一个 clone**：如在 `out_memory\sglang`（`fix/dsv4-a5-explicit-state`）上作业，可 `git diff > c128.patch` 后 `git apply`，需注意两处上下文（`allocation.py` 行号偏移、`pool_configurator` 的 `_c128_tail_pages` 类属性位置）。
