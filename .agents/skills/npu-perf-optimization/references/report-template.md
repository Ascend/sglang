# Report Template (acceptance criterion #4)

`gen_report.py` renders this skeleton. Mandatory sections are marked
**[MUST]** — the acceptance criteria require: per-switch quantified
before/after comparison **[MUST-1]** and a directly reusable recommended
launch command with quantified improvement vs baseline **[MUST-2]**.

---

```markdown
# SGLang Ascend NPU Performance Optimization Report

- Date: <date>
- Hardware: <hardware> ×<cards> | Model: <model> (<quant>) | sglang: <commit>
- Workload: `<verbatim frozen spec JSON>`            [MUST: reproducibility]
- Noise floor: ±3% (deltas within this are noise)

## Baseline                                                [MUST context]
- TPOT p50: **X ms** | TTFT p50: X ms | Throughput: X tok/s
- Bottleneck attribution: **<comm|compute|launch|mixed>**
  (share: {matmul: X%, communication: X%, …})
- Hint applied: <one-line switch guidance>

## Feature-switch A/B (each vs baseline, identical workload)   [MUST-1]
| Variant | Status | TPOT Δ% | Throughput Δ% | Verdict |
|---|---|---|---|---|
| npugraph_bs64   | ok | -8.4 | +9.1 | TPOT 8.4% better; throughput 9.1% better |
| deepep_auto     | ok | -1.2 | +1.9 | no significant change (within noise) |
| dpattn_dp2      | failed | - | - | (log tail linked in report.json) |
| mtp_eagle3_s3d4 | ok | -25.0 | +30.0 | … |

## Recommended launch command                              [MUST-2]
> ⚠️ only if compat check passed; otherwise the warning block replaces use
```bash
export <env…>
python3 -m sglang.launch_server \
    <full merged args incl. recommended switch settings>
```
Applied overrides: `<overrides JSON>`
Expected vs baseline: TPOT −X% / throughput +X% (from A/B artifacts).

## Provenance
- baseline: `<path>` · ab: `<path>` · bottleneck: `<path>`
- (all numbers trace to stored artifacts; no unsourced numbers)
```

---

## Writing rules (enforced by gen_report.py / quality gates)

1. Deltas ≤ noise floor are worded **"no significant change"** — never as
   improvement or regression.
2. Failed/rejected variants stay visible in the table (status column) —
   negative results are results.
3. The recommended command is the **full copy-pasteable command** (env exports
   included), not a diff against the baseline command.
4. `compat_errors` non-empty ⇒ the warning block renders above the command and
   the report is not usable as-is.
5. Every number in prose must appear in `report.json` (single source).
