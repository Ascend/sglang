#!/usr/bin/env python3
"""
参考实现差异审计（通用版）。

只对比目标模型文件与同族参考实现的结构差异：
  - 类 / 方法（reference 有、target 没有）
  - import（reference 有、target 没有）
  - 设备/后端相关分支（if 条件里含 npu/cuda/device/backend/platform 等通用词）

不预设任何具体方法名、算子名或模型名；由 agent 根据差异自己判断哪些需要移植。

用法：
  python audit_model_parity.py \
      --model-file python/sglang/srt/models/qwen3.py \
      --models-dir python/sglang/srt/models \
      --output /path/to/output/reference_audit.json

退出码：
  0 - 审计完成
  1 - 参数/文件错误，或 --fail-on-delta 且 verdict=delta
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# 通用“设备/后端”词，不包含任何具体模型/算子名
DEVICE_TERMS = re.compile(
    r"(npu|cuda|xpu|hip|rocm|device|backend|platform|accelerator|ascend)",
    re.IGNORECASE,
)

CLASS_RE = re.compile(r"^class\s+(\w+)", re.MULTILINE)
CAMEL_RE = re.compile(r"[A-Z][a-z0-9]*")

MAX_ITEMS = 30


def is_device_text(text):
    return bool(DEVICE_TERMS.search(text or ""))


def class_suffix(name):
    """Attention / DecoderLayer / Model / ForCausalLM 等结构后缀，用于跨族匹配。"""
    for suffix in (
        "ForCausalLM",
        "ForConditionalGeneration",
        "ForSequenceClassification",
        "DecoderLayer",
        "Attention",
        "MLP",
        "Model",
    ):
        if name.endswith(suffix):
            return suffix
    return name[:1].lower() + name[1:] if name else name


def parse_file(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)

    classes = {}
    top_funcs = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
            classes[node.name] = methods
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_funcs.add(node.name)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.add(f"{mod}.{alias.name}" if mod else alias.name)

    device_branches = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            try:
                test = ast.unparse(node.test)
            except Exception:  # noqa: BLE001
                continue
            if is_device_text(test):
                device_branches.add(test)

    return {
        "file": str(Path(path).resolve()),
        "classes": classes,
        "top_funcs": top_funcs,
        "imports": imports,
        "device_imports": {i for i in imports if is_device_text(i)},
        "device_branches": device_branches,
    }


def family_of(parsed, fallback_stem):
    counts = {}
    for cls in parsed["classes"]:
        parts = CAMEL_RE.findall(cls)
        if parts:
            head = parts[0]
            counts[head] = counts.get(head, 0) + 1
    if not counts:
        return fallback_stem.split("_")[0]
    return max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def stem_of(path):
    name = Path(path).name
    return name[:-3] if name.endswith(".py") else name


def method_ids(parsed):
    ids = set()
    for methods in parsed["classes"].values():
        ids.update(methods)
    ids.update(parsed["top_funcs"])
    return ids


def device_methods(parsed):
    return {m for m in method_ids(parsed) if is_device_text(m)}


def class_ids(parsed):
    return set(parsed["classes"])


def compare(target, reference):
    t_methods = method_ids(target)
    r_methods = method_ids(reference)
    t_classes = class_ids(target)
    r_classes = class_ids(reference)

    ref_only_methods = sorted(r_methods - t_methods)
    ref_only_classes = sorted(r_classes - t_classes)
    ref_only_imports = sorted(reference["device_imports"] - target["device_imports"])
    ref_only_branches = sorted(reference["device_branches"] - target["device_branches"])

    device_ref_only_methods = [m for m in ref_only_methods if is_device_text(m)]
    device_ref_only_classes = [c for c in ref_only_classes if is_device_text(c)]

    return {
        "reference": reference["file"],
        "device_methods_only_in_reference": device_ref_only_methods[:MAX_ITEMS],
        "device_imports_only_in_reference": ref_only_imports[:MAX_ITEMS],
        "device_branches_only_in_reference": ref_only_branches[:MAX_ITEMS],
        "device_classes_only_in_reference": device_ref_only_classes[:MAX_ITEMS],
        "generic_methods_only_in_reference": [m for m in ref_only_methods if not is_device_text(m)][:MAX_ITEMS],
        "generic_classes_only_in_reference": [c for c in ref_only_classes if not is_device_text(c)][:MAX_ITEMS],
        "methods_only_in_target": sorted(t_methods - r_methods)[:MAX_ITEMS],
        "classes_only_in_target": sorted(t_classes - r_classes)[:MAX_ITEMS],
    }


def audit(model_file, models_dir, explicit_refs):
    model_file = Path(model_file).resolve()
    models_dir = Path(models_dir).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"model file not found: {model_file}")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"models dir not found: {models_dir}")

    target = parse_file(model_file)
    target_stem = stem_of(model_file)
    target_family = family_of(target, target_stem)

    candidates = []
    for p in sorted(models_dir.glob("*.py")):
        if p.resolve() == model_file:
            continue
        parsed = parse_file(p)
        candidates.append(parsed)

    explicit = set()
    for ref in explicit_refs:
        rp = Path(ref).resolve()
        if rp.is_file() and rp != model_file:
            parsed = parse_file(rp)
            parsed["explicit"] = True
            candidates.append(parsed)
            explicit.add(str(rp))

    siblings = []
    for c in candidates:
        c_family = family_of(c, stem_of(c["file"]))
        same_family = c_family == target_family
        same_prefix = stem_of(c["file"]).split("_")[0] == target_stem.split("_")[0]
        if not (same_family or same_prefix or c.get("explicit")):
            continue
        # 结构重合度：目标类后缀在 reference 中出现得越多越相关
        suffix_hits = 0
        target_suffixes = {class_suffix(cn) for cn in target["classes"]}
        for cn in c["classes"]:
            if class_suffix(cn) in target_suffixes:
                suffix_hits += 1
        c["_score"] = (100 if same_family else 0) + (50 if same_prefix else 0) + suffix_hits
        siblings.append(c)

    siblings.sort(key=lambda c: (c["_score"], len(c["device_imports"])), reverse=True)
    if not siblings:
        return {
            "target": {"file": str(model_file), "family": target_family},
            "references": [],
            "delta": {},
            "verdict": "no_reference",
            "recommendation": "没有找到同族/同名前缀的参考实现；请用 --reference 显式指定。",
        }

    comparisons = [compare(target, ref) for ref in siblings[:3]]
    # 汇总：优先看 device 相关差异
    device_methods = []
    device_imports = []
    device_branches = []
    device_classes = []
    for comp in comparisons:
        for key, bucket in [
            ("device_methods_only_in_reference", device_methods),
            ("device_imports_only_in_reference", device_imports),
            ("device_branches_only_in_reference", device_branches),
            ("device_classes_only_in_reference", device_classes),
        ]:
            for item in comp[key]:
                if item not in bucket:
                    bucket.append(item)

    has_device_delta = bool(device_methods or device_imports or device_branches or device_classes)
    verdict = "delta" if has_device_delta else "parity"
    best = comparisons[0]

    if verdict == "delta":
        top = (device_methods + device_imports + device_branches + device_classes)[:5]
        recommendation = (
            f"参考实现 {best['reference']} 中存在 target 没有的设备/后端相关结构：{top}。"
            "这些差异只是候选，需人工判断是否真的需要移植；但在确认前不要仅凭 config.json 判 direct_use。"
        )
    else:
        recommendation = (
            "目标与同族参考未发现设备/后端相关结构差异；仍建议结合运行验证，不要只看 config.json。"
        )

    return {
        "target": {
            "file": str(model_file),
            "family": target_family,
            "classes": sorted(target["classes"])[:MAX_ITEMS],
            "device_imports": sorted(target["device_imports"])[:MAX_ITEMS],
        },
        "references": [
            {
                "file": c["file"],
                "family": family_of(c, stem_of(c["file"])),
                "score": c["_score"],
                "device_imports": sorted(c["device_imports"])[:MAX_ITEMS],
            }
            for c in siblings[:5]
        ],
        "delta": {
            "device_methods_only_in_reference": device_methods[:MAX_ITEMS],
            "device_imports_only_in_reference": device_imports[:MAX_ITEMS],
            "device_branches_only_in_reference": device_branches[:MAX_ITEMS],
            "device_classes_only_in_reference": device_classes[:MAX_ITEMS],
            "generic_methods_only_in_reference": best["generic_methods_only_in_reference"],
            "generic_classes_only_in_reference": best["generic_classes_only_in_reference"],
        },
        "verdict": verdict,
        "recommendation": recommendation,
    }


def main():
    ap = argparse.ArgumentParser(description="Audit structural parity between a model file and sibling reference implementations")
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--reference", action="append", default=[])
    ap.add_argument("--output", default=None)
    ap.add_argument("--fail-on-delta", action="store_true")
    args = ap.parse_args()

    models_dir = args.models_dir or str(Path(args.model_file).resolve().parent)

    try:
        result = audit(args.model_file, models_dir, args.reference)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[audit_model_parity] ERROR: {exc}\n")
        sys.exit(1)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        sys.stderr.write(f"[audit_model_parity] wrote {out}\n")
    print(text)

    sys.stderr.write(
        f"[audit_model_parity] verdict={result['verdict']} "
        f"methods={result['delta'].get('device_methods_only_in_reference', [])[:5]} "
        f"imports={result['delta'].get('device_imports_only_in_reference', [])[:5]}\n"
    )

    if args.fail_on_delta and result["verdict"] == "delta":
        sys.exit(1)


if __name__ == "__main__":
    main()
