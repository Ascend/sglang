#!/usr/bin/env python3
"""
Triton 算子替换脚本（NPU 版）。

扫描 sgl-kernel-npu 本地仓中所有子模块的 native PyTorch 实现，
用 sglang 对应的 Triton 版本替换，并自动修复 NPU 兼容性问题：
- torch.cuda.device → torch.npu.device
- 相对导入依赖自动复制（如 from .mamba_ssm import softplus）
- 不修改 ssd_combined.py 的 NPU import 路径（保持 sgl_kernel_npu.xxx.yyy）

用法:
    python triton_op_replace.py \
        --sglang-dir /path/to/sglang/python \
        --sgl-kernel-npu-dir /path/to/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu \
        --workspace-dir /path/to/workspace \
        [--dry-run]

退出码:
    0 - 替换成功
    1 - 参数错误或文件不存在
"""

import argparse
import json
import os
import re
import shutil
import sys


SKIP_FILES = {"__init__.py", "causal_conv1d.py"}

FNAME_SPECIAL_MAPPING = {
    "selective_state_update.py": "mamba_ssm.py",
}


def is_native_impl(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return "import triton" not in content
    except Exception:
        return False


def find_native_ops(npu_pkg_dir):
    native_ops = []
    for root, dirs, files in os.walk(npu_pkg_dir):
        for fname in sorted(files):
            if not fname.endswith(".py") or fname in SKIP_FILES:
                continue
            fpath = os.path.join(root, fname)
            if is_native_impl(fpath):
                rel_dir = os.path.relpath(root, npu_pkg_dir)
                native_ops.append((fname, fpath, rel_dir))
    return native_ops


def find_triton_counterpart(fname, sglang_layers_dir):
    for root, dirs, files in os.walk(sglang_layers_dir):
        candidates = [fname]
        if fname in FNAME_SPECIAL_MAPPING:
            candidates.append(FNAME_SPECIAL_MAPPING[fname])
        for candidate in candidates:
            cpath = os.path.join(root, candidate)
            if os.path.exists(cpath) and not is_native_impl(cpath):
                return cpath
    return None


def find_relative_import_deps(triton_path, sglang_ops_dir):
    deps = []
    with open(triton_path, "r", encoding="utf-8") as f:
        content = f.read()
    for match in re.finditer(r"^from \.(\w+) import ", content):
        dep_module = match.group(1) + ".py"
        dep_path = os.path.join(sglang_ops_dir, dep_module)
        if os.path.exists(dep_path) and dep_module not in deps:
            deps.append((dep_module, dep_path))
    return deps


def fix_npu_compat(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    original = content
    content = content.replace("torch.cuda.device", "torch.npu.device")
    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def replace_native_with_triton(native_ops, npu_pkg_dir, sglang_layers_dir, sglang_ops_dir, dry_run=False):
    results = []
    for fname, native_path, rel_dir in native_ops:
        triton_path = find_triton_counterpart(fname, sglang_layers_dir)
        if triton_path is None:
            results.append({
                "file": fname,
                "rel_dir": rel_dir,
                "action": "skip",
                "reason": "no_triton_counterpart",
            })
            continue

        dest_path = native_path
        if dry_run:
            results.append({
                "file": fname,
                "rel_dir": rel_dir,
                "action": "would_replace",
                "source": triton_path,
                "dest": dest_path,
            })
            continue

        backup_path = dest_path + ".native_backup"
        shutil.copy2(dest_path, backup_path)
        shutil.copy2(triton_path, dest_path)
        fix_npu_compat(dest_path)

        deps = find_relative_import_deps(dest_path, sglang_ops_dir)
        dep_results = []
        for dep_module, dep_path in deps:
            dest_dep_path = os.path.join(os.path.dirname(dest_path), dep_module)
            if not os.path.exists(dest_dep_path):
                shutil.copy2(dep_path, dest_dep_path)
                fix_npu_compat(dest_dep_path)
                dep_results.append({"module": dep_module, "action": "copied_and_fixed"})
            else:
                fix_npu_compat(dest_dep_path)
                dep_results.append({"module": dep_module, "action": "already_exists_fixed"})

        results.append({
            "file": fname,
            "rel_dir": rel_dir,
            "action": "replaced",
            "source": triton_path,
            "dest": dest_path,
            "backup": backup_path,
            "npu_compat_fixed": True,
            "relative_import_deps": dep_results,
        })

    return results


def main():
    ap = argparse.ArgumentParser(description="Replace native ops with Triton counterparts in sgl-kernel-npu (NPU-compatible)")
    ap.add_argument("--sglang-dir", required=True, help="Path to sglang/python directory")
    ap.add_argument("--sgl-kernel-npu-dir", required=True, help="Path to sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu directory")
    ap.add_argument("--workspace-dir", required=True, help="Workspace directory for output")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = ap.parse_args()

    sglang_layers_dir = os.path.join(args.sglang_dir, "sglang/srt/layers")
    sglang_ops_dir = os.path.join(args.sglang_dir, "sglang/srt/layers/attention/mamba/ops")
    npu_pkg_dir = args.sgl_kernel_npu_dir

    for d, name in [(sglang_layers_dir, "sglang layers"), (npu_pkg_dir, "npu pkg")]:
        if not os.path.isdir(d):
            print(f"ERROR: {name} directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    native_ops = find_native_ops(npu_pkg_dir)
    print(f"Found {len(native_ops)} native implementations:", file=sys.stderr)
    for fname, _, rel_dir in native_ops:
        print(f"  - {rel_dir}/{fname}", file=sys.stderr)

    replace_results = replace_native_with_triton(
        native_ops, npu_pkg_dir, sglang_layers_dir, sglang_ops_dir, dry_run=args.dry_run
    )
    print(f"\nReplacement results:", file=sys.stderr)
    for r in replace_results:
        loc = f"{r.get('rel_dir', '')}/{r['file']}"
        print(f"  {loc}: {r['action']}", file=sys.stderr)
        if r['action'] == 'replaced':
            print(f"    source: {r['source']}", file=sys.stderr)
            print(f"    backup: {r['backup']}", file=sys.stderr)
            print(f"    npu_compat_fixed: {r.get('npu_compat_fixed', False)}", file=sys.stderr)
            if r.get('relative_import_deps'):
                for dep in r['relative_import_deps']:
                    print(f"    dep: {dep['module']} -> {dep['action']}", file=sys.stderr)

    output = {
        "step": "triton_op_replace",
        "native_ops_found": [{"file": f, "path": p, "rel_dir": d} for f, p, d in native_ops],
        "replace_results": replace_results,
        "note": "ssd_combined.py NPU imports kept as sgl_kernel_npu.mamba.xxx (not changed to relative imports)",
        "dry_run": args.dry_run,
    }

    output_dir = os.path.join(args.workspace_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "triton_replace_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOutput written to: {output_path}", file=sys.stderr)

    replaced_count = sum(1 for r in replace_results if r["action"] in ("replaced", "would_replace"))
    if replaced_count == 0:
        print("No native ops to replace. All ops are already Triton.", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()