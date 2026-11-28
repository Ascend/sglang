#!/usr/bin/env python3
"""
Triton 算子替换脚本(NPU 版)。

扫描 sgl-kernel-npu 包中的 native PyTorch 实现(不含 `import triton` 的 .py),
在 sglang 的 `srt/layers/` 下按"文件名 + 顶层符号交集"匹配 Triton 版本并替换。

安全机制:
  - 只按文件名匹配还不够:必须与 native 文件至少共享一个顶层函数/类名,避免误替换 norm.py / rope.py 等同名文件
  - 首次替换前备份为 `<file>.native_backup`;已有备份不会被覆盖
  - 替换后做 `py_compile` 校验,失败自动回滚该文件
  - 记录清单 `<npu_pkg>/.triton_replace_manifest.json`,支持一键 `--rollback`

用法:
    python triton_op_replace.py \
        --sglang-dir /path/to/sglang/python \
        --sgl-kernel-npu-dir /path/to/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu \
        --workspace-dir /path/to/workspace \
        [--dry-run] [--mapping custom.json] [--no-npu-fix]

    # 回滚(只需 npu 包目录)
    python triton_op_replace.py \
        --sgl-kernel-npu-dir /path/to/.../sgl_kernel_npu \
        --rollback

退出码:
    0 - 成功(含 dry-run / rollback)
    1 - 参数错误、文件不存在或编译校验失败
"""

import argparse
import ast
import json
import os
import py_compile
import re
import shutil
import sys
import tempfile

SKIP_FILES = {"__init__.py", "causal_conv1d.py"}

DEFAULT_MAPPING = {
    "selective_state_update.py": "mamba_ssm.py",
}

# 按长度从长到短替换,避免 `torch.cuda.device` 抢先匹配 `torch.cuda.device_count`
CUDA_TO_NPU = [
    ("torch.cuda.current_device", "torch.npu.current_device"),
    ("torch.cuda.device_count", "torch.npu.device_count"),
    ("torch.cuda.get_device_name", "torch.npu.get_device_name"),
    ("torch.cuda.is_available", "torch.npu.is_available"),
    ("torch.cuda.set_device", "torch.npu.set_device"),
    ("torch.cuda.synchronize", "torch.npu.synchronize"),
    ("torch.cuda.empty_cache", "torch.npu.empty_cache"),
    ("torch.cuda.device", "torch.npu.device"),
]

MANIFEST_NAME = ".triton_replace_manifest.json"


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def is_triton_impl(filepath):
    try:
        return "import triton" in read_text(filepath)
    except OSError:
        return False


def public_defs(path):
    """顶层函数/类名集合;解析失败返回空集合。"""
    try:
        tree = ast.parse(read_text(path))
    except (OSError, SyntaxError):
        return set()
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def find_native_ops(npu_pkg_dir):
    native_ops = []
    for root, _dirs, files in os.walk(npu_pkg_dir):
        for fname in sorted(files):
            if not fname.endswith(".py") or fname in SKIP_FILES:
                continue
            fpath = os.path.join(root, fname)
            if not is_triton_impl(fpath):
                rel_dir = os.path.relpath(root, npu_pkg_dir)
                native_ops.append((fname, fpath, rel_dir))
    return native_ops


def find_triton_counterpart(fname, sglang_layers_dir, mapping, native_symbols):
    """返回 (path, reason)。reason 仅在没有匹配时使用。"""
    candidates = [fname]
    if fname in mapping:
        candidates.append(mapping[fname])

    for candidate in candidates:
        for root, _dirs, files in os.walk(sglang_layers_dir):
            if candidate not in files:
                continue
            cpath = os.path.join(root, candidate)
            if is_triton_impl(cpath) is False:
                continue
            shared = native_symbols & public_defs(cpath)
            if shared:
                return cpath, None
    return None, "no_triton_counterpart_with_shared_symbols"


def find_relative_import_deps(triton_path):
    deps = []
    base = os.path.dirname(triton_path)
    content = read_text(triton_path)
    for match in re.finditer(r"^from \.(\w+) import ", content, re.MULTILINE):
        dep_module = match.group(1) + ".py"
        dep_path = os.path.join(base, dep_module)
        if os.path.exists(dep_path) and dep_module not in [d[0] for d in deps]:
            deps.append((dep_module, dep_path))
    return deps


def apply_npu_compat(content):
    for old, new in CUDA_TO_NPU:
        content = content.replace(old, new)
    return content


def compile_ok(path):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            py_compile.compile(path, cfile=os.path.join(tmp, "check.pyc"), doraise=True)
        return True, None
    except py_compile.PyCompileError as exc:
        return False, str(exc)


def load_manifest(npu_pkg_dir):
    path = os.path.join(npu_pkg_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return {"version": 1, "entries": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("entries", [])
        return data
    except Exception:  # noqa: BLE001
        return {"version": 1, "entries": []}


def save_manifest(npu_pkg_dir, manifest):
    path = os.path.join(npu_pkg_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def rollback(npu_pkg_dir):
    manifest = load_manifest(npu_pkg_dir)
    restored, removed, missing = [], [], []

    for entry in manifest.get("entries", []):
        dest = entry.get("dest")
        backup = entry.get("backup")
        if dest and backup and os.path.exists(backup):
            shutil.copy2(backup, dest)
            os.remove(backup)
            restored.append(dest)
        elif dest and not os.path.exists(dest):
            missing.append(dest)
        for dep in entry.get("copied_deps", []):
            if os.path.exists(dep):
                os.remove(dep)
                removed.append(dep)

    # 兜底:即使没有 manifest,也恢复遗留的 .native_backup
    for root, _dirs, files in os.walk(npu_pkg_dir):
        for fname in files:
            if fname.endswith(".native_backup"):
                backup = os.path.join(root, fname)
                dest = backup[: -len(".native_backup")]
                shutil.copy2(backup, dest)
                os.remove(backup)
                if dest not in restored:
                    restored.append(dest)

    manifest_path = os.path.join(npu_pkg_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    return {"restored": restored, "removed_deps": removed, "missing_backups": missing}


def replace_native_with_triton(native_ops, npu_pkg_dir, sglang_layers_dir, mapping, dry_run, npu_fix):
    results = []
    manifest = load_manifest(npu_pkg_dir)

    for fname, native_path, rel_dir in native_ops:
        native_symbols = public_defs(native_path)
        triton_path, skip_reason = find_triton_counterpart(fname, sglang_layers_dir, mapping, native_symbols)
        if triton_path is None:
            results.append({
                "file": fname,
                "rel_dir": rel_dir,
                "action": "skip",
                "reason": skip_reason,
            })
            continue

        if dry_run:
            results.append({
                "file": fname,
                "rel_dir": rel_dir,
                "action": "would_replace",
                "source": triton_path,
                "dest": native_path,
                "shared_symbols": sorted(native_symbols & public_defs(triton_path)),
            })
            continue

        backup_path = native_path + ".native_backup"
        backup_created = False
        if not os.path.exists(backup_path):
            shutil.copy2(native_path, backup_path)
            backup_created = True

        content = read_text(triton_path)
        if npu_fix:
            content = apply_npu_compat(content)
        with open(native_path, "w", encoding="utf-8") as f:
            f.write(content)

        ok, compile_error = compile_ok(native_path)
        if not ok:
            shutil.copy2(backup_path, native_path)
            results.append({
                "file": fname,
                "rel_dir": rel_dir,
                "action": "reverted",
                "source": triton_path,
                "dest": native_path,
                "reason": "compile_failed",
                "error": (compile_error or "")[:500],
            })
            continue

        copied_deps = []
        for dep_module, dep_path in find_relative_import_deps(triton_path):
            dest_dep_path = os.path.join(os.path.dirname(native_path), dep_module)
            if not os.path.exists(dest_dep_path):
                shutil.copy2(dep_path, dest_dep_path)
                if npu_fix:
                    dep_content = apply_npu_compat(read_text(dest_dep_path))
                    with open(dest_dep_path, "w", encoding="utf-8") as f:
                        f.write(dep_content)
                copied_deps.append(dest_dep_path)

        entry = {
            "dest": native_path,
            "backup": backup_path,
            "copied_deps": copied_deps,
            "file": fname,
            "rel_dir": rel_dir,
        }
        # 同一目标重复执行时只保留最新一条
        manifest["entries"] = [e for e in manifest["entries"] if e.get("dest") != native_path]
        manifest["entries"].append(entry)

        results.append({
            "file": fname,
            "rel_dir": rel_dir,
            "action": "replaced",
            "source": triton_path,
            "dest": native_path,
            "backup": backup_path,
            "backup_created": backup_created,
            "npu_compat_fixed": npu_fix,
            "compile_verified": True,
            "shared_symbols": sorted(native_symbols & public_defs(triton_path)),
            "copied_deps": copied_deps,
        })

    if not dry_run:
        save_manifest(npu_pkg_dir, manifest)

    return results


def main():
    ap = argparse.ArgumentParser(description="Replace native ops with Triton counterparts in sgl-kernel-npu (NPU-compatible)")
    ap.add_argument("--sglang-dir", help="Path to sglang/python directory")
    ap.add_argument("--sgl-kernel-npu-dir", required=True,
                    help="Path to sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu directory")
    ap.add_argument("--workspace-dir", help="Workspace directory for output JSON (required unless --rollback)")
    ap.add_argument("--mapping", help="Optional JSON file: {native_filename: triton_filename}")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    ap.add_argument("--rollback", action="store_true", help="Restore all files replaced by this script")
    ap.add_argument("--no-npu-fix", action="store_true", help="Do not rewrite torch.cuda.* to torch.npu.*")
    args = ap.parse_args()

    npu_pkg_dir = args.sgl_kernel_npu_dir
    if not os.path.isdir(npu_pkg_dir):
        print(f"ERROR: npu pkg directory not found: {npu_pkg_dir}", file=sys.stderr)
        sys.exit(1)

    if args.rollback:
        info = rollback(npu_pkg_dir)
        print(json.dumps(info, indent=2, ensure_ascii=False), file=sys.stderr)
        if args.workspace_dir:
            output_dir = os.path.join(args.workspace_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "triton_replace_result.json"), "w", encoding="utf-8") as f:
                json.dump({"step": "triton_op_replace", "rollback": info, "dry_run": False}, f, indent=2, ensure_ascii=False)
        sys.exit(0)

    if not args.sglang_dir or not args.workspace_dir:
        print("ERROR: --sglang-dir and --workspace-dir are required unless --rollback", file=sys.stderr)
        sys.exit(1)

    sglang_layers_dir = os.path.join(args.sglang_dir, "sglang/srt/layers")
    if not os.path.isdir(sglang_layers_dir):
        print(f"ERROR: sglang layers directory not found: {sglang_layers_dir}", file=sys.stderr)
        sys.exit(1)

    mapping = dict(DEFAULT_MAPPING)
    if args.mapping:
        with open(args.mapping, "r", encoding="utf-8") as f:
            mapping.update(json.load(f))

    native_ops = find_native_ops(npu_pkg_dir)
    print(f"Found {len(native_ops)} native implementations:", file=sys.stderr)
    for fname, _, rel_dir in native_ops:
        print(f"  - {rel_dir}/{fname}", file=sys.stderr)

    replace_results = replace_native_with_triton(
        native_ops, npu_pkg_dir, sglang_layers_dir, mapping,
        dry_run=args.dry_run, npu_fix=not args.no_npu_fix,
    )

    print("\nReplacement results:", file=sys.stderr)
    for r in replace_results:
        loc = f"{r.get('rel_dir', '')}/{r['file']}"
        print(f"  {loc}: {r['action']}", file=sys.stderr)
        if r["action"] == "replaced":
            print(f"    source: {r['source']}", file=sys.stderr)
            print(f"    backup: {r['backup']}", file=sys.stderr)
            print(f"    shared_symbols: {r.get('shared_symbols')}", file=sys.stderr)
        elif r["action"] in ("skip", "reverted"):
            print(f"    reason: {r.get('reason')}", file=sys.stderr)

    output = {
        "step": "triton_op_replace",
        "native_ops_found": [{"file": f, "path": p, "rel_dir": d} for f, p, d in native_ops],
        "replace_results": replace_results,
        "manifest": os.path.join(npu_pkg_dir, MANIFEST_NAME),
        "rollback_command": (
            f"python {os.path.abspath(__file__)} "
            f"--sgl-kernel-npu-dir {npu_pkg_dir} --rollback"
        ),
        "dry_run": args.dry_run,
    }

    output_dir = os.path.join(args.workspace_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "triton_replace_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOutput written to: {output_path}", file=sys.stderr)

    failed = [r for r in replace_results if r["action"] == "reverted"]
    if failed:
        print(f"ERROR: {len(failed)} file(s) failed compile check and were reverted", file=sys.stderr)
        sys.exit(1)

    replaced_count = sum(1 for r in replace_results if r["action"] in ("replaced", "would_replace"))
    if replaced_count == 0:
        print("No native ops to replace. All ops are already Triton or lack a safe counterpart.", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
