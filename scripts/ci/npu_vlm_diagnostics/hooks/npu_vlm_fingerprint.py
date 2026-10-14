"""Read-only binary fingerprints; importing this module never initializes NPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def file_info(path, cache=None):
    path = Path(path)
    item = {"path": str(path), "resolved": str(path.resolve())}
    try:
        stat = path.stat()
        cache = {} if cache is None else cache
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in cache:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            cache[key] = digest.hexdigest()
        item.update(bytes=stat.st_size, sha256=cache[key])
    except OSError as error:
        item["error"] = repr(error)
    return item


def binary_manifest():
    cache = {}
    groups = {}
    versions = {}
    for name in ("torch_npu", "torch"):
        try:
            dist = importlib.metadata.distribution(name)
            versions[name] = dist.version
            root = Path(dist.locate_file(name))
            if name == "torch_npu":
                groups[name] = (root, lambda p: ".so" in p.name or p.suffix == ".py")
            else:
                groups[name] = (root / "lib", lambda p: ".so" in p.name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    groups["cann_lib64"] = (
        Path("/usr/local/Ascend/cann/arm64-linux/lib64"),
        lambda p: ".so" in p.name,
    )
    groups["opp"] = (
        Path("/usr/local/Ascend/cann/opp"),
        lambda p: (
            ".so" in p.name
            or (
                p.suffix in (".o", ".json", ".py", ".info")
                and any(
                    word in str(p).lower()
                    for word in ("matmul", "mat_mul", "addmm", "gemm")
                )
            )
        ),
    )
    groups["driver"] = (
        Path("/usr/local/Ascend/driver/lib64"),
        lambda p: ".so" in p.name,
    )
    result = {"time_ns": time.time_ns(), "versions": versions, "groups": {}}
    for label, (root, accept) in groups.items():
        files = {}
        if root.is_dir():
            for parent, _dirs, names in os.walk(root):
                for name in sorted(names):
                    path = Path(parent) / name
                    if accept(path):
                        files[str(path.relative_to(root))] = file_info(path, cache)
        result["groups"][label] = {
            "root": str(root),
            "resolved": str(root.resolve()),
            "files": files,
        }
    result["mounts"] = [
        line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if any(word in line for word in ("Ascend", "site-packages", " /usr ", " / "))
    ]
    result["versions_files"] = {
        path: Path(path).read_text(errors="replace")
        for path in (
            "/usr/local/Ascend/cann/arm64-linux/ascend_toolkit_install.info",
            "/usr/local/Ascend/cann/opp/version.info",
            "/usr/local/Ascend/driver/version.info",
            "/usr/local/Ascend/firmware/version.info",
        )
        if Path(path).is_file()
    }
    return result


def runtime_manifest():
    """Called only inside an already initialized model worker, after server setup."""
    import torch
    import torch_npu

    result = {"initialized": torch_npu.npu.is_initialized(), "options": {}}
    if not result["initialized"]:
        return result
    for name in (
        "ACL_PRECISION_MODE",
        "ACL_OP_SELECT_IMPL_MODE",
        "ACL_OPTYPELIST_FOR_IMPLMODE",
        "ALLOW_MATMUL_HF32",
        "ALLOW_CONV_HF32",
        "CUBE_MATH_TYPE",
        "ALLOW_INTERNAL_FORMAT",
        "MM_BMM_ND_ENABLE",
        "jitCompile",
    ):
        try:
            value = torch_npu._C._npu_getOption(name)
            result["options"][name] = (
                value.decode() if isinstance(value, bytes) else value
            )
        except Exception as error:
            result["options"][name] = {"error": repr(error)}
    for name, query in (
        ("jit_compile_false", torch_npu.npu.is_jit_compile_false),
        ("mm_bmm_format_nd", torch_npu.npu.get_mm_bmm_format_nd),
        ("deterministic_algorithms", torch.are_deterministic_algorithms_enabled),
    ):
        try:
            result[name] = query()
        except Exception as error:
            result[name] = {"error": repr(error)}
    result["device_properties"] = str(
        torch.npu.get_device_properties(torch.npu.current_device())
    )
    result["env"] = {
        name: os.environ.get(name)
        for name in (
            "LD_LIBRARY_PATH",
            "ASCEND_HOME_PATH",
            "ASCEND_OPP_PATH",
            "ASCEND_CUSTOM_OPP_PATH",
            "ASCEND_RT_VISIBLE_DEVICES",
            "PYTORCH_NPU_ALLOC_CONF",
            "NPU_ASD_ENABLE",
            "ACL_OP_COMPILER_CACHE_MODE",
            "ACL_OP_COMPILER_CACHE_DIR",
            "ACL_PRECISION_MODE",
            "ALLOW_MATMUL_HF32",
            "CUBE_MATH_TYPE",
            "TORCH_NPU_DEVICE_CAPABILITY",
        )
    }
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if (
            path.startswith("/")
            and ".so" in path
            and any(word in path for word in ("Ascend", "torch_npu", "/torch/lib/"))
        ):
            paths.add(path)
    result["loaded_libraries"] = [file_info(path) for path in sorted(paths)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = binary_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "versions": manifest["versions"],
                "file_counts": {
                    key: len(value["files"])
                    for key, value in manifest["groups"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
