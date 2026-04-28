#!/usr/bin/env python3
"""
环境检查脚本
检查模型适配所需的运行环境
"""

import subprocess
import sys
import json
import os
import re
import socket
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    return {
        "status": "ok" if version >= (3, 8) else "error",
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "message": "Python版本满足要求" if version >= (3, 8) else "需要Python 3.8+"
    }


def check_package_installed(package_name, import_name=None):
    """检查包是否安装"""
    import_name = import_name or package_name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        return {"status": "ok", "version": version}
    except ImportError:
        return {"status": "missing", "version": None}


def check_torch_device():
    """检查PyTorch设备支持"""
    result = {"gpu": False, "npu": False, "gpu_count": 0, "npu_count": 0}
    
    try:
        import torch
        
        if torch.cuda.is_available():
            result["gpu"] = True
            result["gpu_count"] = torch.cuda.device_count()
            result["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(result["gpu_count"])]
        
        try:
            import torch_npu
            if torch.npu.is_available():
                result["npu"] = True
                result["npu_count"] = torch.npu.device_count()
                result["npu_names"] = [torch.npu.get_device_name(i) for i in range(result["npu_count"])]
        except ImportError:
            pass
            
    except ImportError:
        pass
    
    return result


def check_memory():
    """检查系统内存"""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        total_kb = 0
        available_kb = 0
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                total_kb = int(line.split()[1])
            elif line.startswith('MemAvailable:'):
                available_kb = int(line.split()[1])
        
        return {
            "total_gb": round(total_kb / 1024 / 1024, 2),
            "available_gb": round(available_kb / 1024 / 1024, 2)
        }
    except:
        return {"total_gb": "unknown", "available_gb": "unknown"}


def check_disk_space(path="/"):
    """检查磁盘空间"""
    try:
        stat = os.statvfs(path)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return {
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2)
        }
    except:
        return {"total_gb": "unknown", "free_gb": "unknown"}


def check_container():
    """检测当前进程是否在容器内运行，并采集执行子 agent 命令所需的运行时上下文。

    生成的字段会注入到 device_info.json 的 `container` 段——子 agent
    （debug-engineer / test-validator）凭此判断命令应该在哪个 shell 上下文执行、
    需要 source 哪些环境变量；缺失会导致它们在错误的 shell 中跑命令而找不到 sglang/torch_npu。
    """
    info = {
        "in_container": False,
        "runtime": None,
        "container_id": None,
        "hostname": None,
        "image": None,
        "key_env": {},
        "cwd": os.getcwd(),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
    }

    try:
        info["hostname"] = socket.gethostname()
    except Exception:
        try:
            with open("/etc/hostname") as f:
                info["hostname"] = f.read().strip()
        except Exception:
            pass

    if os.path.exists("/.dockerenv"):
        info["in_container"] = True
        info["runtime"] = "docker"

    runtime_markers = (
        ("docker", "docker"),
        ("containerd", "containerd"),
        ("kubepods", "kubernetes"),
        ("podman", "podman"),
        ("lxc", "lxc"),
    )
    try:
        with open("/proc/1/cgroup") as f:
            cgroup = f.read()
        for marker, runtime_name in runtime_markers:
            if marker in cgroup:
                info["in_container"] = True
                if not info["runtime"]:
                    info["runtime"] = runtime_name
                if not info["container_id"]:
                    m = re.search(r"([0-9a-f]{64})", cgroup)
                    if m:
                        info["container_id"] = m.group(1)
                break
    except Exception:
        pass

    try:
        with open("/proc/1/sched") as f:
            first = f.readline().strip()
        if first and not first.startswith(("systemd", "init")):
            info["in_container"] = True
            info["init_process"] = first.split()[0]
    except Exception:
        pass

    info["image"] = (
        os.environ.get("CONTAINER_IMAGE")
        or os.environ.get("IMAGE_NAME")
        or os.environ.get("DOCKER_IMAGE")
    )

    env_keys = (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "PATH",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_OPP_PATH",
        "ASCEND_AICPU_PATH",
        "HCCL_CONNECT_TIMEOUT",
        "HCCL_BUFFSIZE",
        "PYTORCH_NPU_ALLOC_CONF",
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "VIRTUAL_ENV",
        "CONDA_DEFAULT_ENV",
    )
    info["key_env"] = {k: os.environ[k] for k in env_keys if k in os.environ}

    try:
        info["python_executable"] = sys.executable
    except Exception:
        pass

    return info


def check_sglang():
    """检查SGLang安装"""
    result = check_package_installed("sglang")
    if result["status"] == "ok":
        try:
            import sglang
            result["path"] = str(Path(sglang.__file__).parent)
        except:
            pass
    return result


def check_transformers():
    """检查transformers版本"""
    return check_package_installed("transformers")


def run_environment_check(output_file=None):
    """运行完整的环境检查"""
    results = {
        "python": check_python_version(),
        "packages": {
            "torch": check_package_installed("torch"),
            "transformers": check_transformers(),
            "sglang": check_sglang(),
            "flashinfer": check_package_installed("flashinfer"),
            "requests": check_package_installed("requests")
        },
        "devices": check_torch_device(),
        "memory": check_memory(),
        "disk": check_disk_space(),
        "container": check_container()
    }
    
    summary = {
        "ready": True,
        "issues": []
    }
    
    if results["python"]["status"] != "ok":
        summary["ready"] = False
        summary["issues"].append("Python版本不满足要求")
    
    for pkg, info in results["packages"].items():
        if info["status"] == "missing" and pkg in ["torch", "transformers", "requests"]:
            summary["ready"] = False
            summary["issues"].append(f"缺少必要包: {pkg}")
    
    if not results["devices"]["gpu"] and not results["devices"]["npu"]:
        summary["issues"].append("未检测到GPU或NPU设备")
    
    results["summary"] = summary
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def print_report(results):
    """打印检查报告"""
    print("=" * 60)
    print("环境检查报告")
    print("=" * 60)
    
    print(f"\n[Python] {results['python']['version']} - {results['python']['message']}")
    
    print("\n[包检查]")
    for pkg, info in results["packages"].items():
        status = "✓" if info["status"] == "ok" else "✗"
        version = info.get("version", "N/A")
        print(f"  {status} {pkg}: {version}")
    
    print("\n[设备检查]")
    devices = results["devices"]
    if devices["gpu"]:
        print(f"  ✓ GPU: {devices['gpu_count']}个")
        for name in devices.get("gpu_names", []):
            print(f"    - {name}")
    else:
        print("  ✗ GPU: 未检测到")
    
    if devices["npu"]:
        print(f"  ✓ NPU: {devices['npu_count']}个")
        for name in devices.get("npu_names", []):
            print(f"    - {name}")
    else:
        print("  ✗ NPU: 未检测到")
    
    print("\n[容器检查]")
    container = results.get("container", {})
    if container.get("in_container"):
        print(f"  ✓ 容器: {container.get('runtime') or 'unknown'}")
        if container.get("container_id"):
            print(f"    container_id: {container['container_id'][:12]}")
        if container.get("hostname"):
            print(f"    hostname:     {container['hostname']}")
        if container.get("image"):
            print(f"    image:        {container['image']}")
        env = container.get("key_env") or {}
        if env:
            print(f"    保留环境变量: {', '.join(sorted(env.keys()))}")
    else:
        print("  · 主机环境（非容器）")

    print("\n[资源检查]")
    mem = results["memory"]
    print(f"  内存: {mem['available_gb']}GB 可用 / {mem['total_gb']}GB 总计")
    
    disk = results["disk"]
    print(f"  磁盘: {disk['free_gb']}GB 可用 / {disk['total_gb']}GB 总计")
    
    print("\n[总结]")
    summary = results["summary"]
    if summary["ready"]:
        print("  ✓ 环境就绪")
    else:
        print("  ✗ 环境存在问题")
        for issue in summary["issues"]:
            print(f"    - {issue}")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="环境检查脚本")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，只输出JSON")
    args = parser.parse_args()
    
    results = run_environment_check(args.output)
    
    if not args.quiet:
        print_report(results)
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
