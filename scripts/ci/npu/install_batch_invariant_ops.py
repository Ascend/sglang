"""Install A3 batch-invariant dependencies for the multimodal chat CI test.

Called from that test's setUpClass, after CI selection/partitioning. Installing
here also lets the server inherit LD_LIBRARY_PATH without editing the workflow.
Artifacts are the 2.0.0 packages used by vllm-ascend/csrc/build_batch_invariant_ops.sh.
"""

import fcntl
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BASE_URL = "https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend"
ARTIFACTS = {
    "cann-ops-batch_invariant-A3-2.0.0-linux.aarch64.run": "7b78c05ac31a48398e0902fdbf379d4f481f6fc31ecba0fac646595f70e9c5a5",
    "batch_invariant-torch_ops_extension-2.0.0.zip": "88c0a69d5a9487ccc54bc71ef4e3581756098f47d87188da52c299e792ab76c2",
}
CHECK = """
import ctypes
import sys
import torch
import torch_npu
import batch_invariant_ops

lib = ctypes.CDLL(sys.argv[1])
getattr(lib, "aclnnFusedInferAttentionScoreBatchInvariantV3")
ops = torch.ops.batch_invariant_ops
for name in (
    "npu_mm_batch_invariant",
    "npu_matmul_batch_invariant",
    "npu_reduce_mean_batch_invariant",
    "npu_log_softmax_batch_invariant",
    "_npu_fused_infer_attention_score_batch_invariant_get_max_workspace",
    "_npu_fused_infer_attention_score_batch_invariant_infer_output",
):
    getattr(ops, name).default
ops.npu_fused_infer_attention_score_batch_invariant.default
ops.npu_fused_infer_attention_score_batch_invariant.out
"""


def _download(cache, name, expected):
    path = cache / name

    def verified():
        digest = hashlib.sha256()
        if not path.is_file():
            return False
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected

    if not verified():
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                "--silent",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "600",
                "--output",
                str(path),
                f"{BASE_URL}/{name}",
            ],
            check=True,
        )
        if not verified():
            raise RuntimeError(f"SHA256 mismatch for {name}")
    return path


def install():
    """Prepare this test process's environment and install missing dependencies."""
    if platform.machine() != "aarch64":
        raise RuntimeError("This CI installer requires an A3 aarch64 image")
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        raise RuntimeError("Source the CANN set_env.sh before running this test")
    vendor_lib = Path(ascend_home).resolve() / "opp/vendors/batch_invariant/op_api/lib"
    old_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [str(vendor_lib)] + [p for p in old_paths if p and p != str(vendor_lib)]
    )
    check_cmd = [sys.executable, "-c", CHECK, str(vendor_lib / "libcust_opapi.so")]
    cache = Path(tempfile.gettempdir()) / "sglang-batch-invariant-a3-2.0.0"
    cache.mkdir(parents=True, exist_ok=True)
    # Serialize installation if multiple test processes share the same container.
    with (cache / "install.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            subprocess.run(
                check_cmd, capture_output=True, timeout=120, check=False
            ).returncode
            == 0
        ):
            print("[batch-invariant CI] Dependencies already available", flush=True)
            return

        print("[batch-invariant CI] Installing A3 2.0.0 dependencies", flush=True)
        run_package, source_package = [
            _download(cache, name, sha) for name, sha in ARTIFACTS.items()
        ]
        # The native installer must register in CANN's default vendor directory.
        install_env = os.environ.copy()
        install_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
        subprocess.run(
            ["bash", str(run_package)], env=install_env, check=True, timeout=600
        )
        with tempfile.TemporaryDirectory(prefix="build-", dir=cache) as build_dir:
            with zipfile.ZipFile(source_package) as archive:
                archive.extractall(build_dir)
            source = Path(build_dir) / "torch_ops_extension/batch_invariant_ops"
            build_env = os.environ.copy()
            build_env["USE_NINJA"] = "1"
            # Avoid exhausting smaller CI containers with concurrent C++ compiles.
            build_env.setdefault("MAX_JOBS", "2")
            subprocess.run(
                [sys.executable, "setup.py", "bdist_wheel"],
                cwd=source,
                env=build_env,
                check=True,
                timeout=1200,
            )
            wheels = list((source / "dist").glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(
                    f"Expected one batch_invariant_ops wheel, got {wheels}"
                )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    str(wheels[0]),
                ],
                check=True,
                timeout=300,
            )
        subprocess.run(check_cmd, check=True, timeout=120)
        print("[batch-invariant CI] Dependencies installed and registered", flush=True)


if __name__ == "__main__":
    install()
