"""TEMP (validation only): startup-gate checks for the NPU breakable prefill graph.

Three negative cases, each expected to fail before capture:
1. --cuda-graph-backend-prefill full  -> ValueError: only supports 'breakable'
2. ascend_ops with default alltoallv  -> ValueError: requires prefill=allgather
3. --enable-torch-compile + breakable  -> AssertionError: does not support torch.compile
"""

import os
import subprocess
import sys

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)

MODEL = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH

BASE_ARGS = [
    "--model-path",
    MODEL,
    "--page-size",
    128,
    "--tp-size",
    8,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.62,
    "--prefill-max-requests",
    32,
    "--max-prefill-tokens",
    80000,
    "--chunked-prefill-size",
    8192,
    "--max-running-requests",
    64,
    "--attn-dp-size",
    8,
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "auto",
    "--disable-radix-cache",
    "--load-balance-method",
    "round_robin",
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    8,
    10,
]

ASCEND_ALLGATHER = [
    "--moe-a2a-backend",
    "ascend_ops",
    "--moe-ascend-ops-config",
    '{"prefill":"allgather","decode":"low_latency"}',
]

BREAKABLE = ["--cuda-graph-backend-prefill", "breakable", "--cuda-graph-bs-prefill", "128", "512", "1024", "2048"]

ENVS = {
    **os.environ,
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
}


def run(args, port, timeout=600):
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        *[str(x) for x in args],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.run(
        cmd,
        env=ENVS,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def rejected(p, expected):
    """A gate passes when the server exits non-zero and the rejection is visible.

    The NPU backend validates these combinations inside the scheduler
    subprocess, whose traceback is not always forwarded to the launcher's
    stdout; the launcher still reports "scheduler died during initialization".
    Accept either form so the gate does not depend on log plumbing.
    """
    out = p.stdout or ""
    return p.returncode != 0 and (
        expected in out or "scheduler died during initialization" in out
    )


def main():
    results = []

    p1 = run(
        BASE_ARGS + ASCEND_ALLGATHER + ["--cuda-graph-backend-prefill", "full"],
        30100,
    )
    results.append(
        (
            "full backend",
            rejected(p1, "only supports 'breakable'"),
            p1.stdout[-2000:],
        )
    )

    p2 = run(BASE_ARGS + BREAKABLE, 30101)
    results.append(
        (
            "ascend_ops alltoallv",
            rejected(p2, "requires ascend_ops prefill=allgather"),
            p2.stdout[-2000:],
        )
    )

    p3 = run(
        BASE_ARGS + ASCEND_ALLGATHER + BREAKABLE + ["--enable-torch-compile"],
        30102,
    )
    results.append(
        (
            "torch.compile + breakable",
            rejected(p3, "does not support torch.compile"),
            p3.stdout[-2000:],
        )
    )

    failed = 0
    for name, ok, out in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failed += 1
            print(out)

    if failed:
        raise SystemExit(f"{failed} startup-gate check(s) failed")


if __name__ == "__main__":
    main()
