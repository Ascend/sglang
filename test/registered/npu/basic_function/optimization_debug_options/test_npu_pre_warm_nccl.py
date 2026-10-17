import io
import os
import re
import subprocess
import sys
import threading
import time
import unittest
from urllib.parse import urlparse

import psutil
import requests

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import wait_port_available
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH, logger
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
)

register_npu_ci(est_time=600, suite="full-2-npu-a3", nightly=True)

BASE_URL = DEFAULT_URL_FOR_TEST
HOST = urlparse(BASE_URL).hostname
PORT = urlparse(BASE_URL).port


class TestPreWarmNccl(CustomTestCase):
    """Testcase: verify --pre-warm-nccl server starts and serves correctly

        [Test Category] Parameter
        [Test Target] --pre-warm-nccl

    HCCL comm creation is lazy (first dist.all_reduce).
    Without --pre-warm-nccl, this adds ~700-900ms to the first request's TTFT.
    With --pre-warm-nccl, it happens at bootstrap.

    Three warmers must be disabled to expose this gap:
    1. /health_generate → SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
    2. popen_launch_server's health check → use raw subprocess.Popen
    3. HCCL port conflicts → HCCL_PORT/NPU_SOCKET_PORT_RANGE=auto
    """

    model = QWEN3_0_6B_WEIGHTS_PATH

    def _close(self, *streams):
        for s in streams:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    def _get_base_cmd(self, pre_warm: bool) -> list[str]:
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            "--trust-remote-code",
            "--tp-size",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--skip-server-warmup",
            "--device",
            "npu",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ]
        if pre_warm:
            cmd.append("--pre-warm-nccl")
        return cmd

    def _get_env(self) -> dict:
        return {
            **os.environ,
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
            "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        }

    def _start_server(self, pre_warm: bool, capture_logs: bool = True):
        """Start server process and return (proc, stdout, stderr)."""
        stdout = io.StringIO() if capture_logs else None
        stderr = io.StringIO() if capture_logs else None

        proc = subprocess.Popen(
            self._get_base_cmd(pre_warm),
            stdout=subprocess.PIPE if capture_logs else None,
            stderr=subprocess.PIPE if capture_logs else None,
            text=True,
            env=self._get_env(),
            start_new_session=True,
        )

        if capture_logs:

            def _pipe(src, dst):
                try:
                    for line in src:
                        dst.write(line)
                        dst.flush()
                        sys.__stdout__.write(line)
                        sys.__stdout__.flush()
                except (ValueError, OSError):
                    pass

            threading.Thread(
                target=_pipe, args=(proc.stdout, stdout), daemon=True
            ).start()
            threading.Thread(
                target=_pipe, args=(proc.stderr, stderr), daemon=True
            ).start()

        return proc, stdout, stderr

    def _wait_until_ready(
        self, proc: subprocess.Popen, timeout: int = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    ):
        """Block until server health endpoint responds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited early with code {proc.returncode}")
            try:
                if requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("Server failed to become healthy within timeout")

    def _stop_server(self, proc):
        if proc is None:
            return
        kill_process_tree(proc.pid, wait_timeout=60)
        self._close(proc.stdin, proc.stdout, proc.stderr)
        for _ in range(3):
            try:
                proc.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                logger.warning(f"Server PID {proc.pid} still alive, retrying...")
        wait_port_available(PORT, "server", timeout_s=120)
        for _ in range(60):
            alive = [
                p.info["name"]
                for p in psutil.process_iter(["pid", "name", "status"])
                if p.info["name"]
                and p.info["name"].startswith("sglang")
                and p.info["status"] != psutil.STATUS_ZOMBIE
            ]
            if not alive:
                break
            time.sleep(2)
        else:
            logger.info(f"sglang procs still alive: {alive}")

    def _run_benchmark(self) -> dict:
        """Run the standard serving benchmark and return metrics."""
        args = get_benchmark_args(
            base_url=BASE_URL,
            backend="sglang",
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=100,
            random_input_len=3500,
            random_output_len=1500,
            request_rate=float("inf"),
        )
        args.warmup_requests = 0
        args.model = self.model
        return run_benchmark(args)

    def _get_logs(self, *buffers) -> str:
        """Concatenate log buffers into a single string."""
        return "".join(b.getvalue() if b else "" for b in buffers)

    def test_pre_warm_nccl_colocated_cold_start(self):
        # Phase 0: Warm up kernel cache (discard results)
        logger.info("Phase 0: Warming kernel cache...")
        proc, *_ = self._start_server(pre_warm=False, capture_logs=False)
        self._wait_until_ready(proc)
        self._run_benchmark()
        self._stop_server(proc)

        # Phase 1: With --pre-warm-nccl
        logger.info("Phase 1: Testing with --pre-warm-nccl")
        proc_warm, out_warm, err_warm = self._start_server(pre_warm=True)
        self._wait_until_ready(proc_warm)
        res_warm = self._run_benchmark()
        logs_warm = self._get_logs(out_warm, err_warm)
        self._stop_server(proc_warm)

        # Phase 2: Without --pre-warm-nccl
        logger.info("Phase 2: Testing without --pre-warm-nccl")
        proc_cold, out_cold, err_cold = self._start_server(pre_warm=False)
        self._wait_until_ready(proc_cold)
        res_cold = self._run_benchmark()
        logs_cold = self._get_logs(out_cold, err_cold)
        self._stop_server(proc_cold)

        # Assertions
        self.assertIn(
            "NCCL/RCCL/HCCL warmup completed",
            logs_warm,
            "With --pre-warm-nccl, HCCL warmup message must appear",
        )
        match = re.search(r"NCCL/RCCL/HCCL warmup completed in ([0-9.]+)s", logs_warm)
        if match:
            logger.info(f"HCCL warmup took {match.group(1)}s")

        self.assertNotIn(
            "NCCL/RCCL/HCCL warmup completed",
            logs_cold,
            "Without --pre-warm-nccl, HCCL warmup message must NOT appear",
        )

        p99_warm = res_warm["p99_ttft_ms"]
        p99_cold = res_cold["p99_ttft_ms"]
        logger.info(
            f"\nP99 TTFT comparison:\n"
            f"  With --pre-warm-nccl:  {p99_warm:.1f} ms\n"
            f"  Without:              {p99_cold:.1f} ms\n"
        )
        self.assertLessEqual(
            p99_warm,
            p99_cold,
            f"--pre-warm-nccl should reduce P99 TTFT "
            f"({p99_warm:.1f} ms vs {p99_cold:.1f} ms)",
        )


if __name__ == "__main__":
    unittest.main()
