import io
import os
import re
import subprocess
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
parsed = urlparse(BASE_URL)
HOST = parsed.hostname
PORT = parsed.port


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
    3. HCCL port conflicts → HCCL_HOST/NPU_SOCKET_PORT_RANGE=auto
    """

    model = QWEN3_0_6B_WEIGHTS_PATH

    def _launch_server(self, pre_warm: bool, capture_logs: bool):
        stdout = io.StringIO() if capture_logs else None
        stderr = io.StringIO() if capture_logs else None

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

        env = {
            **os.environ,
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
            "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        }

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if capture_logs else None,
            stderr=subprocess.PIPE if capture_logs else None,
            text=True,
            env=env,
            start_new_session=True,
        )

        if capture_logs:

            def _pipe(src, dst):
                for line in src:
                    dst.write(line)
                    dst.flush()
                src.close()

            threading.Thread(
                target=_pipe, args=(proc.stdout, stdout), daemon=True
            ).start()
            threading.Thread(
                target=_pipe, args=(proc.stderr, stderr), daemon=True
            ).start()

        deadline = time.perf_counter() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early rc={proc.returncode}")
            try:
                if requests.get(BASE_URL + "/health", timeout=5).status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("server failed to become healthy")
        return proc, stdout, stderr

    def _teardown_stack(self, proc):
        t0 = time.perf_counter()
        if proc is not None:
            try:
                kill_process_tree(proc.pid, wait_timeout=60)
            except Exception as e:
                logger.info(f"Error killing {proc.pid}: {e}")
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        t_kill = time.perf_counter()
        wait_port_available(PORT, "server", timeout_s=120)
        t_port = time.perf_counter()
        drain_deadline = time.perf_counter() + 120
        while time.perf_counter() < drain_deadline:
            alive = []
            for p in psutil.process_iter(["pid", "name"]):
                try:
                    n = p.info["name"] or ""
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if n.startswith("sglang"):
                    alive.append((p.info["pid"], n))
            if not alive:
                break
            time.sleep(2)
        else:
            logger.info(f"Warning: sglang procs still alive after 120s: {alive}")
        t_drain = time.perf_counter()
        time.sleep(8)
        t_sleep = time.perf_counter()
        logger.info(
            f"  teardown: kill={t_kill - t0:.1f}s port={t_port - t_kill:.1f}s "
            f"drain={t_drain - t_port:.1f}s sleep15={t_sleep - t_drain:.1f}s "
            f"total={t_sleep - t0:.1f}s"
        )

    def _run_bench(self):
        args = get_benchmark_args(
            base_url=BASE_URL,
            backend="sglang",
            dataset_name="random",
            tokenizer=self.model,
            num_prompts=20,
            random_input_len=128,
            random_output_len=32,
            request_rate=float("inf"),
        )
        args.warmup_requests = 0
        args.model = self.model
        res = run_benchmark(args)
        self.assertGreater(res["mean_ttft_ms"], 0, "TTFT must be > 0 ms")
        return res

    def test_pre_warm_nccl_colocated_cold_start(self):
        t_total = time.perf_counter()

        def _phase(label, fn):
            t = time.perf_counter()
            result = fn()
            logger.info(f"  [{label}] {time.perf_counter() - t:.1f}s")
            return result

        # 1. Throwaway stack: warm the on-disk kernel cache for the exact
        #    benchmark shapes so the measured stacks below do not pay
        #    compilation asymmetrically.
        def _stack1():
            proc, _, _ = self._launch_server(pre_warm=False, capture_logs=False)
            try:
                self._run_bench()
            finally:
                self._teardown_stack(proc)

        _phase("throwaway", _stack1)

        # 2. Stack A: pre-warms the TP HCCL communicator at bootstrap.
        def _stack2():
            proc, dout, derr = self._launch_server(pre_warm=True, capture_logs=True)
            try:
                res = self._run_bench()
                server_logs = (dout.getvalue() + derr.getvalue()) if dout else ""
                self.assertIn(
                    "NCCL/RCCL/HCCL warmup completed",
                    server_logs,
                    "--pre-warm-nccl must log the HCCL warmup completion line",
                )
                m = re.search(
                    r"NCCL/RCCL/HCCL warmup completed in ([0-9.]+)s", server_logs
                )
                if m:
                    logger.info(f"  HCCL warmup duration: {m.group(1)}s")
                return res
            finally:
                self._teardown_stack(proc)

        res_prewarm = _phase("prewarm", _stack2)

        # 3. Stack B: starts with a cold TP HCCL communicator.
        def _stack3():
            proc, dout, derr = self._launch_server(pre_warm=False, capture_logs=True)
            try:
                res = self._run_bench()
                server_logs = (dout.getvalue() + derr.getvalue()) if dout else ""
                self.assertNotIn(
                    "NCCL/RCCL/HCCL warmup completed",
                    server_logs,
                    "without --pre-warm-nccl the HCCL warmup line must be absent",
                )
                return res
            finally:
                self._teardown_stack(proc)

        res_no_prewarm = _phase("cold", _stack3)

        logger.info(f"  [total] {time.perf_counter() - t_total:.1f}s")

        p99_w = res_prewarm["p99_ttft_ms"]
        p99_nw = res_no_prewarm["p99_ttft_ms"]
        logger.info(
            f"\n=== TTFT Comparison: --pre-warm-nccl vs default ===\n"
            f"  P99 TTFT: {p99_w:.1f} ms (warmup) vs {p99_nw:.1f} ms (no-warmup)\n"
        )
        self.assertLessEqual(
            p99_w,
            p99_nw,
            f"Expected --pre-warm-nccl P99 TTFT ({p99_w:.1f} ms) <= "
            f"no-warmup ({p99_nw:.1f} ms). NCCL warmup should prime all-reduce "
            f"communication and reduce first-request tail latency.",
        )


if __name__ == "__main__":
    unittest.main()
