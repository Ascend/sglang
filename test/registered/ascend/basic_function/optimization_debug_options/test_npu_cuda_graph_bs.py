import os
import re
import subprocess
import tempfile
import unittest

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)

MODEL = QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
_LAUNCH_TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

_DECODE_RE = re.compile(r"Capture target decode.*begin.*bs=\[([^\]]+)\]")
_PREFILL_RE = re.compile(r"Capture target prefill.*begin.*num_tokens=\[([^\]]+)\]")


def _read_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_cg_capture(log_text: str):
    """Return (decode_bs, prefill_bs) from CG capture log lines.

    Lines are distinguished by the ``target decode`` / ``target prefill``
    begin markers — no fallback ordering heuristics.
    """
    decode_bs = None
    prefill_bs = None
    for line in log_text.splitlines():
        if m := _DECODE_RE.search(line):
            decode_bs = [int(x.strip()) for x in m.group(1).split(",")]
            print(f"[CG parse] decode  | {line.strip()}")
        if m := _PREFILL_RE.search(line):
            prefill_bs = [int(x.strip()) for x in m.group(1).split(",")]
            print(f"[CG parse] prefill | {line.strip()}")
    if decode_bs is None and prefill_bs is None:
        print("[CG parse] WARNING: no matching CG begin line found in log")
    return decode_bs, prefill_bs


def _run_bench(base_url):
    bench_args = get_benchmark_args(
        base_url=base_url,
        backend="sglang",
        dataset_name="random",
        tokenizer=MODEL,
        num_prompts=10,
        random_input_len=256,
        random_output_len=32,
        request_rate=float("inf"),
    )
    bench_args.warmup_requests = 0
    return run_benchmark(bench_args)


def _launch_server(*, extra_args=None):
    """Launch a non-PD single server, capturing stderr to a temp file.

    Returns (process, stderr_file_path).
    """
    url = DEFAULT_URL_FOR_TEST
    _, host, port = url.split(":")
    host = host[2:]
    err_fd, err_path = tempfile.mkstemp(suffix=".log", prefix="cg_bs_")
    os.close(err_fd)

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        host,
        "--port",
        port,
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "1",
        *(extra_args or []),
    ]
    with open(err_path, "w") as err_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=err_file,
            text=True,
        )
    wait_for_http_ready(url + "/health", timeout=_LAUNCH_TIMEOUT, process=proc)
    return proc, err_path


class TestCudaGraphBs(CustomTestCase):
    """Testcase: verify per-phase CUDA-graph BS parameters in non-PD mode.

    [Test Category] Parameter
    [Test Target] --cuda-graph-max-bs-decode; --cuda-graph-bs-decode;
                  --cuda-graph-max-bs-prefill; --cuda-graph-bs-prefill
    """

    # ---- max_bs only, bs auto-generated for both phases ----
    def test_max_bs_auto_generates_bs(self):
        proc, err_path = _launch_server(
            extra_args=[
                "--cuda-graph-max-bs-decode",
                "8",
                "--cuda-graph-backend-prefill",
                "tc_piecewise",
                "--cuda-graph-max-bs-prefill",
                "256",
            ],
        )
        try:
            log_text = _read_log(err_path)
        finally:
            kill_process_tree(proc.pid)
            try:
                os.remove(err_path)
            except OSError:
                pass

        decode_bs, prefill_bs = _parse_cg_capture(log_text)
        self.assertIsNotNone(decode_bs, "Expected decode CG capture in log")
        self.assertEqual(max(decode_bs), 8)
        self.assertTrue(all(b <= 8 for b in decode_bs))

        self.assertIsNotNone(prefill_bs, "Expected prefill CG capture in log")
        self.assertEqual(max(prefill_bs), 256)
        self.assertTrue(all(b <= 256 for b in prefill_bs))

    # ---- explicit bs for both phases ----
    def test_explicit_bs_used_exactly(self):
        proc, err_path = _launch_server(
            extra_args=[
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "4",
                "8",
                "--cuda-graph-backend-prefill",
                "tc_piecewise",
                "--cuda-graph-bs-prefill",
                "64",
                "128",
                "256",
            ],
        )
        try:
            log_text = _read_log(err_path)
        finally:
            kill_process_tree(proc.pid)
            try:
                os.remove(err_path)
            except OSError:
                pass

        decode_bs, prefill_bs = _parse_cg_capture(log_text)
        self.assertEqual(decode_bs, [1, 2, 4, 8])
        self.assertEqual(prefill_bs, [64, 128, 256])

    # ---- decode-only: explicit bs overrides max_bs ----
    def test_decode_max_bs_overwritten_when_bs_set(self):
        proc, err_path = _launch_server(
            extra_args=[
                "--cuda-graph-max-bs-decode",
                "4",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "8",
            ],
        )
        try:
            log_text = _read_log(err_path)
        finally:
            kill_process_tree(proc.pid)
            try:
                os.remove(err_path)
            except OSError:
                pass

        decode_bs, _ = _parse_cg_capture(log_text)
        self.assertEqual(decode_bs, [1, 2, 8])
        self.assertEqual(max(decode_bs), 8, "max_bs should be 8 (overwritten), not 4")

    # ---- decode-only: disable padding produces sequential bs ----
    def test_decode_disable_padding_sequential_bs(self):
        proc, err_path = _launch_server(
            extra_args=[
                "--cuda-graph-max-bs-decode",
                "8",
                "--disable-cuda-graph-padding",
            ],
        )
        try:
            log_text = _read_log(err_path)
        finally:
            kill_process_tree(proc.pid)
            try:
                os.remove(err_path)
            except OSError:
                pass

        decode_bs, _ = _parse_cg_capture(log_text)
        self.assertEqual(decode_bs, list(range(1, 9)))

    # ---- TTFT comparison with different max_bs values ----
    def test_max_bs_ttft_comparison(self):
        proc1, err1 = _launch_server(
            extra_args=["--cuda-graph-max-bs-decode", "1"],
        )
        try:
            r1 = _run_bench(DEFAULT_URL_FOR_TEST)
            self.assertEqual(r1["completed"], 10)
        finally:
            kill_process_tree(proc1.pid)
            try:
                os.remove(err1)
            except OSError:
                pass

        proc8, err8 = _launch_server(
            extra_args=["--cuda-graph-max-bs-decode", "8"],
        )
        try:
            r8 = _run_bench(DEFAULT_URL_FOR_TEST)
            self.assertEqual(r8["completed"], 10)
        finally:
            kill_process_tree(proc8.pid)
            try:
                os.remove(err8)
            except OSError:
                pass

        t1, t8 = r1["mean_ttft_ms"], r8["mean_ttft_ms"]
        p1, p8 = r1["p99_ttft_ms"], r8["p99_ttft_ms"]
        print(
            f"\n=== TTFT comparison: max_bs=1 vs max_bs=8 ===\n"
            f"  Mean TTFT: {t1:.1f} ms (max_bs=1) vs {t8:.1f} ms (max_bs=8)\n"
            f"  P99  TTFT: {p1:.1f} ms (max_bs=1) vs {p8:.1f} ms (max_bs=8)"
        )
        # Larger max_bs enables request batching, reducing average TTFT.
        self.assertGreater(t1, t8)


if __name__ == "__main__":
    unittest.main()
