import glob
import gzip
import json
import os
import shutil
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH,
    logger,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=3600, suite="full-2-npu-a3", nightly=True)

# Per-script profiler dir: the final cleanup removes it, siblings share the cwd.
OUTPUT_DIR = "./profiler_dir_" + os.path.splitext(os.path.basename(__file__))[0]
ACCURACY = 0.94
MXFP8_MODE = "1"

SERVER_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM": "1",
    "ASCEND_USE_FIA": "1",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "DEEPEP_HCCL_BUFFSIZE": "384",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
}

SERVER_ARGS = [
    "--tp-size",
    "2",
    "--nnodes",
    "1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    "128",
    "--max-total-tokens",
    "600000",
    "--max-prefill-tokens",
    "65536",
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests",
    "10",
    "--max-mamba-cache-size",
    "20",
    "--mem-fraction-static",
    "0.65",
    "--cuda-graph-bs-decode",
    "2",
    "4",
    "8",
    "12",
    "14",
    "16",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]

DEEPEP_ARGS = ["--moe-a2a-backend", "deepep", "--ep-size", "2"]
LOW_LATENCY_ARGS = DEEPEP_ARGS + ["--deepep-mode", "low_latency"]
NORMAL_ARGS = DEEPEP_ARGS + ["--deepep-mode", "normal"]
BF16_FALLBACK_ARGS = LOW_LATENCY_ARGS + ["--deepep-dispatcher-output-dtype", "bf16"]

# Anchor w13 on the matmul op, not its launching python frame (the activation quant
# runs inside it). QUANT_OP_PATTERNS stays MoE-specific; these ops also fire in dense layers.
W13_OP_PATTERNS = ["grouped_matmul_swiglu_quant", "npu::npu_grouped_matmul"]
QUANT_OP_PATTERNS = ["npu/moe/quant.py"]
CANDIDATE_KEYWORDS = ("w13", "gate_up", "gmm", "grouped", "quant", "mxfp8", "pertoken")

# Short decode burst, not GSM8K: the judgement only needs the op order around w13.
PROFILE_NUM_STEPS = 8
PROFILE_BURST_PROMPTS = 32
PROFILE_BURST_MAX_NEW_TOKENS = 32
# The trace export runs in its own process, so the server must stay up until it finishes.
PROFILE_EXPORT_TIMEOUT = 180
PROFILE_EXPORT_POLL_INTERVAL = 5


def ops_pinned():
    return bool(W13_OP_PATTERNS and QUANT_OP_PATTERNS)


def load_ops(trace_path):
    """Return the operator names of a chrome trace, ordered by start time."""
    if str(trace_path).endswith(".gz"):
        with gzip.open(trace_path, "rt") as f:
            trace = json.load(f)
    else:
        with open(trace_path) as f:
            trace = json.load(f)
    # Chrome traces are either a {"traceEvents": ...} object or a bare event array.
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    events = [e for e in events if e.get("ph") == "X"]
    events.sort(key=lambda e: e.get("ts", 0))
    return [e.get("name", "") for e in events]


def find_trace(profiler_dir):
    # Newest match wins; guards only against leftovers from a run that died before cleanup.
    for pattern in ("**/trace_view.json", "**/*.trace.json.gz", "**/*.json.gz"):
        hits = glob.glob(os.path.join(str(profiler_dir), pattern), recursive=True)
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def wait_for_trace(profiler_dir, timeout):
    """Wait for the exporter to finish writing the trace, then return its path.

    The export runs in a torch_npu process spawned after the profiler stops, so an
    early kill aborts it; the file is done once two polls see the same non-zero size.
    """
    deadline = time.time() + timeout
    last_size = None
    while True:
        if time.time() >= deadline:
            raise AssertionError(
                f"no complete trace file found under {profiler_dir} "
                f"within {timeout}s"
            )
        path = find_trace(profiler_dir)
        size = None
        if path is not None:
            try:
                size = os.path.getsize(path)
            except OSError:
                # The exporter can rename/remove a mid-poll candidate; treat as not-ready, not a crash.
                size = None
        if size and size == last_size:
            return path
        last_size = size
        time.sleep(PROFILE_EXPORT_POLL_INTERVAL)


def first_w13_index(ops):
    for index, name in enumerate(ops):
        if any(p.lower() in name.lower() for p in W13_OP_PATTERNS):
            return index
    return None


class _DeepEpCaseBase(CustomTestCase):
    """Launch, measure and op-check helpers shared by the two DeepEP cases."""

    model = QWEN3_6_35B_A3B_FP8_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def post(self, path, payload, tag, timeout=300):
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        self.assertEqual(resp.status_code, 200, f"[{tag}] {path}: {resp.text}")
        return resp

    def launch(self, other_args, tag, cuda_graph):
        """Start one server; CUDA Graph is switched off for the profiling launches."""
        args = list(SERVER_ARGS) + list(other_args)
        if not cuda_graph:
            args.append("--disable-cuda-graph")
        env = {**SERVER_ENVS, "SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8": MXFP8_MODE}
        logger.info(f"[{tag}] cuda_graph={cuda_graph} extra_args={other_args}")
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    def measure_accuracy(self, other_args, tag):
        """E1/E2: GSM8K (200 questions) on the documented public args (CUDA Graph on)."""
        process = self.launch(other_args, f"{tag}-acc", cuda_graph=True)
        try:
            result = run_eval(
                SimpleNamespace(
                    base_url=self.base_url,
                    eval_name="gsm8k",
                    api="chat",
                    num_examples=200,
                    num_shots=5,
                    num_threads=16,
                    max_tokens=4096,
                    temperature=0,
                    chat_template_kwargs={"enable_thinking": False},
                )
            )
        finally:
            kill_process_tree(process.pid)

        score = result["score"]
        logger.info(f"[{tag}] score={score:.4f} bar={ACCURACY}")
        self.assertGreaterEqual(score, ACCURACY, f"[{tag}] GSM8K score below {ACCURACY}")
        return score

    def collect_ops(self, other_args, tag):
        """E3: profile a short decode burst with CUDA Graph off, return the op names."""
        profiler_dir = os.path.join(OUTPUT_DIR, f"prof_{tag}")
        # Clean slate so a stale trace cannot be picked up; the dir is dropped below.
        shutil.rmtree(profiler_dir, ignore_errors=True)
        os.makedirs(profiler_dir, exist_ok=True)
        try:
            process = self.launch(other_args, tag, cuda_graph=False)
            try:
                self.post(
                    "/start_profile",
                    {"num_steps": PROFILE_NUM_STEPS, "output_dir": profiler_dir},
                    tag,
                    timeout=60,
                )
                for i in range(PROFILE_BURST_PROMPTS):
                    self.post(
                        "/generate",
                        {
                            "text": f"Question {i + 1}: The capital of France is",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": PROFILE_BURST_MAX_NEW_TOKENS,
                            },
                        },
                        tag,
                    )
                # The burst spans more forwards than num_steps, so the server auto-stops
                # the profiler; an explicit /stop_profile would 500. The flush runs in a
                # separate process, hence the wait before the kill.
                trace_path = wait_for_trace(profiler_dir, PROFILE_EXPORT_TIMEOUT)
            finally:
                kill_process_tree(process.pid)
            ops = load_ops(trace_path)
        finally:
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        candidates = {
            n for n in ops if any(k in n.lower() for k in CANDIDATE_KEYWORDS)
        }
        if ops_pinned():
            logger.info(f"[{tag}] ops={len(ops)} candidates={len(candidates)}")
        else:
            # While unpinned, list distinct candidate ops in trace order (ordering is what matters).
            logger.info(
                f"[{tag}] ops={len(ops)} candidates={len(candidates)}, in trace order:"
            )
            seen = set()
            for index, name in enumerate(ops):
                if name in candidates and name not in seen:
                    seen.add(name)
                    logger.info(f"[{tag}] #{index} {name}")
        return ops

    def check_quant_before_w13(self, ops, tag, expected):
        """E3: the activation MXFP8 quant op must (not) show up in front of w13."""
        if not ops_pinned():
            logger.warning(
                f"[{tag}] W13_OP_PATTERNS/QUANT_OP_PATTERNS not pinned yet, "
                "op-level check downgraded to the printed candidates"
            )
            return

        self.assertTrue(ops, f"[{tag}] empty trace, the op order cannot be judged")
        w13_index = first_w13_index(ops)
        self.assertIsNotNone(w13_index, f"[{tag}] no w13-like op found in the trace")

        hits = [
            (index, name)
            for index, name in enumerate(ops[:w13_index])
            if any(p.lower() in name.lower() for p in QUANT_OP_PATTERNS)
        ]
        if expected:
            self.assertTrue(
                hits,
                f"[{tag}] no activation MXFP8 quant op before {ops[w13_index]}, "
                "inner quantization fallback is not in effect",
            )
        else:
            detail = f" {hits[0][1]}" if hits else ""
            self.assertFalse(
                hits,
                f"[{tag}] activation quant op{detail} found before "
                f"{ops[w13_index]}, the activation is quantized twice (reuse broken)",
            )


class TestE2E3DeepEpLowLatencyReuse(_DeepEpCaseBase):
    """Testcase: Verify that with DeepEP low-latency dispatch the already-quantized activation is reused by the w13 GEMM, while the normal dispatch control group needs a standalone activation quantization op in front of it.

    [Test Category] Quant
    [Test Target] --deepep-mode low_latency vs normal
    [Test Config] TP2 EP2 MXFP8, DeepEP dispatch, op trace with CUDA Graph off
    """

    def test_low_latency_reuse(self):
        # E1/E2: both dispatch modes keep the accuracy bar.
        self.measure_accuracy(LOW_LATENCY_ARGS, "e2e3-low_latency")
        self.measure_accuracy(NORMAL_ARGS, "e2e3-normal")

        # E3: the control group re-quantizes the activation, the low-latency group reuses it.
        low_latency_ops = self.collect_ops(LOW_LATENCY_ARGS, "e2e3-low_latency")
        normal_ops = self.collect_ops(NORMAL_ARGS, "e2e3-normal")
        self.check_quant_before_w13(normal_ops, "e2e3-normal", expected=True)
        self.check_quant_before_w13(low_latency_ops, "e2e3-low_latency", expected=False)


class TestE2E4DeepEpLowLatencyBf16Fallback(_DeepEpCaseBase):
    """Testcase: Verify the DeepEP low-latency BF16 dispatch fallback neither raises RuntimeError nor loses accuracy, and that the activation MXFP8 quantization still happens inside the w13 path.

    [Test Category] Quant
    [Test Target] --deepep-dispatcher-output-dtype bf16 + low_latency
    [Test Config] TP2 EP2 MXFP8, DeepEP low_latency, op trace with CUDA Graph off
    """

    def test_bf16_dispatch_fallback(self):
        # E1/E2: no RuntimeError on the bf16 fallback path and the accuracy bar holds.
        self.measure_accuracy(BF16_FALLBACK_ARGS, "e2e4-bf16")

        # E3: with bf16 dispatch the activation is quantized inside the w13 path.
        ops = self.collect_ops(BF16_FALLBACK_ARGS, "e2e4-bf16")
        self.check_quant_before_w13(ops, "e2e4-bf16", expected=True)


if __name__ == "__main__":
    unittest.main()
