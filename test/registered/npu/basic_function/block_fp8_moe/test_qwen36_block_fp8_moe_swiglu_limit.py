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
    DEEPSEEK_V4_FLASH_BASE_MODEL_PATH,
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

register_npu_ci(est_time=3600, suite="full-4-npu-a3", nightly=True)

# Per-script profiler dir: the final cleanup removes it, siblings share the cwd.
OUTPUT_DIR = "./profiler_dir_" + os.path.splitext(os.path.basename(__file__))[0]

# No public GSM8K baseline for this Base checkpoint; these floors are provisional.
ACCURACY_NATIVE = 0.90
ACCURACY_MXFP8 = 0.89

# AscendC custom ops: export these in the shell before launching the server.
#   export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.1.0/opp/vendors/customize/op_api/lib:/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
#   export ASCEND_CUSTOM_OPP_PATH="/usr/local/Ascend/cann-9.1.0/opp/vendors/customize:/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer"

# Envs aligned with the DeepSeek-V4-Flash-0731 A5 case.
SERVER_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "1",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    # HCCL deepep
    "DEEPEP_HCCL_BUFFSIZE": "2560",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    # dsv4: consumed by the out-of-tree AscendC custom-ops package.
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    # Keep off: the Ascend block FP8 MoE path needs non-MXFP4 experts.
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    # performance
    "ASCEND_LAUNCH_BLOCKING": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
}

# mem-fraction-static 0.90: FP8 (non-MXFP4) experts cost ~78 GB/rank at TP4, so the
# FP4-case 0.6x starves the KV pool; the configurator's bare minimum is 0.867.
SERVER_ARGS = [
    "--page-size",
    "128",
    "--tp-size",
    "4",
    "--trust-remote-code",
    "--attention-backend",
    "dsv4",
    "--device",
    "npu",
    "--watchdog-timeout",
    "9000",
    "--mem-fraction-static",
    "0.90",
    "--max-running-requests",
    "32",
    "--chunked-prefill-size",
    "131072",
    "--max-prefill-tokens",
    "131072",
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "4",
    "8",
    "--kv-cache-dtype",
    "auto",
    "--enable-dp-lm-head",
    "--disable-radix-cache",
    "--enable-dp-attention",
    "--dp-size",
    "4",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "fp8",
]

# w13 runs unfused here; the clamp is a separate op (swiglu_group_quant with
# clamp_value=limit). npu_clipped_swiglu belongs to the dense path, not the experts.
W13_OP_PATTERNS = ["npu::npu_grouped_matmul"]
SWIGLU_OP_PATTERNS = ["swiglu_group_quant"]
CANDIDATE_KEYWORDS = (
    "w13",
    "gate_up",
    "gmm",
    "grouped",
    "swiglu",
    "silu",
    "quant",
    "mxfp8",
    "pertoken",
    "clamp",
    "limit",
)

PROFILE_NUM_STEPS = 8
PROFILE_BURST_PROMPTS = 32
PROFILE_BURST_MAX_NEW_TOKENS = 32
# The trace export runs in its own process, so the server must stay up until it finishes.
PROFILE_EXPORT_TIMEOUT = 180
PROFILE_EXPORT_POLL_INTERVAL = 5


def ops_pinned():
    return bool(W13_OP_PATTERNS and SWIGLU_OP_PATTERNS)


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


class _SwigluCaseBase(CustomTestCase):
    """Shared setup and launch/measure helpers for the swiglu_limit case."""

    model = DEEPSEEK_V4_FLASH_BASE_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def launch(self, mode, tag, cuda_graph=True):
        args = list(SERVER_ARGS)
        if not cuda_graph:
            args.append("--disable-cuda-graph")
        env = dict(SERVER_ENVS)
        env["SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8"] = mode
        logger.info(f"[{tag}] mode={mode} cuda_graph={cuda_graph} model={self.model}")
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    def post(self, path, payload, tag, timeout=300):
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        self.assertEqual(resp.status_code, 200, f"[{tag}] {path}: {resp.text}")
        return resp

    def run_gsm8k(self):
        result = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                eval_name="gsm8k",
                api="completion",
                num_examples=200,
                num_shots=5,
                num_threads=16,
                max_tokens=512,
            )
        )
        return result["score"]

    def measure_accuracy(self, mode, tag, threshold):
        process = self.launch(mode, f"{tag}-acc")
        try:
            score = self.run_gsm8k()
        finally:
            kill_process_tree(process.pid)
        logger.info(f"[{tag}] score={score:.4f} threshold={threshold:.2f}")
        self.assertGreaterEqual(
            score,
            threshold,
            f"[{tag}] GSM8K score {score:.4f} is below the floor {threshold}",
        )
        return score

    def collect_ops(self, mode, tag):
        profiler_dir = os.path.join(OUTPUT_DIR, f"prof_{tag}")
        # Clean slate so a stale trace cannot be picked up; the dir is dropped below.
        shutil.rmtree(profiler_dir, ignore_errors=True)
        os.makedirs(profiler_dir, exist_ok=True)
        try:
            process = self.launch(mode, tag, cuda_graph=False)
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

    def assert_swiglu_separate(self, ops, tag):
        """Op-level check; recorded only while the op names are not pinned yet."""
        if not ops_pinned():
            logger.warning(
                f"[{tag}] W13_OP_PATTERNS/SWIGLU_OP_PATTERNS not pinned yet, "
                "op-level check downgraded to the printed candidates"
            )
            return

        self.assertTrue(ops, f"[{tag}] empty trace, the op order cannot be judged")
        w13_indices = [
            index
            for index, name in enumerate(ops)
            if any(p.lower() in name.lower() for p in W13_OP_PATTERNS)
        ]
        self.assertTrue(w13_indices, f"[{tag}] no w13-like op found in the trace")
        swiglu_indices = [
            index
            for index, name in enumerate(ops)
            if any(p.lower() in name.lower() for p in SWIGLU_OP_PATTERNS)
        ]
        self.assertTrue(
            swiglu_indices,
            f"[{tag}] no standalone SwiGLU activation op found, it looks fused into "
            "the GMM while swiglu_limit is set",
        )
        self.assertFalse(
            set(w13_indices) & set(swiglu_indices),
            f"[{tag}] the SwiGLU activation and w13 are the same op, the clamped "
            "branch is not in effect",
        )


class TestE2E7SwigluLimitBranch(_SwigluCaseBase):
    """Testcase: Verify a model whose config carries swiglu_limit>0 runs fine in both modes, and that the clamped SwiGLU activation is executed outside the w13 GMM.

    [Test Category] Accuracy
    [Test Target] swiglu_limit>0 branch (non-fused w13 + NPUSwigluWithLimit)
    [Test Config] DeepSeek-V4-Flash-Base, A5 TP4 DP4 dsv4, native vs MXFP8, op trace with CUDA Graph off
    """

    def test_swiglu_limit_branch(self):
        native_score = self.measure_accuracy("0", "e2e7-native", ACCURACY_NATIVE)
        mxfp8_score = self.measure_accuracy("1", "e2e7-mxfp8", ACCURACY_MXFP8)

        ops = self.collect_ops("1", "e2e7-mxfp8")
        self.assert_swiglu_separate(ops, "e2e7-mxfp8")

        logger.info(
            f"[{type(self).__name__}] native_score={native_score:.4f} "
            f"mxfp8_score={mxfp8_score:.4f} ops={len(ops)}"
        )


if __name__ == "__main__":
    unittest.main()
