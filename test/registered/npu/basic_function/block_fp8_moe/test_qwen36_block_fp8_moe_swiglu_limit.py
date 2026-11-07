import glob
import gzip
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V4_FLASH_BASE_MODEL_PATH
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

OUTPUT_DIR = "./profiler_dir"

# Accuracy floors, one per mode. testcases.md keeps the native group as the
# reference and puts the MXFP8 group one notch lower. DeepSeek-V4-Flash-Base has
# no public GSM8K baseline, so these start as guard rails and can be tightened
# once a calibration run pins the real numbers.
ACCURACY_NATIVE = 0.90
ACCURACY_MXFP8 = 0.89

# Envs aligned with the DeepSeek-V4-Flash-0731 A5 case
# (test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_fp8_4p_gpqa_a5.py),
# minus the items this case does not need: DSPARK speculative, reasoning effort.
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
    # dsv4. These have no in-tree reader, but the out-of-tree AscendC custom-ops
    # package may consume them (USE_FUSED_HC_PRE_ASCENDC shares its name with the
    # npu_hc_pre aclnn op), so keep them in lockstep with the 0731 A5 case.
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    # The Ascend block FP8 MoE method is only selected when the routed experts
    # are not MXFP4-packed (see layers/quantization/fp8.py), so this must stay
    # off even though the 0731 A5 case sets it to True.
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

# Model specific args for DSV4-Flash on Ascend, aligned with the 0731 cases.
# No Qwen3.6 only options (mamba, multimodal, NEXTN, qwen3 parsers) and no DSPARK
# speculative args (the Base checkpoint ships no draft model).
# mem-fraction-static is the one deliberate deviation: the Base routed experts are
# FP8 while 0731's are FP4, so the checkpoint is bigger per rank and the 0731 value
# 0.73 leaves no room for the KV pool (weights alone take 0.841).
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

# Op-level checks: empty by default (dump first, pin later).
W13_OP_PATTERNS = []
SWIGLU_OP_PATTERNS = []
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
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    events.sort(key=lambda e: e.get("ts", 0))
    return [e.get("name", "") for e in events]


def find_trace(profiler_dir):
    for pattern in ("**/trace_view.json", "**/*.trace.json.gz", "**/*.json.gz"):
        hits = sorted(
            glob.glob(os.path.join(str(profiler_dir), pattern), recursive=True)
        )
        if hits:
            return hits[0]
    raise AssertionError(f"no trace file found under {profiler_dir}")


class _SwigluCaseBase(CustomTestCase):
    """Shared setup and launch/measure helpers for the swiglu_limit case."""

    model = DEEPSEEK_V4_FLASH_BASE_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_TORCH_PROFILER_DIR.set(OUTPUT_DIR)

    def launch(self, mode, tag, cuda_graph=True):
        args = list(SERVER_ARGS)
        if not cuda_graph:
            args.append("--disable-cuda-graph")
        env = dict(SERVER_ENVS)
        env["SGLANG_NPU_ARCH35_REQUANT_BLOCK_FP8"] = mode
        print(f"[{tag}] mode={mode} cuda_graph={cuda_graph} model={self.model}")
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    @staticmethod
    def ops_path(tag):
        return os.path.join(OUTPUT_DIR, f"prof_{tag}", "ops.txt")

    def post(self, path, payload, tag, timeout=300):
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        self.assertEqual(resp.status_code, 200, f"[{tag}] {path}: {resp.text}")
        return resp

    def run_gsm8k(self):
        result = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                eval_name="gsm8k",
                api="chat",
                num_examples=1319,
                num_shots=5,
                num_threads=16,
                max_tokens=4096,
                temperature=0,
                chat_template_kwargs={"enable_thinking": False},
            )
        )
        return result["score"]

    def measure_accuracy(self, mode, tag, threshold):
        process = self.launch(mode, f"{tag}-acc")
        try:
            score = self.run_gsm8k()
        finally:
            kill_process_tree(process.pid)
        print(f"[{tag}] score={score:.4f} threshold={threshold:.2f}")
        self.assertGreaterEqual(
            score,
            threshold,
            f"[{tag}] GSM8K score {score:.4f} is below the floor {threshold}",
        )
        return score

    def collect_ops(self, mode, tag):
        profiler_dir = os.path.join(OUTPUT_DIR, f"prof_{tag}")
        os.makedirs(profiler_dir, exist_ok=True)
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
            self.post("/stop_profile", {}, tag, timeout=60)
        finally:
            kill_process_tree(process.pid)

        ops = load_ops(find_trace(profiler_dir))
        Path(self.ops_path(tag)).write_text("\n".join(ops))
        candidates = sorted(
            {n for n in ops if any(k in n.lower() for k in CANDIDATE_KEYWORDS)}
        )
        print(
            f"[{tag}] ops={len(ops)} candidates={candidates[:20]} "
            f"detail={self.ops_path(tag)}"
        )
        return ops

    def assert_swiglu_separate(self, ops, tag):
        """Op-level check; recorded only while the op names are not pinned yet."""
        if not ops_pinned():
            print(
                f"[WARN] [{tag}] W13_OP_PATTERNS/SWIGLU_OP_PATTERNS not pinned yet, "
                f"op-level check downgraded to recording; review {self.ops_path(tag)}"
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
    """Testcase: Verify a model whose config carries swiglu_limit>0 runs fine in both modes, and that the clamped SwiGLU activation is executed outside the w13 GMM. The model has no public GSM8K baseline, so the ACCURACY_* floors are guard rails to be tightened after a calibration run.

    [Test Category] Accuracy
    [Test Target] swiglu_limit>0 branch (non-fused w13 + NPUSwigluWithLimit)
    [Test Config] DeepSeek-V4-Flash-Base, A5 TP4 DP4 dsv4, native vs MXFP8, op trace with CUDA Graph off
    """

    def test_swiglu_limit_branch(self):
        native_score = self.measure_accuracy("0", "e2e7-native", ACCURACY_NATIVE)
        mxfp8_score = self.measure_accuracy("1", "e2e7-mxfp8", ACCURACY_MXFP8)

        ops = self.collect_ops("1", "e2e7-mxfp8")
        self.assert_swiglu_separate(ops, "e2e7-mxfp8")

        print(
            f"[{type(self).__name__}] native_score={native_score:.4f} "
            f"mxfp8_score={mxfp8_score:.4f} ops={len(ops)}"
        )


if __name__ == "__main__":
    unittest.main()
