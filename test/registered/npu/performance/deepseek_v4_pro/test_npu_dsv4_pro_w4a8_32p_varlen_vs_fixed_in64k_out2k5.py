import csv
import logging
import os
import subprocess
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME, check_role
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=5400,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

logger = logging.getLogger(__name__)

MODEL_PATH_CANDIDATES = [
    "/root/.cache/modelscope/hub/models/Eco-Tech/DeepSeek-V4-Pro-0813-w4a8",
    "/home/weights/DeepSeek-V4-Pro-0813-w4a8",
]


def _resolve_model_path():
    for path in MODEL_PATH_CANDIDATES:
        if os.path.isdir(path):
            return path
    return MODEL_PATH_CANDIDATES[0]


DEEPSEEK_V4_PRO_W4A8_MODEL_PATH = _resolve_model_path()

DSV4_PRO_TWO_NODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "DEEPEP_HCCL_BUFFSIZE": "2000",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "300",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
}

DSV4_PRO_TWO_NODE_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--trust-remote-code",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    200,
    "--mem-fraction-static",
    0.72,
    "--quantization",
    "modelslim",
    "--chunked-prefill-size",
    32768,
    "--kv-cache-dtype",
    "auto",
    "--moe-dense-tp-size",
    1,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    6,
    8,
    10,
    16,
    "--load-balance-method",
    "round_robin",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-metrics",
    "--disable-radix-cache",
    "--served-model-name",
    "dsv4",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_PRO_W4A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    7,
    "--speculative-dspark-block-size",
    6,
]

DSV4_PRO_TWO_NODE_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_PRO_W4A8_MODEL_PATH,
    "other_args": DSV4_PRO_TWO_NODE_OTHER_ARGS,
    "node_envs": DSV4_PRO_TWO_NODE_ENVS,
}

TOOLKIT_REPO = "https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git"
TOOLKIT_DIR = "/tmp/aisbench_auto_tools_prefix"
VENV_DIR = os.path.expanduser("~/test_env_aisbench_prefix")
AISBENCH_SOURCE_PATH = "/root/.cache/.cache/benchmark"
AISBENCH_PKG_PATH = "/root/.cache/.cache/aisbench-packages"
PIP_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
RESULT_CSV = os.path.join(TOOLKIT_DIR, "aisbench_result.csv")


def _run(cmd, cwd=None, env=None):
    logger.info(f"Run: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env, executable="/bin/bash")


class TestNPUDSV4ProW4A8_32P_VarlenVsFixed_In64k_Out2k5(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """DSV4-Pro w4a8 2-node PD-mix: variable-length (40k~80k, mean 64k) vs
    fixed-length (64k) input, 2.5k output. Pass criteria: TTFT avg <= 10s,
    TPOT avg <= 30ms, per-card output throughput degradation < 15%."""

    model_config = DSV4_PRO_TWO_NODE_MODEL_CONFIG
    max_attempts = 1

    input_len = 64000
    output_len = 2500
    data_num = 80
    concurrency = 20
    repeat_rate = 0.5
    prefix_num = 20
    seed = 42
    dp = 4
    npu_num = 32
    length_mean = 64000
    length_std = 8000
    length_min = 40000
    length_max = 80000
    ttft_threshold_ms = 10000
    tpot_threshold_ms = 30
    degradation_threshold = 0.15

    @classmethod
    def _setup_prefix_tool(cls):
        if not os.path.isdir(TOOLKIT_DIR):
            _run(f"git clone --depth 1 {TOOLKIT_REPO} {TOOLKIT_DIR}")
        if not os.path.isfile(os.path.join(VENV_DIR, "bin", "ais_bench")):
            _run(f"python3 -m venv --system-site-packages {VENV_DIR}")
            if os.path.isdir(AISBENCH_PKG_PATH) and os.path.isdir(AISBENCH_SOURCE_PATH):
                _run(
                    f"{VENV_DIR}/bin/pip install -U pip --no-index --find-links={AISBENCH_PKG_PATH}"
                )
                _run(
                    f"{VENV_DIR}/bin/pip install -e {AISBENCH_SOURCE_PATH} --use-pep517"
                    f" --no-index --find-links={AISBENCH_PKG_PATH}"
                )
                _run(
                    f"{VENV_DIR}/bin/pip install -r {AISBENCH_SOURCE_PATH}/requirements/api.txt"
                    f" --no-index --find-links={AISBENCH_PKG_PATH}"
                )
                _run(
                    f"{VENV_DIR}/bin/pip install -r {AISBENCH_SOURCE_PATH}/requirements/extra.txt"
                    f" --no-index --find-links={AISBENCH_PKG_PATH}"
                )
            else:
                source_path = AISBENCH_SOURCE_PATH
                if not os.path.isdir(source_path):
                    _run("git clone https://github.com/AISBench/benchmark.git /tmp/benchmark")
                    source_path = "/tmp/benchmark"
                _run(f"{VENV_DIR}/bin/pip install -U pip -i {PIP_MIRROR}")
                _run(
                    f"{VENV_DIR}/bin/pip install -e {source_path} --use-pep517 -i {PIP_MIRROR}"
                )
                _run(
                    f"{VENV_DIR}/bin/pip install -r {source_path}/requirements/api.txt -i {PIP_MIRROR}"
                )
                _run(
                    f"{VENV_DIR}/bin/pip install -r {source_path}/requirements/extra.txt -i {PIP_MIRROR}"
                )
        work_path = subprocess.check_output(
            f"{VENV_DIR}/bin/pip show ais_bench_benchmark | grep Location | awk '{{print $2}}'",
            shell=True,
            executable="/bin/bash",
        ).decode().strip()
        dataset_path = os.path.join(TOOLKIT_DIR, "datasets")
        os.makedirs(dataset_path, exist_ok=True)
        config_content = f'''DATASET_PATH = "{dataset_path}"
WORK_PATH = "{work_path}"
MODEL_NAME = "dsv4"
MODEL_PATH = "{DSV4_PRO_TWO_NODE_MODEL_CONFIG["model_path"]}"
HOST_IP = "{cls.host}"
HOST_PORT = "{cls.port}"
URL = ""
API_KEY = ""
DEFAULT_PERFORMANCE_TEST = "default_perf"
OUTPUT_DIR = "./outputs/default"
POD_INFO = []
'''
        with open(os.path.join(TOOLKIT_DIR, "config.py"), "w") as f:
            f.write(config_content)
        logger.info(f"prefix tool config.py:\n{config_content}")

    def _run_once(self, extra_args=""):
        cmd = (
            f"{VENV_DIR}/bin/python aisbench_test.py"
            f" --dataset_type prefix_cache --input_len {self.input_len}"
            f" --output_len {self.output_len} --data_num {self.data_num}"
            f" --concurrency {self.concurrency} --request_rate 0"
            f" --repeat_rate {self.repeat_rate} --prefix_num {self.prefix_num}"
            f" --seed {self.seed} --prefix_test --dp {self.dp}"
            f" --npu_num {self.npu_num} {extra_args}"
        )
        env = os.environ.copy()
        env["PATH"] = os.path.join(VENV_DIR, "bin") + os.pathsep + env["PATH"]
        _run(cmd, cwd=TOOLKIT_DIR, env=env)
        with open(RESULT_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        return {k: v for k, v in rows[-1].items()}

    @staticmethod
    def _metrics(row):
        return {
            "ttft_avg": float(row["TTFT avg"]),
            "ttft_p90": float(row["TTFT P90"]),
            "tpot_avg": float(row["TPOT avg"]),
            "tpot_p90": float(row["TPOT SLO_P90"]),
            "output_throughput": float(row["output_throughput"]),
            "single_output_throughput": float(row["single_output_throughput"]),
        }

    @check_role(allowed_roles=["master"])
    def test_npu_dsv4_pro_w4a8_32p_varlen_vs_fixed_in64k_out2k5(self):
        self._setup_prefix_tool()

        fixed = self._metrics(self._run_once())
        logger.info(f"fixed-length(64k) metrics: {fixed}")

        varlen = self._metrics(
            self._run_once(
                f"--length_mean {self.length_mean} --length_std {self.length_std}"
                f" --length_min {self.length_min} --length_max {self.length_max}"
            )
        )
        logger.info(f"variable-length(40k~80k, mean 64k) metrics: {varlen}")

        degradation = (
            fixed["single_output_throughput"] - varlen["single_output_throughput"]
        ) / fixed["single_output_throughput"]
        logger.info(
            f"per-card output throughput degradation: {degradation:.2%} "
            f"(fixed={fixed['single_output_throughput']:.2f}, "
            f"varlen={varlen['single_output_throughput']:.2f} tokens/s/card)"
        )

        self.assertLessEqual(fixed["ttft_avg"], self.ttft_threshold_ms)
        self.assertLessEqual(varlen["ttft_avg"], self.ttft_threshold_ms)
        self.assertLessEqual(fixed["tpot_avg"], self.tpot_threshold_ms)
        self.assertLessEqual(varlen["tpot_avg"], self.tpot_threshold_ms)
        self.assertLess(degradation, self.degradation_threshold)


if __name__ == "__main__":
    unittest.main()
