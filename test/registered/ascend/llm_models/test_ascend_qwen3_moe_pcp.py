"""Ascend/NPU test for Qwen3 MoE Prefill Context Parallel (PCP).

PCP splits the input sequence across CP workers during prefill.
It is only active in disaggregated-prefill mode
(forward_mode.is_context_parallel_extend() must be True).

GPU layout (8 NPUs total):
  Prefill server: TP=4, ATTN_CP=2 — NPUs 0–3
                  (2 CP workers, each with 2 attention-TP NPUs)
  Decode  server: TP=4             — NPUs 4–7 (--base-gpu-id 4)
"""

import os
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_with_error_check,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=900, suite="nightly-8-npu-a3", nightly=True)

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

PREFILL_TP = 4
PREFILL_CP = 2
DECODE_TP = 4
DECODE_BASE_GPU_ID = PREFILL_TP  # NPUs 4–7

GSM8K_MIN_ACCURACY = 0.88

# Ascend-specific environment variables required for stable multi-NPU inference.
ASCEND_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "200",
    "HCCL_EXEC_TIMEOUT": "200",
    "STREAMS_PER_DEVICE": "32",
    "USE_VLLM_CUSTOM_ALLREDUCE": "1",
    "SGLANG_ENBLE_TORCH_COMILE": "1",
    "AUTO_USE_UC_MEMORY": "0",
    "P2P_HCCL_BUFFSIZE": "20",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
}

# Common server arguments for Ascend backend.
ASCEND_COMMON_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    "0.8",
    "--disaggregation-transfer-backend",
    "ascend",
]


class TestAscendQwen3MoePCP(CustomTestCase):
    """End-to-end test for Qwen3 MoE PCP on Ascend NPU (PD disaggregated).

    Validates:
    1. Prefill server starts with --attn-cp-size=2 on 4 NPUs.
    2. Decode  server starts on the remaining 4 NPUs.
    3. End-to-end GSM8K accuracy meets the expected threshold.
    """

    @classmethod
    def _apply_ascend_envs(cls):
        os.environ.update(ASCEND_ENVS)

    @classmethod
    def setUpClass(cls):
        cls._apply_ascend_envs()
        env = os.environ.copy()

        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        base_port = parsed.port
        cls.prefill_port = str(base_port + 100)
        cls.decode_port = str(base_port + 200)
        cls.lb_port = str(base_port)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.process_lb = cls.process_prefill = cls.process_decode = None

        MODELPATH = "LOCAL_PATH" #modify to actual model path when running the test
        cls.model = MODELPATH

        # Start prefill and decode servers in parallel, then block until both ready.
        cls.start_prefill(env)
        cls.start_decode(env)

        wait_for_http_ready(
            url=cls.prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_prefill,
        )
        wait_for_http_ready(
            url=cls.decode_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_decode,
        )

        cls.launch_lb()

    @classmethod
    def start_prefill(cls, env):
        prefill_args = ASCEND_COMMON_ARGS + [
            "--disaggregation-mode",
            "prefill",
            "--tp",
            str(PREFILL_TP),
            "--attn-cp-size",
            str(PREFILL_CP),
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "-1",
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls, env):
        decode_args = ASCEND_COMMON_ARGS + [
            "--disaggregation-mode",
            "decode",
            "--tp",
            str(DECODE_TP),
            "--base-gpu-id",
            str(DECODE_BASE_GPU_ID),
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        cls.process_lb = popen_with_error_check(lb_command)
        wait_for_http_ready(
            url=cls.lb_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_lb,
        )

    @classmethod
    def tearDownClass(cls):
        for proc in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if proc:
                try:
                    kill_process_tree(proc.pid)
                except Exception as e:
                    print(f"Error killing process {proc.pid}: {e}")
        time.sleep(5)

    def test_gsm8k_accuracy(self):
        """GSM8K accuracy validates PCP correctness end-to-end."""
        GSM8K_PATH = "LOCAL_PATH" #modify to actual GSM8K path when running the test
        args = SimpleNamespace(
            num_shots=5,
            data_path=GSM8K_PATH if os.path.exists(GSM8K_PATH) else None,
            num_questions=200,
            max_new_tokens=512,
            parallel=8,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(
            f"GSM8K accuracy (PCP TP={PREFILL_TP} CP={PREFILL_CP}): "
            f"{metrics['accuracy']:.3f}"
        )
        self.assertGreater(
            metrics["accuracy"],
            GSM8K_MIN_ACCURACY,
            f"Expected >{GSM8K_MIN_ACCURACY}, got {metrics['accuracy']:.3f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
