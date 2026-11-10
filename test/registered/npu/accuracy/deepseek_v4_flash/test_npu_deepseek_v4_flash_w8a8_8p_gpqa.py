import logging
import os
import subprocess
import threading
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

logger = logging.getLogger(__name__)

# Number of server launches used to observe how much startup time the model
# cache saves on each restart.
SERVER_LAUNCH_TIMES = 5

# Log line printed once server warmup has finished, i.e. the server is fully
# ready to serve requests.
WARMUP_DONE_MARKER = "fired up and ready to roll"

register_npu_ci(
    est_time=3600,
    suite="nightly-acc-16-npu-a3",
    nightly=True,
)

DEEPSEEK_V4_FLASH_W8A8_DSPARK_8P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    # skip gpu branch
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    # DSPARK
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    # mtp
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    # deepep
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    # war barrier
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    # ZBAL
    # "SGLANG_ZBAL_LOCAL_MEM_SIZE": "61000",
    # "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    # "SGLANG_ZBAL_BOOTSTRAP_URL": "tcp://127.0.0.1:14699",
    # "ZBAL_ENABLE_GRAPH": "1",
}

DEEPSEEK_V4_FLASH_W8A8_DSPARK_8P_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.68,
    "--prefill-max-requests",
    192,
    "--max-prefill-tokens",
    80000,
    "--chunked-prefill-size",
    131072,
    "--max-running-requests",
    192,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    7,
    "--speculative-dspark-block-size",
    6,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    6,
    8,
    10,
    "--disable-radix-cache",
]


DEEPSEEK_V4_FLASH_W8A8_GENERATION_CONFIG_HIGH = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}


class TestNPUDeepSeekV4FlashW8A88PGPQA(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for DeepSeek-V4-Flash W8A8 8p DSPARK GPQA."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_DSPARK_8P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_W8A8_DSPARK_8P_ENVS
    accuracy = 0.874
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = DEEPSEEK_V4_FLASH_W8A8_GENERATION_CONFIG_HIGH
    eval_batch_size = 128
    stream = True
    timeout = 6000
    seed = 1

    @classmethod
    def _launch_server_and_wait_warmup(cls, command, env):
        """Launch one server and block until its warmup-done marker is logged.

        The server output is streamed through a reader thread so the warmup
        marker can be detected while the process keeps running. The launched
        process is stored in ``cls.process`` for the caller to kill.

        Returns:
            Startup time in seconds from process spawn to warmup completion.
        """
        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        cls.process = process
        warmup_done = threading.Event()

        def _stream_output():
            for line in process.stdout:
                print(line, end="", flush=True)
                if WARMUP_DONE_MARKER in line:
                    warmup_done.set()

        threading.Thread(target=_stream_output, daemon=True).start()

        deadline = start + cls.server_timeout
        while not warmup_done.wait(timeout=5):
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited with code {process.returncode} before warmup"
                )
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"Server warmup did not finish within {cls.server_timeout}s"
                )
        return time.perf_counter() - start

    @classmethod
    def setUpClass(cls):
        """Launch server via `python3 -m sglang.launch_server` instead of `sglang serve`.

        The server is launched SERVER_LAUNCH_TIMES times so we can observe how
        much startup time the model cache saves on each restart. Each launch is
        timed until the warmup-done marker is logged, then the server is killed
        before the next one is launched.
        """
        cls._setup_per_case_output()
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        if cls.envs:
            env.update(cls.envs)

        _, host, port = cls.base_url.split(":")
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            *[str(x) for x in cls.other_args],
            "--host",
            host[2:],
            "--port",
            port,
        ]

        cls.launch_seconds = []
        try:
            for i in range(SERVER_LAUNCH_TIMES):
                elapsed = cls._launch_server_and_wait_warmup(command, env)
                cls.launch_seconds.append(elapsed)
                if i == 0:
                    logger.info(
                        "Server launch %d/%d reached warmup done in %.1fs "
                        "(cold cache baseline)",
                        i + 1,
                        SERVER_LAUNCH_TIMES,
                        elapsed,
                    )
                else:
                    logger.info(
                        "Server launch %d/%d reached warmup done in %.1fs "
                        "(saved %.1fs vs launch 1)",
                        i + 1,
                        SERVER_LAUNCH_TIMES,
                        elapsed,
                        cls.launch_seconds[0] - elapsed,
                    )
                # Kill the fully-ready server, then launch the next one.
                kill_process_tree(cls.process.pid)
                cls.process.wait()
                cls.process = None
        finally:
            # unittest does not call tearDownClass when setUpClass fails, so make
            # sure a failed/timeout launch never leaves a server holding the
            # port and NPU devices for the next runs.
            if cls.process is not None:
                kill_process_tree(cls.process.pid, wait_timeout=None)
                cls.process = None

    def test_npu_deepseek_v4_flash_w8a8_8p_server_launch_time(self):
        """Report the launch time of each of the SERVER_LAUNCH_TIMES starts."""
        launches = self.__class__.launch_seconds
        self.assertEqual(len(launches), SERVER_LAUNCH_TIMES)
        for i, elapsed in enumerate(launches, start=1):
            logger.info(
                "Launch %d/%d reached warmup done: %.1fs (saved %.1fs vs launch 1)",
                i,
                SERVER_LAUNCH_TIMES,
                elapsed,
                launches[0] - elapsed,
            )

    # def test_npu_deepseek_v4_flash_w8a8_8p_gpqa(self):
    #     """Run NPU accuracy test for DeepSeek-V4-Flash W8A8 8p DSPARK GPQA."""
    #     self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
