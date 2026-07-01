import os
import threading
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_27B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    logger,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

QWEN3_6_27B_64K_1K_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
}

QWEN3_6_27B_64K_1K_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    48000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    6,
    "--max-mamba-cache-size",
    16,
    "--mem-fraction-static",
    0.58,
    "--cuda-graph-bs",
    1,
    2,
    4,
    5,
    6,
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
    "--skip-server-warmup",
    "--dp-size",
    2,
    "--enable-dp-attention",
]

cmd = "npu-smi info"


class TestNPUQwen3_6_27B_1P_In3k5_Out1k5_50ms(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    other_args = QWEN3_6_27B_64K_1K_OTHER_ARGS
    envs = QWEN3_6_27B_64K_1K_ENVS
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 65536
    output_len = 1024
    random_range_ratio = 1
    out = open(f"./out_log.txt", "w+", encoding="utf-8")
    err = open(f"./err_log.txt", "w+", encoding="utf-8")

    @classmethod
    def setUpClass(cls):
        raw_result = run_command(cmd)
        logger.info("S1、服务启动前执行npu-smi info")
        logger.info(raw_result)

        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        for key, value in env.items():
            logger.info(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                logger.info(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

        other_args = list(cls.other_args)

        logger.info("S2、启动服务")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
            return_stdout_stderr=(cls.out, cls.err),
        )

    def test_1(self):
        logger.info("S3、记录HCCL初始化完成后、模型加载前，通信域内存占用")
        res1 = run_command("cat ./err_log.txt | grep 'Init torch distributed ends'")
        logger.info(res1)
        logger.info("S4、记录模型加载后，模型权重内存占用")
        res2 = run_command("cat ./err_log.txt | grep 'Load weight end'")
        logger.info(res2)
        logger.info("S5、记录KV cache分配后，KV cache内存占用")
        res3 = run_command("cat ./err_log.txt | grep 'KV Cache is allocated'")
        logger.info(res3)
        logger.info("S6、记录NPU graph buffer分配后，NPU graph buffer内存占用")
        res4 = run_command("cat ./err_log.txt | grep 'Capture npu graph end'")
        logger.info(res4)
        logger.info("S7、服务启动成功后执行npu-smi info")
        raw_result = run_command(cmd)
        logger.info(raw_result)

    def test_2(self):
        logger.info("S9、curl一条64k长序列请求，同时持续监控HBM占用")

        stop_event = threading.Event()

        def monitor_npu():
            """线程2：持续采集并实时打印 npu-smi info"""
            while not stop_event.is_set():
                try:
                    raw_result = run_command(cmd)
                    logger.info("----- npu-smi info -----\n%s", raw_result)
                except Exception as e:
                    logger.warning(f"npu-smi monitor error: {e}")
                time.sleep(0.5)  # 采样间隔，可按需调小

        # 启动监控线程（daemon=True，防止主线程异常退出时挂死）
        monitor_thread = threading.Thread(target=monitor_npu, daemon=True)
        monitor_thread.start()

        # 线程1：执行请求
        self.run_throughput()

        # 请求完成，通知监控线程停止
        stop_event.set()
        monitor_thread.join(timeout=5)

        logger.info("S9、请求完成，最终 npu-smi info")
        raw_result = run_command(cmd)
        logger.info(raw_result)

    def test_3(self):
        if self.process:
            try:
                kill_process_tree(self.process.pid)
                for _ in range(60):
                    if self.process.poll() is not None:
                        logger.info("Process fully exited")
                        break
                    time.sleep(1)
                else:
                    logger.warning("Process did NOT exit in time")
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")
        logger.info("S9、停止服务，等待服务完全停止后，记录每张卡的HBM内存占用和总内")
        time.sleep(31)
        raw_result = run_command(cmd)
        logger.info(raw_result)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")


if __name__ == "__main__":
    unittest.main()
