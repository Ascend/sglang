import os
import threading
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    logger,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(
    est_time=1800,
    suite="",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

KIMI_K2_6_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "HCCL_BUFFSIZE": "1200",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

KIMI_K2_6_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.865,
    "--max-running-requests",
    80,
    "--chunked-prefill-size",
    32768,
    "--context-length",
    6144,
    "--max-prefill-tokens",
    65536,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    4,
    5,
    "--disable-radix-cache",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--speculative-draft-model-quantization",
    "unquant",
    "--prefill-delayer-max-delay-passes",
    200,
    "--enable-prefill-delayer",
    "--reasoning-parser",
    "kimi_k2",
    "--tool-call-parser",
    "kimi_k2",
]

cmd = "npu-smi info"


class TestKimiK25W4A8(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = KIMI_K2_6_W4A8_MODEL_PATH
    other_args = KIMI_K2_6_OTHER_ARGS
    envs = KIMI_K2_6_ENVS
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
            timeout=3000,
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
        time.sleep(30)
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
