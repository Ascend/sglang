import os
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_7_W8A8_MODEL_PATH,
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

MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "140000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "HCCL_BUFFSIZE": "1024",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
}

MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_OTHER_ARGS = [
    "--tp-size",
    8,
    "--mem-fraction-static",
    0.63,
    "--max-running-requests",
    26,
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
    "--enable-prefill-delayer",
    "--prefill-max-requests",
    10,
    "--chunked-prefill-size",
    67072,
    "--max-prefill-token",
    67000,
    "--cuda-graph-bs",
    2,
    4,
    8,
    12,
    16,
    18,
    20,
    22,
    24,
    26,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
]

cmd = "npu-smi info"


class TestKimiK25W4A8(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MINIMAX_M2_7_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_OTHER_ARGS
    envs = MINIMAX_M2_5_W8A8_4P_IN64K_OUT1K_PREFIX90_ENVS
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
        logger.info("S9、curl一条64k长序列请求，完成后记录每张卡的HBM内存占用和总内存")
        self.run_throughput()
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
