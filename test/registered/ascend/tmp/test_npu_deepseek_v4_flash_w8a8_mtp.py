import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH,
)
from sglang.test.ascend.test_ascend_utils import (
    logger,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

DEEPSEEK_V4_FLASH_W8A8_MTP_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_BUFFSIZE": "2000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "0",
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
}

DEEPSEEK_V4_FLASH_W8A8_MTP_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "dsv4",
    "--quantization",
    "modelslim",
    "--kv-cache-dtype",
    "auto",
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--page-size",
    128,
    "--max-running-requests",
    16,
    "--mem-fraction-static",
    0.65,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--disable-overlap-schedule",
    "--skip-server-warmup",
    "--watchdog-timeout",
    9000,
    "--cuda-graph-bs",
    1,
    2,
    4,
]

cmd = "npu-smi info"


class TestDEEPSEEKV4FLASHW8A8MTP(CustomTestCase):
    model = DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_MTP_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_W8A8_MTP_ENVS
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
        res1 = run_command("cat ./out_log.txt | grep 'Init torch distributed ends'")
        logger.info(res1)
        logger.info("S4、记录模型加载后，模型权重内存占用")
        res2 = run_command("cat ./out_log.txt | grep 'Load weight end'")
        logger.info(res2)
        logger.info("S5、记录KV cache分配后，KV cache内存占用")
        res3 = run_command("cat ./out_log.txt | grep 'KV Cache is allocated'")
        logger.info(res3)
        logger.info("S6、记录NPU graph buffer分配后，NPU graph buffer内存占用")
        res4 = run_command("cat ./out_log.txt | grep 'Capture npu graph end'")
        logger.info(res4)
        logger.info("S7、服务启动成功后执行npu-smi info")
        raw_result = run_command(cmd)
        logger.info(raw_result)

    def test_2(self):
        logger.info("S9、curl一条请求，完成后记录每张卡的HBM内存占用和总内存")
        requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 100,
                },
            },
        )
        raw_result = run_command(cmd)
        logger.info(raw_result)

    def test_3(self):
        if hasattr(self, "process") and self.process:
            try:
                kill_process_tree(self.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")
        logger.info("S9、停止服务，等待服务完全停止后，记录每张卡的HBM内存占用和总内")
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
