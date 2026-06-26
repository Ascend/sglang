import os
import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
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
    est_time=1800,
    suite="full-16-npu-a3",
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


def get_hbm_usage_list_16(raw_result: str) -> list:
    """
    返回长度为 16 的列表，索引 0~15 对应 chip 0~15 的 HBM 占用(MB)
    """
    pattern = re.compile(r"\d+\s+/\s*\d+\s+\d+\s+/\s*\d+\s+(\d+)\s+/\s*(\d+)$")
    hbm_list = []

    for line in raw_result.splitlines():
        match = pattern.search(line)
        if match:
            hbm_used = int(match.group(1))
            hbm_list.append(hbm_used)

            # 防御性校验：防止异常输出
            if len(hbm_list) == 16:
                break

    return hbm_list


cmd = "npu-smi info"


class TestKimiK25W4A8(CustomTestCase):
    model = KIMI_K2_6_W4A8_MODEL_PATH
    other_args = KIMI_K2_6_OTHER_ARGS
    envs = KIMI_K2_6_ENVS
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
