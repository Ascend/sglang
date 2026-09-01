"""NPU Qwen 3 gsm8k lm-eval Evaluation Test (2-NPU)

Tests Qwen/Qwen3-30B-A3B with lm-eval GSM8K benchmark on NPU
"""

import unittest
import numpy as np
import yaml
import requests
import logging

from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.ascend.e2e.test_npu_kvtc_utils import (
    KVTC_CALIBRATION_PARAMS,
    TestAscendPerformanceKvtcTestCaseLME,
    KVTC_EVALSCOPE,
    KVTC_LM_EVAL,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_30B_A3B_MODEL_PATH,
)

logger = logging.getLogger("kvtc_utils")

ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "DP_ROUND_ROBIN": "1",
    "SGLANG_USE_MAX_DP_ATT": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    168,
    "--chunked-prefill-size",
    -1,
    "--tp-size",
    2,
    "--enable-dp-attention",
    "--dp-size",
    1,
    "--mem-fraction-static",
    0.85,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    "--dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen",
    "--tokenizer-path",
    QWEN3_30B_A3B_MODEL_PATH,
]

BENCHMARK_SIZE_LIMIT = None
KVTC_CALIBRATION_LIMIT = None

class TestNPUQwen3_30BA3B_1P_gsm8k_cr_8(TestAscendPerformanceKvtcTestCaseLME, CustomTestCase):
    """Qwen 3 gsm8k lm-eval Test for NPU"""

    model = QWEN3_30B_A3B_MODEL_PATH
    benchmark_tool = KVTC_EVALSCOPE
    other_args = OTHER_ARGS
    kvtc_keys_compression_ratio = 8
    kvtc_values_compression_ratio = 8
    kvtc_hicache_size = 80
    envs = ENVS
    benchmark_size_limit = BENCHMARK_SIZE_LIMIT
    kvtc_limit_calibration = KVTC_CALIBRATION_LIMIT
    kvtc_force_calibration = False
    kvtc_client_concurrency = 4
    task_list = [{"name": "gsm8k"}]
    kvtc_calibration_params = {
        **KVTC_CALIBRATION_PARAMS,
    }

    def test_KVTC_evalscope_gsm8k_cr_8(self):
        # filling kvcache run
        logger.info("First iteration")
        results = self.launch_eval()
        resp = requests.get(url=self.base_url+ "/trim_cache", timeout=30)
        logger.info(resp.text)
        resp.raise_for_status()
        resp = requests.get(url=self.base_url+ "/radix_tree", timeout=30)
        logger.info(resp.text)
        logger.info("Second iteration")
        results = self.launch_eval()


class TestNPUQwen3_30BA3B_1P_gsm8k_cr_16(TestAscendPerformanceKvtcTestCaseLME, CustomTestCase):
    """Qwen 3 gsm8k lm-eval Test for NPU"""

    model = QWEN3_30B_A3B_MODEL_PATH
    benchmark_tool = KVTC_EVALSCOPE
    other_args = OTHER_ARGS
    kvtc_keys_compression_ratio = 16
    kvtc_values_compression_ratio = 16
    kvtc_hicache_size = 80
    envs = ENVS
    benchmark_size_limit = BENCHMARK_SIZE_LIMIT
    kvtc_limit_calibration = KVTC_CALIBRATION_LIMIT
    kvtc_force_calibration = False
    kvtc_client_concurrency = 4
    task_list = [{"name": "gsm8k"}]
    kvtc_calibration_params = {
        **KVTC_CALIBRATION_PARAMS,
    }

    def test_KVTC_evalscope_gsm8k_cr_16(self):
        # filling kvcache run
        logger.info("First iteration")
        results = self.launch_eval()
        resp = requests.get(url=self.base_url+ "/trim_cache", timeout=30)
        logger.info(resp.text)
        resp.raise_for_status()
        resp = requests.get(url=self.base_url+ "/radix_tree", timeout=30)
        logger.info(resp.text)
        logger.info("Second iteration")
        results = self.launch_eval()

class TestNPUQwen3_30BA3B_1P_gsm8k_cr32(TestAscendPerformanceKvtcTestCaseLME, CustomTestCase):
    """Qwen 3 gsm8k lm-eval Test for NPU"""

    model = QWEN3_30B_A3B_MODEL_PATH
    benchmark_tool = KVTC_EVALSCOPE
    other_args = OTHER_ARGS
    kvtc_keys_compression_ratio = 32
    kvtc_values_compression_ratio = 32
    kvtc_hicache_size = 80
    envs = ENVS
    benchmark_size_limit = BENCHMARK_SIZE_LIMIT
    kvtc_limit_calibration = KVTC_CALIBRATION_LIMIT
    kvtc_force_calibration = False
    kvtc_client_concurrency = 4
    task_list = [{"name": "gsm8k"}]
    kvtc_calibration_params = {
        **KVTC_CALIBRATION_PARAMS,
    }

    def test_KVTC_evalscope_gsm8k_cr_32(self):
        # filling kvcache run
        logger.info("First iteration")
        results = self.launch_eval()
        resp = requests.get(url=self.base_url+ "/trim_cache", timeout=30)
        logger.info(resp.text)
        resp.raise_for_status()
        resp = requests.get(url=self.base_url+ "/radix_tree", timeout=30)
        logger.info(resp.text)
        logger.info("Second iteration")
        results = self.launch_eval()


if __name__ == "__main__":
    unittest.main()
