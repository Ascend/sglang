import unittest
import logging

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_30B_A3B_MODEL_PATH,
)
from sglang.test.ascend.e2e.test_npu_kvtc_utils import (
    TestAscendPerformanceKvtcTestCaseBase,
    KVTC_CACHE_PATH
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

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
    "--disable-radix-cache",
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
]

import asyncio
from openai import AsyncOpenAI

logger = logging.getLogger(__file__)

async def run_requests(dataset_name, client, requests, client_concurrency):
    concurrency_semaphore = asyncio.Semaphore(client_concurrency)

    async def send_request(dataset_name, client, entry):
        prompt_id, prompt = entry

        async with concurrency_semaphore:
            logger.debug(f"KVTC calibration {dataset_name} request {prompt_id}")
            response = await client.chat.completions.create(
                model="Qwen3-30B-A3B",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )


    tasks = [send_request(dataset_name, client, r) for r in requests]

    return await asyncio.gather(*tasks)

class TestKVTCQwen30B_dump_openmath_smoke(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS + [
        "--dump-kv-path",
        KVTC_CACHE_PATH / "openmath_dump",
    ]
    envs = ENVS
    kvtc_dataset_name = "openmath"
    client_concurrency = 16

    def test_kvtc_qwen3_30b_dump_openmath(self):
        client = AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key="None")
        prompts = self.get_kvtc_prompts()[:3]

        asyncio.run(
            run_requests(
                self.kvtc_dataset_name, client, prompts, self.client_concurrency
            )
        )

class TestKVTCQwen30B_dump_fineweb_smoke(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS + [
        "--dump-kv-path",
        KVTC_CACHE_PATH / "fineweb_dump",
    ]
    envs = ENVS
    kvtc_dataset_name = "fineweb"
    client_concurrency = 16

    def test_kvtc_qwen3_30b_dump_fineweb(self):
        client = AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key="None")
        prompts = self.get_kvtc_prompts()[:3]

        asyncio.run(
            run_requests(
                self.kvtc_dataset_name, client, prompts, self.client_concurrency
            )
        )


class TestKVTCQwen30B_dump_openmath(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS + [
        "--dump-kv-path",
        KVTC_CACHE_PATH / "openmath_dump",
    ]
    envs = ENVS
    kvtc_dataset_name = "openmath"
    client_concurrency = 16

    def test_kvtc_qwen3_30b_dump_openmath(self):
        client = AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key="None")
        prompts = self.get_kvtc_prompts()

        asyncio.run(
            run_requests(
                self.kvtc_dataset_name, client, prompts, self.client_concurrency
            )
        )


class TestKVTCQwen30B_dump_fineweb(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS + [
        "--dump-kv-path",
        KVTC_CACHE_PATH / "fineweb_dump",
    ]
    envs = ENVS
    kvtc_dataset_name = "fineweb"
    client_concurrency = 16

    def test_kvtc_qwen3_30b_dump_fineweb(self):
        client = AsyncOpenAI(base_url=f"{self.base_url}/v1", api_key="None")
        prompts = self.get_kvtc_prompts()

        asyncio.run(
            run_requests(
                self.kvtc_dataset_name, client, prompts, self.client_concurrency
            )
        )


if __name__ == "__main__":
    unittest.main()
