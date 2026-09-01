import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_30B_A3B_MODEL_PATH,
)
from sglang.test.ascend.e2e.test_npu_kvtc_utils import (
    TestAscendPerformanceKvtcTestCaseBase,
    KVTC_CALIBRATION_PARAMS,
)
from sglang.test.ci.ci_register import register_npu_ci
import openai

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

class TestKVTCQwen30BCalibrateSmoke(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    kvtc_limit_calibration = 5
    kvtc_force_calibration = True
    kvtc_calibration_params = {
        **KVTC_CALIBRATION_PARAMS,
        "N": 10000,
        "q": 100,
    }

    def test_kvtc_qwen3_30b_dump(self):
        client = openai.Client(base_url=f"{self.base_url}/v1", api_key="None")

        messages = [
            {"role": "system", "content": "You are a helpful asistant."},
            {"role": "user", "content": "Compute (3+5)"},
        ]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
        )

        self.assertTrue(
            self._has_current_kvtc_dump(),
            f"KVTC dump was not created: {self.kvtc_dump_path}",
        )

class TestKVTCQwen30BCalibrateReuseSmoke(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    kvtc_limit_calibration = 5
    kvtc_force_calibration = False
    kvtc_calibration_params = {
        **KVTC_CALIBRATION_PARAMS,
        "N": 10000,
        "q": 100,
    }

    def test_kvtc_qwen3_30b_dump_reuse(self):
        client = openai.Client(base_url=f"{self.base_url}/v1", api_key="None")

        messages = [
            {"role": "system", "content": "You are a helpful asistant."},
            {"role": "user", "content": "Compute (3+5)"},
        ]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
        )

        self.assertTrue(
            self._has_current_kvtc_dump(),
            f"KVTC dump was not created: {self.kvtc_dump_path}",
        )


class TestKVTCQwen30BCalibrate(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    kvtc_calibration_params = {
        **KVTC_CALIBRATION_PARAMS,
        "N": 200000,
        "q": 10000,
    }

    def test_kvtc_qwen3_30b_dump_reuse(self):
        client = openai.Client(base_url=f"{self.base_url}/v1", api_key="None")

        messages = [
            {"role": "system", "content": "You are a helpful asistant."},
            {"role": "user", "content": "Compute (3+5)"},
        ]
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
        )

        self.assertTrue(
            self._has_current_kvtc_dump(),
            f"KVTC dump was not created: {self.kvtc_dump_path}",
        )

if __name__ == "__main__":
    unittest.main()
