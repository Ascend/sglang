import unittest
from types import SimpleNamespace

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    check_role,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendPerfMultiNodePdSepTestCaseBase,
    run_aisbench,
)
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="multi nodes testcase",
)

# ====================== Base Configuration ======================
BASE_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

BASE_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

BASE_PREFILL_ARGS = [
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    "16",
    "--mem-fraction-static",
    "0.8",
    "--quantization",
    "modelslim",
    "--context-length",
    "8192",
    "--chunked-prefill-size",
    "-1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]

BASE_DECODE_ARGS = [
    "--nnodes",
    "1",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--tp-size",
    "16",
    "--mem-fraction-static",
    "0.8",
    "--quantization",
    "modelslim",
    "--context-length",
    "8192",
    "--chunked-prefill-size",
    "-1",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--trust-remote-code",
    "--cuda-graph-bs",
    "256",
    "128",
    "64",
    "--watchdog-timeout",
    "9000",
    "--dtype",
    "bfloat16",
]

# ====================== Disable L1&L2 Cache Config ======================
MODEL_CONFIG_DISABLE_HIERARCHICAL_CACHE = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "prefill_args": BASE_PREFILL_ARGS,
    "decode_args": BASE_DECODE_ARGS,
    "router_args": [],
}

# ====================== Enable L1&L2 Cache Config ======================
MODEL_CONFIG_ENABLE_HIERARCHICAL_CACHE = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": BASE_PREFILL_ENVS,
    "decode_envs": BASE_DECODE_ENVS,
    "prefill_args": BASE_PREFILL_ARGS + ["--enable-hierarchical-cache"],
    "decode_args": BASE_DECODE_ARGS,
    "router_args": [],
}


# ====================== Test Case ======================
class TestDeepSeekV32HierarchicalCacheHit(TestAscendPerfMultiNodePdSepTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    request_rate = 40
    max_concurrency = 1
    num_prompts = 1
    input_len = 1000
    output_len = 20
    random_range_ratio = 1
    seed = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH, trust_remote_code=True
        )

    @check_role(allowed_roles=["router"])
    def send_long_prompt_request(self, prompt_token_len=600, max_new_tokens=1):
        prompt = "hello world " * (prompt_token_len // 2 + 1)
        prompt = self.tokenizer.decode(
            self.tokenizer.encode(prompt, add_special_tokens=False)[:prompt_token_len]
        )

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
        )

        self.assertEqual(response.status_code, 200, "Failed to call generate API")
        result = response.json()
        cached_tokens = result.get("meta_info").get("cached_tokens", 0)
        return cached_tokens

    @check_role(allowed_roles=["router"])
    def run_gsm8k_test(
        self,
        expect_accuracy,
        num_shots=8,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
    ):
        args = SimpleNamespace(
            num_shots=num_shots,
            data_path=data_path,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            parallel=parallel,
            host=f"http://{self.host}",
            port=self.port,
        )
        metrics = run_eval_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )

    def test_hierarchical_cache_hit_and_ttft_reduce(self):
        self.__class__.model_config = MODEL_CONFIG_DISABLE_HIERARCHICAL_CACHE
        try:
            self.start_pd_server()
            self.start_router_server()

            cached_tokens_1 = self.send_long_prompt_request(
                prompt_token_len=600, max_new_tokens=1
            )
            self.assertEqual(
                cached_tokens_1, 0, msg="First request cached tokens should be 0"
            )

            cached_tokens_2 = self.send_long_prompt_request(
                prompt_token_len=600, max_new_tokens=1
            )
            self.assertEqual(cached_tokens_2, 0, msg="Cache hit tokens should be 512")

            metrics1 = run_aisbench(
                host=self.host,
                port=str(self.port),
                model_path=self.model_config.get("model_path"),
                dataset_type=self.dataset_type,
                dataset_path=self.dataset_path,
                input_len=self.input_len,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                image_resolution=self.image_resolution,
                random_range_ratio=self.random_range_ratio,
                dp=self.dp,
                generation_kwargs=self.generation_kwargs,
            )
        finally:
            if self.process:
                kill_process_tree(self.process.pid)

        self.__class__.model_config = MODEL_CONFIG_ENABLE_HIERARCHICAL_CACHE
        try:
            self.start_pd_server()
            self.start_router_server()

            cached_tokens_1 = self.send_long_prompt_request(
                prompt_token_len=600, max_new_tokens=1
            )
            self.assertEqual(
                cached_tokens_1, 0, msg="First request cached tokens should be 0"
            )

            cached_tokens_2 = self.send_long_prompt_request(
                prompt_token_len=600, max_new_tokens=1
            )
            self.assertEqual(cached_tokens_2, 512, msg="Cache hit tokens should be 512")

            metrics2 = run_aisbench(
                host=self.host,
                port=str(self.port),
                model_path=self.model_config.get("model_path"),
                dataset_type=self.dataset_type,
                dataset_path=self.dataset_path,
                input_len=self.input_len,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                image_resolution=self.image_resolution,
                random_range_ratio=self.random_range_ratio,
                dp=self.dp,
                generation_kwargs=self.generation_kwargs,
            )

            self.assertLess(
                metrics2["TTFT"],
                metrics1["TTFT"],
                msg="TTFT should be reduced after cache hit",
            )

            self.run_gsm8k_test(0.95, num_shots=5)

        finally:
            if self.process:
                kill_process_tree(self.process.pid)


if __name__ == "__main__":
    unittest.main()
