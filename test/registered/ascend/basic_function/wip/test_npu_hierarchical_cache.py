import unittest
from types import SimpleNamespace

import requests
from transformers import AutoTokenizer

from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    check_role,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    TestAscendPerfMultiNodePdSepTestCaseBase,
    logger,
)
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval

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
    "prefill_args": BASE_PREFILL_ARGS + ["--disable-radix-cache"],
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


def run_gsm8k_accuracy(base_url: str, model_path: str) -> float:
    """
    Run GSM8K evaluation and return accuracy.

    Used to verify no precision regression when enabling hierarchical cache.
    """
    args = SimpleNamespace(
        max_tokens=512,
        base_url=base_url,
        model=model_path,
        eval_name="gsm8k",
        api="completion",
        num_examples=200,
        num_threads=128,
        num_shots=8,
    )
    metrics = run_eval(args)
    return metrics["accuracy"]


# ====================== Cross-Test Shared Context ======================
class HierarchicalCacheBenchmarkContext:
    """
    Shared context across test cases running in the same Python process.

    Used to pass baseline accuracy from the 'cache disabled' test case
    to the 'cache enabled' test case for precision regression checking.
    """

    def __init__(self):
        self.baseline_accuracy_without_cache: float | None = None

    def ensure_baseline_accuracy(self) -> float:
        if self.baseline_accuracy_without_cache is None:
            raise RuntimeError(
                "Baseline accuracy not found. "
                "Please ensure test order: "
                "TestDeepSeekV32W8A8PdSepDisableHierarchicalCache.test_gsm8k_baseline_accuracy"
            )
        return self.baseline_accuracy_without_cache


hierarchical_cache_ctx = HierarchicalCacheBenchmarkContext()


# ====================== Test Case: Disable L1 & L2 Cache (Baseline) ======================
class TestDeepSeekV32W8A8PdSepDisableHierarchicalCache(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """
    Verify long-context inference works correctly with L1/L2 cache disabled

    [Test Category] Functional
    [Test Target] Long-Context Inference Correctness (Hierarchical Cache Disabled)
    --disable-radix-cache
    """

    model_config = MODEL_CONFIG_DISABLE_HIERARCHICAL_CACHE

    @check_role(allowed_roles=["router"])
    def test_gsm8k_baseline_accuracy(self):
        # Establish GSM8K accuracy baseline with hierarchical cache disabled
        accuracy = run_gsm8k_accuracy(
            self.base_url,
            self.model_config.get("model_path"),
        )
        hierarchical_cache_ctx.baseline_accuracy_without_cache = accuracy


# ====================== Test Case: Enable L1 & L2 Cache ======================
class TestDeepSeekV32W8A8PdSepEnableHierarchicalCache(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """
    Verify long-context inference works correctly with L1/L2 cache enabled

    [Test Category] Functional
    [Test Target] Hierarchical Cache Correctness & Performance (L1/L2 Enabled)
    --enable-hierarchical-cache
    """

    model_config = MODEL_CONFIG_ENABLE_HIERARCHICAL_CACHE

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
            trust_remote_code=True,
        )
        prompt_text = "hello world " * (600 // 2 + 1)
        prompt_ids = cls.tokenizer.encode(prompt_text, add_special_tokens=False)[:600]
        cls.shared_prefix_prompt = cls.tokenizer.decode(prompt_ids)

    @check_role(allowed_roles=["router"])
    def _send_shared_prefix_request(
        self,
        max_new_tokens: int = 1,
    ):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": self.shared_prefix_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )

        self.assertEqual(response.status_code, 200, "Generate API call failed")
        result = response.json()
        logger.info(f"Shared prefix response: {result}")
        meta = result.get("meta_info", {})

        cached_tokens = meta.get("cached_tokens", 0)
        e2e_latency = meta.get("e2e_latency", 0)

        return cached_tokens, e2e_latency

    @check_role(allowed_roles=["router"])
    def test_accuracy_no_regression_with_cache(self):
        # Verify model accuracy does not regress after enabling hierarchical cache
        baseline_accuracy = hierarchical_cache_ctx.ensure_baseline_accuracy()
        current_accuracy = run_gsm8k_accuracy(
            self.base_url,
            self.model_config.get("model_path"),
        )

        self.assertGreaterEqual(
            current_accuracy,
            baseline_accuracy - 0.02,
            msg="Accuracy regression detected after enabling hierarchical cache",
        )

    @check_role(allowed_roles=["router"])
    def test_hierarchical_cache_hit_and_latency_reduction(self):
        # First request: expect no cache hit (cold run)
        cached_tokens_1, e2e_latency_1 = self._send_shared_prefix_request()
        self.assertEqual(
            cached_tokens_1,
            0,
            "First request should have zero cached tokens",
        )

        # Second request: expect deterministic cache hit and lower latency
        cached_tokens_2, e2e_latency_2 = self._send_shared_prefix_request()
        self.assertEqual(
            cached_tokens_2,
            512,
            "Second request should hit hierarchical cache",
        )
        self.assertLess(
            e2e_latency_2,
            e2e_latency_1,
            "E2E latency should decrease on cache hit",
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(
        TestDeepSeekV32W8A8PdSepDisableHierarchicalCache("test_gsm8k_baseline_accuracy")
    )
    suite.addTest(
        TestDeepSeekV32W8A8PdSepEnableHierarchicalCache(
            "test_accuracy_no_regression_with_cache"
        )
    )
    suite.addTest(
        TestDeepSeekV32W8A8PdSepEnableHierarchicalCache(
            "test_hierarchical_cache_hit_and_latency_reduction"
        )
    )

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
