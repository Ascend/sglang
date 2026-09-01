import logging
import unittest

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1200, suite="base-c-test-perf-16-npu-a3")
register_npu_ci(est_time=1200, suite="nightly-perf-16-npu-a3", nightly=True)

# Environment variables for DSV4-Flash single-node PD-mix deployment.
# Identical to the 8p performance case; the radix cache flag is the only
# difference.
DEEPSEEK_V4_FLASH_W8A8_8P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # deepep
    "DEEPEP_HCCL_BUFFSIZE": "1000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    # skip gpu branch
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
    # MTP (EAGLE) related envs
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

# Server launch arguments for DSV4-Flash W8A8 single-node 8p with the unified
# radix cache ENABLED.
#
# Note: the 8p performance case passes `--disable-radix-cache`. That flag is
# intentionally omitted here so the evict -> unevict path of the unified radix
# cache is exercised. DeepSeek-V4 auto-registers a sidecar component into the
# unified radix cache; when the cache fills up and later re-inserts an evicted
# prefix, the tree walk hits `recover_after_unevict` / `update_component_on_insert_overlap`.
DEEPSEEK_V4_FLASH_W8A8_8P_RADIX_CACHE_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.6,
    "--prefill-max-requests",
    2,
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    160,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    10,
    # MTP (EAGLE) configuration.
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
]

# Filler used to build long, distinct prefixes. Each prefix is tagged with a
# unique marker so that different requests occupy different radix-tree branches
# (a shared prefix would defeat the "fill the cache" step by reusing nodes).
_FILLER = "The quick brown fox jumps over the lazy dog. "


class TestNPUDeepSeekV4FlashW8A8RadixCacheEvictUnevict(CustomTestCase):
    """Functional model test: DSV4-Flash unified radix cache evict -> unevict.

    Covers the scenario described in sgl-project/sglang#37091, where a cache
    contract change ("Unified Cache][1/N]") added a required ``result``
    parameter to :meth:`TreeComponent.recover_after_unevict` /
    :meth:`TreeComponent.update_component_on_insert_overlap`. A sidecar
    component that overrides the old signature then raises ``TypeError`` on the
    evict -> unevict path.

    This case deliberately keeps the unified radix cache on and drives enough
    distinct long prefixes through the server to force LRU eviction, then
    re-sends the evicted anchor prefix to trigger the unevict path. The core
    assertion is that every request succeeds (no server-side crash / TypeError);
    ``cached_tokens`` is logged for diagnostics.
    """

    model = DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # Tunable knobs. `fill_count * fill_tokens` must exceed the residual KV-cache
    # budget left after the 16-rank model weights are loaded (mem-fraction 0.6).
    anchor_tokens = 8000
    fill_tokens = 8000
    fill_count = 32
    max_new_tokens = 1

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=DEEPSEEK_V4_FLASH_W8A8_8P_RADIX_CACHE_ARGS,
            env=DEEPSEEK_V4_FLASH_W8A8_8P_ENVS,
        )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _gen_prompt(self, num_tokens: int, tag: str) -> str:
        """Build a ~num_tokens prompt with a unique leading marker."""
        text = f"### {tag} ### " + _FILLER
        while len(self.tokenizer.encode(text)) < num_tokens:
            text += _FILLER
        encoded = self.tokenizer.encode(text)
        return self.tokenizer.decode(encoded[:num_tokens])

    def _generate(self, prompt: str) -> dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.max_new_tokens,
                },
            },
            timeout=600,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"generate failed with status {response.status_code}: {response.text}",
        )
        return response.json()

    @staticmethod
    def _cached_tokens(response_json: dict) -> int:
        return int(response_json.get("meta_info", {}).get("cached_tokens", 0))

    def test_radix_cache_evict_then_unevict(self):
        anchor_prompt = self._gen_prompt(self.anchor_tokens, "anchor")

        # 1) Populate the cache with the anchor prefix.
        first = self._generate(anchor_prompt)
        first_cached = self._cached_tokens(first)
        logging.warning("anchor first request cached_tokens=%d", first_cached)
        self.assertEqual(first_cached, 0, "first anchor request should not hit cache")

        # 2) Fill the cache with distinct long prefixes to force LRU eviction of
        #    the anchor (and earlier fill) nodes.
        for i in range(self.fill_count):
            fill_prompt = self._gen_prompt(self.fill_tokens, f"fill-{i}")
            self._generate(fill_prompt)

        # 3) Re-send the anchor prefix. If it was evicted, the tree walk hits the
        #    unevict path (recover_after_unevict). A TypeError here would surface
        #    the #37091 regression; only a successful response is asserted.
        second = self._generate(anchor_prompt)
        second_cached = self._cached_tokens(second)
        logging.warning(
            "anchor re-send request cached_tokens=%d (evict->unevict path)",
            second_cached,
        )


if __name__ == "__main__":
    unittest.main()
