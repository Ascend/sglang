#!/usr/bin/env python3

"""Verify Qwen3-32B can restore an L1-evicted prefix from HiCache L2."""

import logging
import math
import os
import random
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestHiCacheL2LoadBack(CustomTestCase):
    """Force L1 eviction and verify the request-level L2 hit breakdown."""

    model = QWEN3_32B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    target_tokens = 768
    fill_prompt_tokens = 1024
    cache_hit_threshold = 600

    @classmethod
    def setUpClass(cls):
        cls.process = None

        env = os.environ.copy()
        env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("HCCL_SOCKET_IFNAME", "lo")
        env.setdefault("GLOO_SOCKET_IFNAME", "lo")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1200),
            other_args=[
                "--tp-size",
                "2",
                "--nnodes",
                "1",
                "--attention-backend",
                "ascend",
                "--device",
                "npu",
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--disable-cuda-graph",
                "--max-total-tokens",
                "32768",
                "--max-prefill-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2.0",
                "--hicache-write-policy",
                "write_through",
                "--radix-eviction-policy",
                "lru",
                "--hicache-io-backend",
                "kernel_ascend",
                "--hicache-mem-layout",
                "page_first_direct",
                "--enable-cache-report",
                "--page-size",
                "128",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    @staticmethod
    def gen_input_ids(num_tokens: int, seed: int) -> list[int]:
        """Generate an exact-length deterministic prompt with little prefix overlap."""
        # Qwen3-32B has a much larger vocabulary; this conservative range avoids
        # added/special token IDs while keeping every generated ID valid.
        lower = 1_000
        upper = 30_000
        rng = random.Random(seed)
        return [rng.randrange(lower, upper) for _ in range(num_tokens)]

    def send_generate_request(
        self, input_ids: list[int], max_new_tokens: int = 1
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def get_cache_tokens(response_json: dict) -> tuple[int, int, int, int]:
        meta_info = response_json.get("meta_info", {})
        details = meta_info.get("cached_tokens_details") or {}
        return (
            int(meta_info.get("cached_tokens", 0)),
            int(details.get("device", 0)),
            int(details.get("host", 0)),
            int(details.get("storage", 0)),
        )

    def get_l1_capacity(self) -> int:
        response = requests.get(f"{self.base_url}/server_info", timeout=60)
        self.assertEqual(response.status_code, 200, response.text)
        server_info = response.json()
        self.assertIn("max_total_num_tokens", server_info)
        return int(server_info["max_total_num_tokens"])

    def assert_l2_hit(self, response_json: dict, label: str) -> None:
        total, device, host, storage = self.get_cache_tokens(response_json)
        logging.warning(
            "%s cache hit: total=%d, device=%d, host=%d, storage=%d",
            label,
            total,
            device,
            host,
            storage,
        )
        self.assertEqual(storage, 0, f"{label} unexpectedly hit L3 storage")
        self.assertGreater(
            host,
            self.cache_hit_threshold,
            f"{label} did not load back from L2: "
            f"total={total}, device={device}, host={host}, storage={storage}",
        )

    def test_l2_cache_reuse_after_l1_eviction(self):
        target_prompt = self.gen_input_ids(self.target_tokens, seed=1)
        canary_prompt = self.gen_input_ids(self.target_tokens, seed=2)

        # 1. Populate target and canary. Both are initially cache misses.
        target_first = self.send_generate_request(target_prompt)
        self.assertEqual(self.get_cache_tokens(target_first)[0], 0)

        canary_first = self.send_generate_request(canary_prompt)
        self.assertEqual(self.get_cache_tokens(canary_first)[0], 0)

        # 2. Fill beyond the server-reported L1 token capacity without probing
        # target/canary. Probing them during filling would refresh their LRU age.
        l1_capacity = self.get_l1_capacity()
        fill_iterations = math.ceil(l1_capacity / self.fill_prompt_tokens) + 4
        logging.warning(
            "Filling L1: capacity=%d tokens, iterations=%d, prompt_tokens=%d",
            l1_capacity,
            fill_iterations,
            self.fill_prompt_tokens,
        )

        for index in range(fill_iterations):
            fill_prompt = self.gen_input_ids(
                self.fill_prompt_tokens, seed=10_000 + index
            )
            self.send_generate_request(fill_prompt)

        # 3. Canary is newer than target. A host hit here demonstrates that L1
        # eviction happened and that the test stopped before exhausting L2.
        canary_probe = self.send_generate_request(canary_prompt)
        self.assert_l2_hit(canary_probe, "canary")

        # 4. The older target must also be restored from CPU L2.
        target_load_back = self.send_generate_request(target_prompt)
        self.assert_l2_hit(target_load_back, "target")

        # 5. After load-back, the next target request should hit device L1.
        target_after_load_back = self.send_generate_request(target_prompt)
        total, device, host, storage = self.get_cache_tokens(target_after_load_back)
        logging.warning(
            "target after load-back: total=%d, device=%d, host=%d, storage=%d",
            total,
            device,
            host,
            storage,
        )
        self.assertEqual(storage, 0)
        self.assertGreater(
            device,
            self.cache_hit_threshold,
            "Target was not retained in L1 after L2 load-back: "
            f"total={total}, device={device}, host={host}, storage={storage}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
