"""NPU adaptation of test_hicache_storage_file_backend.py.

Verifies HiCache file storage backend backup/prefetch on NPU. See the
migration report for NPU adaptation rationale.

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_storage_file_backend.py -v
"""

import logging
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=600,
    suite="stage-b-test-4-npu-a3",
    nightly=False,
)

os.environ.setdefault("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")


def _is_in_ci() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI") == "true"


class HiCacheStorageBaseMixin:
    """Base mixin class with common setup and utilities."""

    probe_prompt_tokens = 768

    @classmethod
    def _get_model_name(cls) -> str:
        return LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def _get_extra_server_args(cls) -> list:
        """Return the hicache args specific to this variant (no common args)."""
        return []

    @classmethod
    def _launch_server_with_hicache(cls):
        temp_dir = tempfile.mkdtemp(prefix="hicache_file_backend_")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = temp_dir

        common_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.6",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--page-size",
            "128",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "1.2",
            "--enable-cache-report",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            '{"hicache_storage_pass_prefix_keys": true}',
        ]

        process = popen_launch_server(
            cls._get_model_name(),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=common_args + cls._get_extra_server_args(),
        )
        return process

    @classmethod
    def setUpClass(cls):
        cls.model = cls._get_model_name()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = cls._launch_server_with_hicache()
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def gen_prompt(self, num_tokens: int) -> str:
        """Generate a prompt whose tokenized length is >= num_tokens."""
        base = "The quick brown fox jumps over the lazy dog. "
        text = base
        while len(self.tokenizer.encode(text)) < num_tokens:
            text += base
        encoded = self.tokenizer.encode(text)
        return self.tokenizer.decode(encoded[:num_tokens])

    def send_request(self, prompt: str, max_tokens: int = 150) -> dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                },
            },
            timeout=180,
        )
        assert response.status_code == 200, response.text
        return response.json()

    @staticmethod
    def get_cached_tokens(response_json: dict) -> int:
        return int(response_json.get("meta_info", {}).get("cached_tokens", 0))

    def flush_cache(self):
        """Flush device-tier cache. Retries with backoff because NPU radix-tree
        node release is delayed: /flush_cache may transiently return 400 even
        when queue=0/running=0.
        """
        backoff = [0.5, 1.0, 2.0, 4.0, 8.0]
        last_text = ""
        for i, wait in enumerate(backoff):
            response = requests.post(f"{self.base_url}/flush_cache", timeout=60)
            if response.status_code == 200:
                return
            last_text = response.text
            if response.status_code != 400:
                break
            logging.warning(
                "flush_cache attempt %d returned 400 (%s); retrying in %.1fs",
                i + 1,
                last_text.strip()[:120],
                wait,
            )
            time.sleep(wait)
        assert False, f"Flush cache failed after {len(backoff)} retries: {last_text}"

    def trigger_offloading_and_flush(self):
        """Evict the prefix from device tier, then flush device cache."""
        # Long unrelated prompt evicts the previous prefix to file backend.
        offload_prompt = self.gen_prompt(2048)
        self.send_request(offload_prompt, max_tokens=1)
        self.flush_cache()

    def test_basic_backup_and_prefetch(self):
        """A prefix cached then flushed from device tier must be served from
        the remote file backend on the next request."""
        base_prompt = self.gen_prompt(self.probe_prompt_tokens)

        # 1) Populate device cache with the prefix.
        self.send_request(base_prompt, max_tokens=150)

        # 2) Offload to remote storage and clear device cache.
        self.trigger_offloading_and_flush()

        # 3) Same prefix must hit the remote (file) backend.
        response2 = self.send_request(base_prompt, max_tokens=150)
        cached_tokens = self.get_cached_tokens(response2)
        self.assertGreater(cached_tokens, 700)


class TestHiCacheStoragePageFirstDirectIO(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: page_first_direct mem-layout + direct IO backend (CI runs this)."""

    @classmethod
    def _get_extra_server_args(cls) -> list:
        return [
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-io-backend",
            "direct",
            "--tp-size",
            "2",
        ]


@unittest.skipIf(
    _is_in_ci(), "Skipped in CI: page_first is remapped to page_first_direct on NPU"
)
class TestHiCacheStoragePageFirstLayout(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: page_first mem-layout (remapped to page_first_direct on NPU)."""

    @classmethod
    def _get_extra_server_args(cls) -> list:
        return [
            "--hicache-mem-layout",
            "page_first",
            "--hicache-io-backend",
            "direct",
            "--tp-size",
            "2",
        ]


@unittest.skipIf(
    _is_in_ci(),
    "Skipped in CI: MLA + file backend covered by test_npu_hicache_mla.py",
)
class TestHiCacheStorageMLA(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: MLA model + file backend (tp=2)."""

    @classmethod
    def _get_model_name(cls) -> str:
        return DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH

    @classmethod
    def _get_extra_server_args(cls) -> list:
        return [
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-io-backend",
            "direct",
            "--tp-size",
            "2",
        ]


@unittest.skipIf(_is_in_ci(), "Skipped in CI: long-running accuracy consistency check")
class TestHiCacheStorageAccuracy(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: GSM8K accuracy must be consistent before/after flushing cache."""

    accuracy_threshold = 0.6
    accuracy_delta = 0.03

    @classmethod
    def _get_extra_server_args(cls) -> list:
        return [
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-io-backend",
            "direct",
            "--tp-size",
            "2",
            "--hicache-ratio",
            "1.5",
        ]

    def _run_gsm8k(self) -> float:
        args = SimpleNamespace(
            base_url=self.base_url,
            eval_name="gsm8k",
            api="completion",
            model=self.model,
            num_examples=200,
            num_threads=64,
            max_tokens=512,
            num_shots=5,
            temperature=0.0,
        )
        metrics = run_eval(args)
        return float(metrics["score"])

    def test_eval_accuracy(self):
        """GSM8K score must not drift across cache flush."""
        score_before = self._run_gsm8k()
        self.flush_cache()
        score_after = self._run_gsm8k()
        self.assertGreaterEqual(score_before, self.accuracy_threshold)
        self.assertGreaterEqual(score_after, self.accuracy_threshold)
        self.assertLess(abs(score_before - score_after), self.accuracy_delta)


if __name__ == "__main__":
    unittest.main()
