"""NPU adaptation of test_hicache_storage_file_backend.py."""

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

# Force deterministic inference so that the cached_tokens assertion is stable
# across runs (mirrors the GPU test's SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1).
os.environ.setdefault("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")


class HiCacheStorageBaseMixin:
    """Shared setup / helpers for the file-backend storage tests.

    Mirrors `HiCacheStorageBaseMixin` from the GPU test
    `test_hicache_storage_file_backend.py`. Each concrete test class
    selects its own model and server arguments by overriding
    `_get_model_name()` / `_get_extra_server_args()`.
    """

    # Number of tokens used for the backup/prefetch probe. Kept identical to
    # the GPU test so the `cached_tokens > 700` threshold stays meaningful.
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

    # ---------- helpers ----------

    def gen_prompt(self, num_tokens: int) -> str:
        """Generate a prompt whose tokenized length is >= num_tokens."""
        # Repeat a base sentence and trim to the requested token count.
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
        """Flush the device-tier radix cache so the next request is forced to
        pull from the remote (file) storage backend.

        On NPU, `/flush_cache` may transiently return 400 with "Cache not
        flushed because there are pending requests" even though
        #queue-req=0 and #running-req=0, because the just-finished request's
        radix-tree node has not been released yet. Retry with backoff.
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
        """Force device-cache offload to remote storage, then flush device cache.

        Sending a large unrelated prompt evicts the previous prefix from the
        device tier (forcing it to be offloaded to the file backend); the
        subsequent `/flush_cache` clears the device tier so the next request
        with the original prefix must hit the remote storage.
        """
        offload_prompt = self.gen_prompt(2048)
        self.send_request(offload_prompt, max_tokens=1)
        self.flush_cache()

    # ---------- core test ----------

    def test_basic_backup_and_prefetch(self):
        """A prefix that was cached, then flushed from the device tier, must be
        served from the remote file backend on the next request."""
        logging.warning(
            "\n=== test_basic_backup_and_prefetch [%s] ===", self.__class__.__name__
        )
        base_prompt = self.gen_prompt(self.probe_prompt_tokens)

        # 1) First request - populate device cache with the prefix.
        response1 = self.send_request(base_prompt, max_tokens=150)
        logging.warning(
            "first request cached_tokens=%d",
            self.get_cached_tokens(response1),
        )

        # 2) Force the prefix to be offloaded to remote storage and clear the
        #    device cache.
        self.trigger_offloading_and_flush()

        # 3) Second request with the same prefix - must hit the remote (file)
        #    backend, so the vast majority of the 768 prefix tokens are cached.
        response2 = self.send_request(base_prompt, max_tokens=150)
        cached_tokens = self.get_cached_tokens(response2)
        logging.warning("second request cached_tokens=%d (expect > 700)", cached_tokens)
        self.assertGreater(
            cached_tokens,
            700,
            "Remote (file backend) cache miss: cached_tokens=%d" % cached_tokens,
        )


class TestHiCacheStoragePageFirstDirectIO(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: page_first_direct mem-layout + direct IO backend (single NPU)."""

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


class TestHiCacheStorageAccuracy(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: GSM8K accuracy must be consistent before / after flushing cache.

    Ported from the GPU `TestHiCacheStorageAccuracy.test_eval_accuracy`. The
    GPU version asserts two GSM8K runs (before / after flush) differ by < 0.03
    and both > 0.6. We reuse `run_eval` (NPU convention) instead of
    `MGSMEnMixin`.
    """

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
        logging.warning("\n=== test_eval_accuracy [%s] ===", self.__class__.__name__)
        score_before = self._run_gsm8k()
        self.flush_cache()
        score_after = self._run_gsm8k()
        logging.warning("gsm8k score before=%.4f after=%.4f", score_before, score_after)
        self.assertGreaterEqual(score_before, self.accuracy_threshold)
        self.assertGreaterEqual(score_after, self.accuracy_threshold)
        self.assertLess(
            abs(score_before - score_after),
            self.accuracy_delta,
            "GSM8K score drifted across cache flush: %.4f vs %.4f"
            % (score_before, score_after),
        )


if __name__ == "__main__":
    unittest.main()
