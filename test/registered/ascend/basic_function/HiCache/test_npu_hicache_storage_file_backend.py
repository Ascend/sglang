"""NPU adaptation of test_hicache_storage_file_backend.py.

Test target: verify the HiCache file storage backend's backup (offload to
remote file storage) and prefetch (reload from remote) mechanism on NPU.

[Test Category] Functional
[Test Target] --hicache-storage-backend file (backup / prefetch / accuracy)

Key observation points ported from the GPU test:
  * After the device cache is flushed, a follow-up request with the same
    prefix must hit the remote (file) storage backend, i.e. the
    `cached_tokens` reported by `/generate` must be > 700 for a 768-token
    prefix prompt.
  * GSM8K accuracy must stay consistent before / after flushing the cache
    (delta < 0.03, both runs > 0.6).

NPU adaptation notes (see report for the full rationale):
  * `--attention-backend ascend` and `--disable-cuda-graph` are mandatory
    on NPU and are added to every server launch.
  * `--mem-fraction-static` is bumped from 0.6 to 0.8 to follow the NPU
    HiCache convention used by the existing `test_npu_hicache.py`.
  * `--page-size` is forced to 128 because NPU only supports 128
    (GPU used 64). The `cached_tokens > 700` threshold is independent of
    page size, so the assertion still holds.
  * `--hicache-io-backend direct` is kept (NPU supports `direct`;
    `kernel` is remapped to `kernel_ascend` on NPU).
  * `--hicache-mem-layout page_first_direct` is the only layout NPU
    supports; GPU's `page_first` is internally converted to
    `page_first_direct` on NPU, so the `PageFirstLayout` variant is kept
    but skipped in CI (it exercises the same code path as
    `PageFirstDirectIO` on NPU).
  * Models are taken from `test_ascend_utils.py`:
      - base / page-first / accuracy variants: LLAMA_3_2_1B_INSTRUCT
        (matches the existing `test_npu_hicache.py` choice; 1B model boots
        fast and leaves enough Host-Tier room for the 768-token backup
        test).
      - MLA variant: DEEPSEEK_CODER_V2_LITE (the NPU-available MLA model
        closest to GPU's DeepSeek-Coder-V2-Lite-Instruct).
  * `--hicache-storage-prefetch-policy wait_complete` and
    `--hicache-storage-backend-extra-config {"hicache_storage_pass_prefix_keys": true}`
    are documented as "Special for GPU" in 参数.txt. They are kept here so
    that, if NPU silently accepts them, the test still exercises the same
    code path as GPU; if NPU rejects them the server launch will fail and
    we will trim them in a follow-up commit.
  * Evaluation uses `sglang.test.run_eval.run_eval` (the same helper used
    by `test_npu_hicache_page.py` and `test_npu_speculative_attention_mode.py`)
    instead of GPU's `MMLUMixin` / `MGSMEnMixin` to stay consistent with
    the existing NPU test style.
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
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
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


def _is_in_ci() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI") == "true"


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
        return LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

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
            "0.8",
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
    """Variant: page_first_direct mem-layout + direct IO backend (single NPU).

    This is the only variant that runs in CI; the other three variants below
    are skipped in CI to avoid redundant coverage (see their docstrings).
    """

    @classmethod
    def _get_extra_server_args(cls) -> list:
        return [
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-io-backend",
            "direct",
        ]


@unittest.skipIf(
    _is_in_ci(), "Skipped in CI: page_first is remapped to page_first_direct on NPU"
)
class TestHiCacheStoragePageFirstLayout(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: page_first mem-layout.

    Skipped in CI because NPU does not support `page_first` and internally
    remaps it to `page_first_direct`, which is already covered by
    `TestHiCacheStoragePageFirstDirectIO`. Kept here for parity with the GPU
    test and for manual / nightly runs.
    """

    @classmethod
    def _get_extra_server_args(cls) -> list:
        return [
            "--hicache-mem-layout",
            "page_first",
            "--hicache-io-backend",
            "direct",
        ]


@unittest.skipIf(
    _is_in_ci(),
    "Skipped in CI: MLA + file backend covered by test_npu_hicache_mla.py smoke",
)
class TestHiCacheStorageMLA(HiCacheStorageBaseMixin, CustomTestCase):
    """Variant: MLA model + file backend (tp=2).

    Uses DeepSeek-Coder-V2-Lite (the NPU-available MLA model closest to the
    GPU test's DeepSeek-Coder-V2-Lite-Instruct). Skipped in CI to keep the
    suite short; the MLA + HiCache combination is already smoke-tested by
    `test_npu_hicache_mla.py`.
    """

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
    """Variant: GSM8K accuracy must be consistent before / after flushing cache.

    Ported from the GPU `TestHiCacheStorageAccuracy.test_eval_accuracy`. The
    GPU version asserts two GSM8K runs (before / after flush) differ by < 0.03
    and both > 0.6. We reuse `run_eval` (NPU convention) instead of
    `MGSMEnMixin`. Skipped in CI because it is a long-running evaluation.
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
