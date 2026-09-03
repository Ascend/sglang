"""End-to-end behavior test for --enable-session-radix-cache (UnifiedRadixCache).
Within-run A/B under a bounded KV pool (``--max-total-tokens``):
- Prompt A is requested with a top-level ``session_id``; once the request
finishes, its reusable cache leaves are registered under the session.
- Prompt B is requested without a session. B doubles as the flag-off
control: it is exactly what happens to every prompt without session
protection.
- Unique flood prompts pressure the pool to ~2x capacity. The unprotected
B must be evicted while the session-referenced A survives (soft
protection).
- ``/close_session`` releases A's references; a second flood round must
then evict A like any unprotected entry.

"""

import random
import tempfile
import unittest
import uuid

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)

# Total KV pool size used both as context length and max total tokens.
# Flood prompts are sized so that ~2x capacity of unique tokens is pushed
# through the pool, forcing eviction of unprotected entries.
MAX_TOTAL_TOKENS = 8192
WORDS_PER_PROMPT = 1200
NUM_FLOOD_PROMPTS = 10
# A session-protected prompt must keep most of its cache hits.
KEEP_THRESHOLD = 0.95
# An unprotected prompt must lose all of its cache hits after the flood.
EVICT_THRESHOLD = 0.0

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega quantum vector tensor "
    "matrix gradient manifold topology lattice kernel signal photon proton "
    "neutron electron quark gluon plasma fusion fission orbit comet nebula "
    "galaxy pulsar quasar meteor asteroid eclipse horizon zenith nadir tundra "
    "savanna prairie canyon mesa plateau fjord archipelago lagoon reef river "
    "glacier volcano earthquake tornado hurricane monsoon blizzard"
).split()


def _make_prompt(seed: int) -> str:
    """Build a deterministic but unique long prompt.

    The random salt guarantees every prompt has a distinct prefix, so each
    request inserts fresh cache entries instead of sharing the radix tree.
    """
    rng = random.Random(seed)
    salt = uuid.uuid4().hex
    body = " ".join(rng.choice(_WORDS) for _ in range(WORDS_PER_PROMPT))
    return f"[{salt}] {body}"


class TestSessionRadixCacheE2E(CustomTestCase):
    """Testcase: Verify set the parameter --enable-session-radix-cache,Prioritize evicting KV cache that are not
     referenced by other sessions during eviction. Set the parameter --model-checksum, the model weights will be verified.

    [Test Category] Parameter
    [Test Target] --enable-session-radix-cache, --model-checksum
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--context-length",
            str(MAX_TOTAL_TOKENS),
            "--max-total-tokens",
            str(MAX_TOTAL_TOKENS),
            "--enable-session-radix-cache",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.6",
            "--model-checksum",
            "Qwen/Qwen3-0.6B",
        ]
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, text, session_id=None):
        # One generate request; the session id registers the request's cache
        # leaves under that session so they become protected from eviction.
        payload = {
            "text": text,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
        if session_id is not None:
            payload["session_id"] = session_id

        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        return meta["prompt_tokens"], meta.get("cached_tokens", 0)

    def _cached_ratio(self, text, session_id=None):
        prompt_tokens, cached_tokens = self._generate(
            text + " Continue.", session_id=session_id
        )
        self.assertGreater(prompt_tokens, 0)
        ratio = cached_tokens / prompt_tokens
        print(
            f"cached_ratio={ratio:.3f} "
            f"(cached={cached_tokens}, prompt={prompt_tokens}, session_id={session_id})"
        )
        return ratio

    def test_session_protection_and_release(self):
        session_id = f"e2e-session-{uuid.uuid4().hex[:8]}"
        prompt_a = _make_prompt(seed=1)
        prompt_b = _make_prompt(seed=2)

        # Seed both prompts: A under the session (protected), B without one
        # (unprotected control). Both must be cold on first request.
        a_seed = self._generate(prompt_a, session_id=session_id)
        b_seed = self._generate(prompt_b)

        self.assertEqual(a_seed[1], 0, "first request should have no cache hit")
        self.assertEqual(b_seed[1], 0, "first request should have no cache hit")

        # Flood the pool with unique prompts to create eviction pressure.
        for i in range(NUM_FLOOD_PROMPTS):
            self._generate(_make_prompt(seed=100 + i))

        # Unprotected B should have been evicted by the flood.
        b_ratio = self._cached_ratio(prompt_b)
        self.assertEqual(
            b_ratio,
            EVICT_THRESHOLD,
            f"unprotected prompt B should be evicted, cached_ratio={b_ratio:.3f}",
        )

        # Session-referenced A should survive the same flood.
        a_ratio = self._cached_ratio(prompt_a, session_id=session_id)
        self.assertGreaterEqual(
            a_ratio,
            KEEP_THRESHOLD,
            f"session-protected prompt A should survive, cached_ratio={a_ratio:.3f}",
        )

        # Closing the session drops the protection references on A's leaves.
        response = requests.post(
            f"{self.base_url}/close_session",
            json={"session_id": session_id},
            timeout=60,
        )
        response.raise_for_status()

        # A second flood should now evict A like any unprotected entry.
        for i in range(NUM_FLOOD_PROMPTS):
            self._generate(_make_prompt(seed=200 + i))

        a_ratio_after = self._cached_ratio(prompt_a)
        self.assertEqual(
            a_ratio_after,
            EVICT_THRESHOLD,
            "prompt A should be evicted after close_session, "
            f"cached_ratio={a_ratio_after:.3f}",
        )

    def test_model_checksum(self):
        # Model Weight File Verification
        self.out_file.seek(0)
        content = self.out_file.read()
        self.assertIn("ModelFileVerifier", content)


if __name__ == "__main__":
    unittest.main()
