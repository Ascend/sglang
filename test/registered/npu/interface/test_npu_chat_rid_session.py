import json
import re
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="debug-full-1-npu-a3", nightly=True)

HEX32 = re.compile(r"^[0-9a-f]{32}$")


class TestChatRid(CustomTestCase):
    """Testcase: rid behavior on /v1/chat/completions.

    [Test Category] Interface
    [Test Target] rid (auto-generation / n>1 regeneration / stream echo / duplicate rejection)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _payload(self, **kwargs):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "temperature": 0,
            "max_tokens": 16,
        }
        payload.update(kwargs)
        return payload

    def test_rid_auto_generated_32hex(self):
        response = self._post(self._payload())
        self.assertEqual(response.status_code, 200, response.text)
        rid = response.json()["id"]
        self.assertRegex(rid, HEX32, f"auto-generated rid should be 32-hex: {rid}")

    def test_rid_n2_regenerated(self):
        """n>1: regenerate_rid() overwrites the expanded rids with fresh UUIDs,
        so the client-supplied rid never appears in the response."""
        response = self._post(self._payload(n=2, rid="req"))
        self.assertEqual(response.status_code, 200, response.text)
        rid = response.json()["id"]
        self.assertRegex(
            rid, HEX32, f"n>1 rid should be a regenerated UUID, got: {rid}"
        )

    def test_rid_stream_echo(self):
        """Stream: every chunk echoes the client-supplied rid (n=1)."""
        response = self._post(self._payload(stream=True, rid="sssss"))
        ids = []
        for line in response.iter_lines():
            if line and line.startswith(b"data: ") and line != b"data: [DONE]":
                ids.append(json.loads(line[6:])["id"])
        self.assertTrue(ids, "no chunks received")
        self.assertTrue(
            all(rid == "sssss" for rid in ids), f"chunk ids should echo rid: {ids}"
        )

    def test_rid_duplicate_concurrent(self):
        """A second request reusing a running request's rid is rejected with 400."""
        long_payload = self._payload(
            messages=[{"role": "user", "content": "Write a very long story."}],
            max_tokens=500,
            rid="dup1",
        )
        result = {}

        def run_long():
            response = self._post(long_payload)
            result["status"] = response.status_code

        thread = threading.Thread(target=run_long)
        thread.start()
        # Wait until the long request is actually running before reusing its rid.
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                loads = requests.get(f"{self.base_url}/loads", timeout=5).json()
                if loads.get("aggregate", {}).get("total_running_reqs", 0) > 0:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            self.fail("long request never became running")

        response = self._post(self._payload(rid="dup1"))
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(
            "Duplicate request ID detected", response.json()["error"]["message"]
        )

        thread.join()
        self.assertEqual(result["status"], 200)


class TestChatCacheKeyIsolation(CustomTestCase):
    """Testcase: extra_key / cache_salt isolation via cached_tokens.

    The final cache key is cache_salt + extra_key (no separator); any change
    isolates the radix cache entry.

    [Test Category] Interface
    [Test Target] extra_key / cache_salt
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
                "--enable-cache-report",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, payload):
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return response.json()

    def _prompt_payload(self, **kwargs):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the theory of relativity in one paragraph.",
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
        }
        payload.update(kwargs)
        return payload

    def _cached_tokens(self, payload):
        response = self._post(payload)
        return response["usage"]["prompt_tokens_details"]["cached_tokens"]

    def test_extra_key_isolation(self):
        # Warm the cache with extra_key="a" (fixed cache_salt="s1").
        warm = self._prompt_payload(cache_salt="s1", extra_key="a")
        self._post(warm)
        self.assertGreater(self._cached_tokens(warm), 0, "same key should hit")

        # Different extra_key → cache isolation.
        different = self._prompt_payload(cache_salt="s1", extra_key="b")
        self.assertEqual(
            self._cached_tokens(different),
            0,
            "different extra_key should not hit the cache",
        )

    def test_cache_salt_isolation(self):
        # Warm with cache_salt="a" (fixed extra_key="e1").
        warm = self._prompt_payload(cache_salt="a", extra_key="e1")
        self._post(warm)
        self.assertGreater(self._cached_tokens(warm), 0, "same salt should hit")

        # Different cache_salt → cache isolation.
        different = self._prompt_payload(cache_salt="b", extra_key="e1")
        self.assertEqual(
            self._cached_tokens(different),
            0,
            "different cache_salt should not hit the cache",
        )


class TestChatSessionRadixCache(CustomTestCase):
    """Testcase: session_id enables session KV retention under the session radix cache.

    [Test Category] Interface
    [Test Target] session_id
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--attention-backend",
                "ascend",
                "--enable-session-radix-cache",
                "--radix-eviction-policy",
                "priority",
                "--enable-cache-report",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_session_id_cache_hit(self):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the theory of relativity in one paragraph.",
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "session_id": "sess-1",
        }
        first = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ).json()
        second = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ).json()

        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        self.assertGreater(cached, 0, "second request with same session_id should hit cache")


if __name__ == "__main__":
    unittest.main()
