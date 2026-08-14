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

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)


class TestChatDeadParams(CustomTestCase):
    """Testcase: accepted-but-ignored parameters (suspected dead code).

    Each case pins the current silent-ignore contract: the parameter is accepted
    by Pydantic, never forwarded/consumed, and must not change the output.

    [Test Category] Interface
    [Test Target] session_params / user / best_of (dead params)
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

    def _post(self, endpoint, payload):
        return requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def test_session_params_silently_ignored(self):
        """serving_chat.py never reads request.session_params — the field is
        silently dropped (only /generate forwards top-level session_params)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "temperature": 0,
            "max_tokens": 16,
        }
        baseline = self._post("/chat/completions", payload).json()

        payload["session_params"] = {
            "id": "sess-x",
            "rid": "rid-x",
            "offset": 1,
        }
        with_params = self._post("/chat/completions", payload).json()

        self.assertEqual(
            baseline["choices"][0]["message"]["content"],
            with_params["choices"][0]["message"]["content"],
            "session_params should be silently ignored (not forwarded to engine)",
        )

    def test_user_silently_ignored(self):
        """The engine never consumes `user` — output must be identical with/without it."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "temperature": 0,
            "max_tokens": 16,
        }
        baseline = self._post("/chat/completions", payload).json()

        payload["user"] = "test-user"
        with_user = self._post("/chat/completions", payload).json()

        self.assertEqual(
            baseline["choices"][0]["message"]["content"],
            with_user["choices"][0]["message"]["content"],
            "user should be silently ignored (engine does not consume it)",
        )

    def test_best_of_silently_ignored(self):
        """best_of is defined on CompletionRequest but never read by
        serving_completions.py — the user gets 1 choice silently."""
        payload = {
            "model": self.model,
            "prompt": "The capital of France is",
            "temperature": 0,
            "max_tokens": 16,
            "best_of": 3,
        }
        response = self._post("/completions", payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            len(response.json()["choices"]),
            1,
            "best_of is ignored: exactly one choice is returned",
        )


if __name__ == "__main__":
    unittest.main()
