import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="debug-full-1-npu-a3", nightly=True)


class TestCompletionMisc(CustomTestCase):
    """Testcase: user / rid / logit_bias / priority on /v1/completions.

    [Test Category] Interface
    [Test Target] user / rid / logit_bias / priority
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
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, payload):
        return requests.post(
            f"{self.base_url}/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _payload(self, **kwargs):
        payload = {
            "model": self.model,
            "prompt": "Say hi.",
            "temperature": 0,
            "max_tokens": 16,
        }
        payload.update(kwargs)
        return payload

    def test_user_ignored(self):
        baseline = self._post(self._payload()).json()
        with_user = self._post(self._payload(user="test-user")).json()
        self.assertEqual(
            baseline["choices"][0]["text"],
            with_user["choices"][0]["text"],
            "user is not consumed by the engine",
        )

    def test_rid_batch_expansion(self):
        """Batch prompt + str rid: _normalize_rid expands to rid_0/rid_1."""
        response = self._post(
            self._payload(
                prompt=["The capital of France is", "The capital of Germany is"],
                rid="batch",
            )
        )
        self.assertEqual(response.status_code, 200, response.text)
        # Non-streaming response id comes from the first sub-request.
        self.assertEqual(response.json()["id"], "batch_0")

    def test_rid_duplicate_concurrent(self):
        result = {}

        def run_long():
            response = self._post(
                self._payload(
                    prompt="Write a very long story.", max_tokens=500, rid="dup1"
                )
            )
            result["status"] = response.status_code

        thread = threading.Thread(target=run_long)
        thread.start()
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                loads = requests.get(f"{self.base_url}/loads", timeout=5).json()
                total = sum(
                    rank.get("num_running_reqs", 0) for rank in loads.get("loads", [])
                )
                if total > 0:
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

    def test_logit_bias_force_eos(self):
        """Positive bias on the EOS token: the first sampled token is EOS."""
        eos_id = self.tokenizer.eos_token_id
        response = self._post(
            self._payload(logit_bias={str(eos_id): 42}, max_tokens=64)
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["usage"]["completion_tokens"], 1)
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")

    def test_priority_ignored_without_flag(self):
        baseline = self._post(self._payload()).json()
        with_priority = self._post(self._payload(priority=10)).json()
        self.assertEqual(
            baseline["choices"][0]["text"],
            with_priority["choices"][0]["text"],
            "priority is silently ignored without --enable-priority-scheduling",
        )


class TestCompletionCustomLabels(CustomTestCase):
    """Testcase: custom_labels on /v1/completions.

    The body field is dead code — only the X-Custom-Labels HTTP header is
    read, filtered by --tokenizer-metrics-allowed-custom-labels.

    [Test Category] Interface
    [Test Target] custom_labels
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
                "--enable-metrics",
                "--tokenizer-metrics-allowed-custom-labels",
                "tenant",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _metrics_labels(self):
        metrics = requests.get(
            f"{self.base_url.replace('/v1', '')}/metrics", timeout=10
        ).text
        return metrics

    def _post_completion(self, payload, headers=None):
        return requests.post(
            f"{self.base_url}/completions",
            json=payload,
            headers=headers or {"Authorization": f"Bearer {self.api_key}"},
        )

    def test_custom_labels_header_takes_effect(self):
        payload = {
            "model": self.model,
            "prompt": "Say hi.",
            "temperature": 0,
            "max_tokens": 16,
        }
        response = self._post_completion(
            payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "X-Custom-Labels": '{"tenant":"acme"}',
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        # The label appears in the Prometheus metrics endpoint.
        deadline = time.time() + 30
        while time.time() < deadline:
            if 'tenant="acme"' in self._metrics_labels():
                break
            time.sleep(1)
        else:
            self.fail('tenant="acme" not found in /metrics')

    def test_custom_labels_body_dead_code(self):
        """The body field is silently dropped — the label never reaches metrics."""
        payload = {
            "model": self.model,
            "prompt": "Say hi.",
            "temperature": 0,
            "max_tokens": 16,
            "custom_labels": {"tenant": "body-only"},
        }
        response = self._post_completion(payload)
        self.assertEqual(response.status_code, 200, response.text)
        # Distinct label value avoids interference from the header test.
        self.assertNotIn(
            'tenant="body-only"',
            self._metrics_labels(),
            "body custom_labels must be ignored",
        )


if __name__ == "__main__":
    unittest.main()
