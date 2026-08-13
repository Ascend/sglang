import requests

from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

from test_npu_disaggregation_basic import DisaggregationTestBase

register_npu_ci(est_time=800, suite="debug-full-2-npu-a3", nightly=True)


class TestChatBootstrapParams(DisaggregationTestBase):
    """Testcase: bootstrap_host/port/room in the request body.

    Posting the trio directly to the decode node bypasses the router: the decode
    node pulls the room's KV cache from the given bootstrap address. The body
    params are a KV-cache rendezvous contract, not a routing instruction.

    [Test Category] Parameter
    [Test Target] bootstrap_host / bootstrap_port / bootstrap_room
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

    def _trio_payload(self, host, port, room):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "temperature": 0,
            "max_tokens": 16,
            "bootstrap_room": room,
        }
        if host is not None:
            payload["bootstrap_host"] = host
        if port is not None:
            payload["bootstrap_port"] = port
        return payload

    def _post_trio(self, host, port, room):
        return requests.post(
            self.decode_url + "/v1/chat/completions",
            json=self._trio_payload(host, port, room),
            timeout=120,
        )

    def _warm_prefill(self, host, port, room):
        """Register the room on the prefill side first: the prefill node
        computes KV and announces it at bootstrap_host:port for this room."""
        return requests.post(
            self.prefill_url + "/v1/chat/completions",
            json=self._trio_payload(host, port, room),
            timeout=120,
        )

    def test_trio_completes(self):
        """Full trio: after the prefill registers the room, decode pulls KV
        from the bootstrap address and completes."""
        self._warm_prefill(self.base_host, self.bootstrap_port, 101)
        response = self._post_trio(self.base_host, self.bootstrap_port, 101)
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertTrue(content, "generation should produce content")

    def test_port_omitted_defaults(self):
        """port is the only optional member: it defaults to
        --disaggregation-bootstrap-port."""
        self._warm_prefill(self.base_host, None, 102)
        response = self._post_trio(self.base_host, None, 102)
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertTrue(content, "generation should produce content")

    def test_room_omitted_400(self):
        """room is mandatory in real PD mode: omitted → 400."""
        response = self._post_trio(self.base_host, self.bootstrap_port, None)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("bootstrap room id", response.json()["error"]["message"])

    # NOTE: host-omitted is intentionally NOT tested: NetworkAddress(None, port)
    # raises AttributeError in __post_init__ (network.py:456-459) inside the
    # decode scheduler event loop, which has no try/except — the scheduler
    # thread dies and every subsequent request on the server hangs.
    # A graceful 4xx/5xx validation for a missing bootstrap_host should be
    # added server-side first (decode.py:581).


if __name__ == "__main__":
    import unittest

    unittest.main()
