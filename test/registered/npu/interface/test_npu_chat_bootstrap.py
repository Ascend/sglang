import os
import sys
import threading
import time

import requests

# Allow importing the PD fixture from the pd_disaggregation directory when
# run directly by run_suite.py.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "basic_function", "pd_disaggregation"
    ),
)

from test_npu_disaggregation_basic import DisaggregationTestBase

from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=800, suite="full-2-npu-a3", nightly=True)


class TestChatBootstrapParams(DisaggregationTestBase):
    """Testcase: bootstrap_host/port/room in the request body.

    Posting the trio directly to the decode node bypasses the router: the decode
    node pulls the room's KV cache from the given bootstrap address. The body
    params are a KV-cache rendezvous contract, not a routing instruction.

    [Test Category] Interface
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

    def _fire_prefill(self, host, port, room):
        """Fire-and-forget the prefill request: the prefill node computes KV
        and registers the room, but in PD mode it has no generation response
        path — its HTTP request must NOT be awaited (a synchronous POST would
        hang until the client timeout, then the prefill aborts its own
        bootstrap with KVTransferError)."""

        def post():
            try:
                requests.post(
                    self.prefill_url + "/v1/chat/completions",
                    json=self._trio_payload(host, port, room),
                    timeout=120,
                )
            except Exception:
                # The prefill side may abort its own bootstrap after the KV
                # transfer — the decode response is all the test needs.
                pass

        threading.Thread(target=post, daemon=True).start()
        # Let the prefill compute KV and register the room before decode polls.
        time.sleep(1)

    def test_trio_completes(self):
        """Full trio: the prefill registers the room (fire-and-forget), then
        decode pulls KV from the bootstrap address and returns the response."""
        self._fire_prefill(self.base_host, self.bootstrap_port, 101)
        response = self._post_trio(self.base_host, self.bootstrap_port, 101)
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertTrue(content, "generation should produce content")

    def test_port_omitted_defaults(self):
        """port is the only optional member: it defaults to
        --disaggregation-bootstrap-port."""
        self._fire_prefill(self.base_host, None, 102)
        response = self._post_trio(self.base_host, None, 102)
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertTrue(content, "generation should produce content")

    def test_room_omitted_400(self):
        """room is mandatory in real PD mode: omitted → 400."""
        response = self._post_trio(self.base_host, self.bootstrap_port, None)
        self.assertEqual(response.status_code, 400, response.text)
        # The abort path may serialize the message under "error" or "detail";
        # assert on the raw text to stay format-agnostic.
        self.assertIn("bootstrap room id", response.text)

    # NOTE: host-omitted is intentionally NOT tested: NetworkAddress(None, port)
    # raises AttributeError in __post_init__ (network.py:456-459) inside the
    # decode scheduler event loop, which has no try/except — the scheduler
    # thread dies and every subsequent request on the server hangs.
    # A graceful 4xx/5xx validation for a missing bootstrap_host should be
    # added server-side first (decode.py:581).


if __name__ == "__main__":
    import unittest

    unittest.main()
