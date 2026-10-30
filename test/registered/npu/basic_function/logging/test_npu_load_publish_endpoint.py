import json
import socket
import unittest

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

# Fixed ZMQ port for the kv-events publisher; with --load-publish-endpoint=auto
# the load publisher port base is placed right after the kv port range.
KV_PORT = 5592


class TestNPULoadPublishEndpoint(CustomTestCase):
    """Testcase：Verify --load-publish-endpoint=auto on single NPU. The load PUB
    socket is bound right after the kv-events port range (kv_base + dp_size) and
    advertised via /server_info, so a ZMQ subscriber can discover it without
    extra configuration.

    [Test Category] Logging
    [Test Target] --load-publish-endpoint; --kv-events-config
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--kv-events-config",
            json.dumps(
                {
                    "publisher": "zmq",
                    "endpoint": f"tcp://*:{KV_PORT}",
                    "topic": "kv",
                }
            ),
            "--load-publish-endpoint",
            "auto",
        ]
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.addClassCleanup(kill_process_tree, cls.process.pid)

    def test_load_publish_endpoint_auto(self):
        """--load-publish-endpoint=auto: load PUB socket bound and advertised via /server_info."""
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info")
        self.assertEqual(response.status_code, 200)
        info = response.json()

        kv = info.get("kv_events")
        self.assertIsNotNone(
            kv, "kv_events descriptor missing (kv-events-config did not take effect)"
        )
        base = kv.get("load_endpoint_port_base")
        self.assertIsNotNone(base, f"load_endpoint_port_base missing: {kv}")
        self.assertEqual(kv.get("load_topic"), "load")
        # auto should place base after the kv port range: kv_base + dp_size
        self.assertEqual(base, KV_PORT + kv["dp_size"])
        for rank in range(kv["dp_size"]):
            self.assertTrue(
                port_listening(base + rank),
                f"load publisher rank {rank} port {base + rank} not listening",
            )


def port_listening(port, host="127.0.0.1", timeout=2):
    """Check whether a TCP port is listening (used for ZMQ PUB sockets)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


if __name__ == "__main__":
    unittest.main()
