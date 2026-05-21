"""
Test KV Events with HiCache on Ascend NPU.

Tests that KV cache events (BlockStored) are correctly emitted when
HiCache stores KV blocks to file storage.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_kv_events.py -v
"""

import shutil
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHiCacheKVEvents(CustomTestCase):
    """Test KV Events with HiCache on NPU.

    [Test Category] HiCache
    [Test Target] KV events emission with HiCache file storage
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="npu-hicache-kv-events-")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "1.2",
                "--hicache-size",
                "0",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-io-backend",
                "kernel_ascend",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--kv-events-config",
                '{"publisher": "zmq", "topic": "kv-events"}',
            ],
            env={
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def test_kv_events_smoke(self):
        """Test that KV events are emitted when HiCache stores blocks."""
        try:
            import zmq
            from msgspec.msgpack import Decoder

            from sglang.srt.disaggregation.kv_events import BlockStored, KVEventBatch
        except ImportError:
            self.skipTest("zmq or msgspec not available for KV events test")

        decoder = Decoder(type=KVEventBatch)
        context = zmq.Context()
        sub = context.socket(zmq.SUB)
        sub.connect("tcp://localhost:5557")
        sub.setsockopt_string(zmq.SUBSCRIBE, "kv-events")

        try:
            time.sleep(1.0)

            prompt = "HiCache KV event compatibility check on NPU. " * 64
            res = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 10},
                },
                timeout=120,
            )
            res.raise_for_status()

            events = []
            deadline = time.time() + 10
            while time.time() < deadline and not any(
                isinstance(event, BlockStored) for event in events
            ):
                if sub.poll(timeout=100):
                    _, _, payload = sub.recv_multipart()
                    events.extend(decoder.decode(payload).events)

            self.assertTrue(
                any(isinstance(event, BlockStored) for event in events),
                "Expected at least one BlockStored event from NPU HiCache server",
            )
        finally:
            sub.close()
            context.term()


if __name__ == "__main__":
    unittest.main()
