import asyncio
import os
import unittest

import aiohttp

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestNPUDisaggregationPauseResumePrefillLeak(PDDisaggregationServerBase):
    """Test pause/resume leak prevention in disaggregation mode on NPU.

    [Test Category] Functional
    [Test Target] Pause generation request leak prevention
    """

    MAX_RUNNING = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24667"
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        # Use ascend transfer backend for NPU
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL", None)
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            str(cls.MAX_RUNNING),
            "--enable-metrics",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "0",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_retract_pause_no_leak_on_prefill(self):
        asyncio.run(self._run_pause_resume_leak_test("retract"))

    def test_retract_pause_empty_running_batch(self):
        asyncio.run(self._run_pause_on_idle("retract"))

    async def _run_pause_on_idle(self, mode):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.prefill_url + "/pause_generation",
                json={"mode": mode},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
            async with session.post(
                self.prefill_url + "/continue_generation",
                json={},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()

            async with session.post(
                self.lb_url + "/generate",
                json={
                    "text": "What is 1+1?",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
                self.assertIn("text", body)
                self.assertGreater(len(body["text"]), 0)

    async def _get_num_running_reqs(self, session):
        async with session.get(
            self.prefill_url + "/metrics",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
            for line in text.splitlines():
                if (
                    line.startswith("sglang:num_running_reqs{")
                    and "priority=" not in line
                ):
                    return int(float(line.split()[-1]))
            return 0

    async def _run_pause_resume_leak_test(self, mode):
        NUM_WORKERS = 64
        NUM_PAUSE_RESUME_CYCLES = self.MAX_RUNNING * 4
        MAX_NEW_TOKENS = 1
        LONG_PROMPT = "Tell me a story. " * 200

        async def _background_worker(session, worker_id, cancel_event):
            seq = 0
            while not cancel_event.is_set():
                try:
                    async with session.post(
                        self.lb_url + "/generate",
                        json={
                            "text": f"[w{worker_id}-{seq}] {LONG_PROMPT}",
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": MAX_NEW_TOKENS,
                            },
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        await resp.read()
                except Exception:
                    pass
                seq += 1

        async def _post(session, url, json_data):
            async with session.post(
                url,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()

        cancel_event = asyncio.Event()

        async with aiohttp.ClientSession() as session:
            workers = [
                asyncio.create_task(_background_worker(session, i, cancel_event))
                for i in range(NUM_WORKERS)
            ]

            for _ in range(NUM_PAUSE_RESUME_CYCLES):
                await _post(
                    session,
                    self.prefill_url + "/pause_generation",
                    {"mode": mode},
                )
                await _post(
                    session,
                    self.prefill_url + "/continue_generation",
                    {},
                )
                await asyncio.sleep(0.1)

            cancel_event.set()
            await _post(
                session, self.prefill_url + "/abort_request", {"abort_all": True}
            )
            await _post(
                session, self.decode_url + "/abort_request", {"abort_all": True}
            )
            await asyncio.gather(*workers, return_exceptions=True)

            await asyncio.sleep(2)
            num_running = await self._get_num_running_reqs(session)
            self.assertEqual(
                num_running,
                0,
                f"Prefill node has {num_running} phantom running requests "
                f"after abort — pause_generation is leaking into running_batch",
            )


if __name__ == "__main__":
    unittest.main()
