import os
import tempfile
import unittest
from pathlib import Path

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestNPUDumperHttp(CustomTestCase):
    """Test /dumper/* HTTP control on NPU.

    [Test Category] Feature
    [Test Target] Dumper HTTP API on NPU backend
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.dumper_url = "http://127.0.0.1:30001"
        cls.dump_dir = tempfile.mkdtemp(prefix="npu_dumper_test_")
        # Use a dedicated port for dumper HTTP server to avoid conflicts with detokenizer
        env = {**os.environ, "DUMPER_SERVER_PORT": "30001"}
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.3",
                "--max-total-tokens",
                "128",
                "--skip-server-warmup",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, method: str, **kwargs):
        resp = requests.post(f"{self.dumper_url}/dumper/{method}", json=kwargs or None)
        resp.raise_for_status()
        states = resp.json()
        self.assertIsInstance(states, list)
        self.assertGreaterEqual(len(states), 1)
        return states

    def _assert_all_ranks(self, states, path: str, expected):
        keys = path.split(".")
        for rank, state in enumerate(states):
            val = state
            for k in keys:
                val = val[k]
            self.assertEqual(
                val, expected, f"rank {rank}: {path}={val!r}, expected {expected!r}"
            )

    def test_configure_enable_toggle(self):
        for enable in [True, False]:
            self._post("configure", enable=enable)
            states = self._post("get_state")
            self._assert_all_ranks(states, "config.enable", enable)

    def test_configure_multi_field(self):
        self._post(
            "configure",
            enable=True,
            filter="layer_id == 0",
            dir="/tmp/test_npu_http",
        )
        states = self._post("get_state")
        self._assert_all_ranks(states, "config.enable", True)
        self._assert_all_ranks(states, "config.filter", "layer_id == 0")
        self._assert_all_ranks(states, "config.dir", "/tmp/test_npu_http")

    def test_configure_clear_optional(self):
        self._post("configure", filter="layer_id == 0")
        self._post("configure", filter=None)
        states = self._post("get_state")
        self._assert_all_ranks(states, "config.filter", None)

    def test_reset(self):
        self._post("configure", enable=True)
        self._post("reset")
        states = self._post("get_state")
        self._assert_all_ranks(states, "dump_index", 0)
        self._assert_all_ranks(states, "step", 0)

    def test_get_state(self):
        self._post(
            "configure",
            enable=True,
            filter="layer_id is not None and layer_id < 3",
        )
        states = self._post("get_state")
        self._assert_all_ranks(states, "config.enable", True)
        self._assert_all_ranks(
            states, "config.filter", "layer_id is not None and layer_id < 3"
        )
        for state in states:
            self.assertIn("dump_index", state)
            self.assertIn("step", state)

    def test_all_ranks_consistent(self):
        self._post("configure", enable=True, dir="/tmp/npu_multi")
        states = self._post("get_state")
        configs = [s["config"] for s in states]
        for rank_config in configs[1:]:
            self.assertEqual(
                rank_config, configs[0], f"rank configs diverged: {configs}"
            )

    def test_error_unknown_field(self):
        resp = requests.post(
            f"{self.base_url}/dumper/configure",
            json={"nonexistent_field": 123},
        )
        self.assertEqual(resp.status_code, 400)

    def test_error_unknown_method(self):
        resp = requests.post(
            f"{self.base_url}/dumper/nonexistent",
            json={},
        )
        self.assertEqual(resp.status_code, 400)

    def test_error_wrong_type(self):
        resp = requests.post(
            f"{self.base_url}/dumper/configure",
            json={"enable": "not_a_bool"},
        )
        self.assertEqual(resp.status_code, 400)


class TestNPUDumperE2E(CustomTestCase):
    """Test dumper functionality end-to-end on NPU.

    [Test Category] Feature
    [Test Target] Dumper step tracking and non-intrusive hooks on NPU
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.dumper_url = "http://127.0.0.1:30001"
        cls.dump_dir = tempfile.mkdtemp(prefix="npu_dumper_e2e_")
        # Use a dedicated port for dumper HTTP server to avoid conflicts with detokenizer
        env = {**os.environ, "DUMPER_SERVER_PORT": "30001"}
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.3",
                "--max-total-tokens",
                "128",
                "--skip-server-warmup",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_step_and_non_intrusive_hooks(self):
        states = requests.post(f"{self.base_url}/dumper/get_state", json={}).json()
        self.assertEqual(len(states), 2, f"Expected 2 ranks (tp=2), got {len(states)}")
        for state in states:
            self.assertFalse(state["config"]["enable"])
            self.assertEqual(state["step"], 0)

        requests.post(
            f"{self.base_url}/dumper/configure",
            json={"enable": True, "dir": self.dump_dir},
        ).raise_for_status()

        states = requests.post(f"{self.base_url}/dumper/get_state", json={}).json()
        self.assertEqual(len(states), 2)
        for rank, state in enumerate(states):
            self.assertTrue(
                state["config"]["enable"],
                f"rank {rank}: enable should be True after configure",
            )
            self.assertEqual(state["config"]["dir"], self.dump_dir)

        resp = requests.post(
            f"{self.base_url}/generate",
            json={"text": "Hello", "sampling_params": {"max_new_tokens": 8}},
        )
        self.assertEqual(resp.status_code, 200, f"Generate failed: {resp.text}")

        states = requests.post(f"{self.base_url}/dumper/get_state", json={}).json()
        self.assertEqual(len(states), 2)
        steps = [s["step"] for s in states]
        for rank, step in enumerate(steps):
            self.assertGreater(step, 0, f"rank {rank}: step should be > 0, got {step}")
        self.assertEqual(steps[0], steps[1], f"step mismatch across ranks: {steps}")

        dump_files = list(Path(self.dump_dir).glob("dump_*/*.pt"))
        self.assertGreater(len(dump_files), 0, f"No dump files in {self.dump_dir}")
        filenames = {f.name for f in dump_files}

        for field in ("input_ids", "positions", "rids"):
            self.assertTrue(
                any(f"name={field}" in f for f in filenames),
                f"Missing {field} dump from non-intrusive hooks, "
                f"got: {sorted(filenames)[:10]}",
            )

        for rank in range(2):
            self.assertTrue(
                any(f"rank={rank}" in f for f in filenames),
                f"No dump files for rank {rank}",
            )

        sample_file = dump_files[0]
        loaded = torch.load(sample_file, map_location="cpu", weights_only=False)
        self.assertIsInstance(loaded, dict, f"Expected dict, got {type(loaded)}")
        self.assertIn("value", loaded)
        self.assertIn("meta", loaded)
        self.assertIn("name", loaded["meta"])
        self.assertIn("rank", loaded["meta"])
        self.assertIn("step", loaded["meta"])

        par = loaded["meta"].get("sglang_parallel_info", {})
        expected_keys = [
            "tp_rank",
            "tp_size",
            "pp_rank",
            "pp_size",
            "moe_ep_rank",
            "moe_ep_size",
            "moe_tp_rank",
            "moe_tp_size",
            "moe_dp_rank",
            "moe_dp_size",
            "enable_dp_attention",
            "attn_tp_rank",
            "attn_tp_size",
            "attn_dp_rank",
            "attn_dp_size",
            "local_attn_dp_rank",
            "local_attn_dp_size",
            "attn_cp_rank",
            "attn_cp_size",
        ]
        for key in expected_keys:
            self.assertIn(
                key, par, f"Missing {key} in sglang_parallel_info, got: {sorted(par)}"
            )

        rids_files = [f for f in dump_files if "name=rids" in f.name]
        rids_loaded = torch.load(rids_files[0], map_location="cpu", weights_only=False)
        rids_value = rids_loaded["value"]
        self.assertIsInstance(
            rids_value, list, f"rids should be a list, got {type(rids_value)}"
        )
        self.assertGreater(len(rids_value), 0, "rids should be non-empty")
        self.assertTrue(
            all(isinstance(r, str) for r in rids_value),
            f"each rid should be a str, got {[type(r) for r in rids_value]}",
        )


if __name__ == "__main__":
    unittest.main()
