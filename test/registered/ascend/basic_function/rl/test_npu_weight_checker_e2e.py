import base64
import io
import pickle
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=150, suite="nightly-2-npu-a3", nightly=True)

_MODEL_NAME = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
_UP_PROJ_SHAPE = (3072, 1024)


class TestNPUWeightCheckerE2E(CustomTestCase):
    """Test weights checker HTTP endpoint on NPU.

    [Test Category] RL Weight Checker
    [Test Target] /weights_checker endpoint
    """

    @classmethod
    def setUpClass(cls):
        cls.url = "http://127.0.0.1:30001"
        cls.process = popen_launch_server(
            _MODEL_NAME,
            cls.url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mem-fraction-static",
                "0.7",
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        try:
            kill_process_tree(cls.process.pid)
        except Exception:
            pass  # Ignore cleanup errors

    def _post(self, action: str) -> requests.Response:
        return requests.post(
            f"{self.url}/weights_checker", json={"action": action}, timeout=120
        )

    def _update_weights(self, named_tensors):
        # Use ForkingPickler and base64 encoding to match server-side deserialize
        buf = io.BytesIO()
        pickle.dump(named_tensors, buf)
        buf.seek(0)
        serialized = base64.b64encode(buf.read()).decode("utf-8")
        return requests.post(
            f"{self.url}/update_weights_from_tensor",
            json={
                "serialized_named_tensors": [serialized],
                "flush_cache": True,
            },
            timeout=120,
        )

    def test_a_snapshot_then_compare_unchanged_succeeds(self):
        resp = self._post("snapshot")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

        resp = self._post("compare")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_b_unknown_action_returns_400(self):
        resp = self._post("nonsense_action")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported", resp.json()["message"])

    def test_c_update_with_diff_tensor_makes_compare_fail(self):
        self.assertEqual(self._post("snapshot").status_code, 200)

        upload_name = "model.layers.5.mlp.up_proj.weight"
        new_tensor = torch.full(_UP_PROJ_SHAPE, 1.5)
        update_resp = self._update_weights([(upload_name, new_tensor)])
        self.assertEqual(update_resp.status_code, 200)
        self.assertTrue(update_resp.json()["success"])

        resp = self._post("compare")
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertFalse(body["success"])
        self.assertIn("model.layers.5.mlp.gate_up_proj.weight", body["message"])
        self.assertIn("max_abs_err", body["message"])

    def test_d_update_with_same_tensor_keeps_compare_passing(self):
        param_name = "model.layers.6.mlp.up_proj.weight"
        same_tensor = torch.full(_UP_PROJ_SHAPE, 0.25)

        self.assertTrue(
            self._update_weights([(param_name, same_tensor)]).json()["success"]
        )
        self.assertEqual(self._post("snapshot").status_code, 200)
        self.assertTrue(
            self._update_weights([(param_name, same_tensor)]).json()["success"]
        )
        resp = self._post("compare")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

    def test_e_checksum_returns_ranks_with_hashes(self):
        resp = self._post("checksum")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["success"])
        self.assertIn("ranks", body)
        ranks = body["ranks"]
        self.assertIsInstance(ranks, list)
        self.assertGreaterEqual(len(ranks), 1)

        first = ranks[0]
        self.assertIn("checksums", first)
        self.assertIn("parallelism_info", first)

        info = first["parallelism_info"]
        for key in (
            "tp_rank",
            "tp_size",
            "dp_rank",
            "dp_size",
            "pp_rank",
            "pp_size",
            "rank",
            "size",
        ):
            self.assertIn(key, info)

        checksums = first["checksums"]
        self.assertGreater(len(checksums), 0)
        for name, h in checksums.items():
            self.assertIsInstance(h, str)
            self.assertEqual(len(h), 16, f"unexpected hash length for {name!r}: {h!r}")

    def test_e_checksum_is_stable_across_calls(self):
        first = self._post("checksum").json()["ranks"]
        second = self._post("checksum").json()["ranks"]
        self.assertEqual(first, second)

    def test_e_checksum_changes_after_weight_update(self):
        param_name = "model.layers.7.mlp.up_proj.weight"
        fused_name = "model.layers.7.mlp.gate_up_proj.weight"

        before = self._post("checksum").json()["ranks"][0]["checksums"]
        before_hash = before.get(fused_name)
        self.assertIsNotNone(before_hash, f"missing {fused_name!r} in checksum keys")

        new_tensor = torch.full(_UP_PROJ_SHAPE, 0.5)
        self.assertTrue(
            self._update_weights([(param_name, new_tensor)]).json()["success"]
        )

        after = self._post("checksum").json()["ranks"][0]["checksums"]
        self.assertNotEqual(after[fused_name], before_hash)

    def test_z_snapshot_reset_compare_detects_diff(self):
        self.assertEqual(self._post("snapshot").status_code, 200)
        self.assertEqual(self._post("reset_tensors").status_code, 200)

        resp = self._post("compare")
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertFalse(body["success"])
        self.assertIn("max_abs_err", body["message"])


if __name__ == "__main__":
    unittest.main()
