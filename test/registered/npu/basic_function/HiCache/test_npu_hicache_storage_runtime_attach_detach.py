"""NPU adaptation of test_hicache_storage_runtime_attach_detach.py."""

import json
import logging
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(
    est_time=500,
    suite="stage-b-test-4-npu-a3",
    nightly=False,
)

ADMIN_API_KEY = "admin-hicache-test-key"


def _common_hicache_args(port: int) -> list:
    """HiCache args shared by Phase A and Phase B (no storage backend args).

    `--hicache-storage-backend` is intentionally not set here: the runtime
    attach/detach test must start with NO backend attached.
    """
    return [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.6",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--page-size",
        "128",
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "1.2",
        "--hicache-size",
        "100",
        "--enable-cache-report",
        "--tp-size",
        "2",
        "--port",
        str(port),
    ]


class TestHiCacheStorageRuntimeAttachDetach(CustomTestCase):
    """Verify runtime attach/detach of the HiCache storage backend + admin auth."""

    # ---------- HTTP helpers ----------

    @staticmethod
    def _get_backend(base_url, headers=None):
        return requests.get(
            f"{base_url}/hicache/storage-backend",
            headers=headers,
            timeout=60,
        )

    @staticmethod
    def _attach_backend(
        base_url,
        backend,
        extra_config=None,
        prefetch_policy=None,
        write_policy=None,
        headers=None,
    ):
        """PUT /hicache/storage-backend.

        Field names must match `AttachHiCacheStorageReqInput`:
          - hicache_storage_backend (str, required)
          - hicache_storage_backend_extra_config_json (JSON string)
          - hicache_storage_prefetch_policy (str)
          - hicache_write_policy (str)
        """
        payload = {"hicache_storage_backend": backend}
        if extra_config is not None:
            payload["hicache_storage_backend_extra_config_json"] = json.dumps(
                extra_config
            )
        if prefetch_policy is not None:
            payload["hicache_storage_prefetch_policy"] = prefetch_policy
        if write_policy is not None:
            payload["hicache_write_policy"] = write_policy
        return requests.put(
            f"{base_url}/hicache/storage-backend",
            json=payload,
            headers=headers,
            timeout=60,
        )

    @staticmethod
    def _detach_backend(base_url, headers=None):
        return requests.delete(
            f"{base_url}/hicache/storage-backend",
            headers=headers,
            timeout=60,
        )

    @staticmethod
    def _admin_headers() -> dict:
        return {"Authorization": f"Bearer {ADMIN_API_KEY}"}

    # ---------- main test ----------

    def test_runtime_attach_detach(self):
        logging.warning("\n=== test_runtime_attach_detach ===")

        # ===================== Phase A: no admin-api-key =====================
        # Without --admin-api-key, the storage-backend endpoints must be
        # disabled (return 400) for ALL verbs.
        logging.warning("Phase A: launch server WITHOUT --admin-api-key")
        port_a = find_available_port(20000)
        base_url_a = f"http://127.0.0.1:{port_a}"
        process_a = popen_launch_server(
            LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
            base_url_a,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_common_hicache_args(port_a),
        )
        try:
            wait_for_http_ready(base_url_a, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

            # No admin header.
            self.assertEqual(self._get_backend(base_url_a).status_code, 400)
            self.assertEqual(self._attach_backend(base_url_a, "file").status_code, 400)
            self.assertEqual(self._detach_backend(base_url_a).status_code, 400)

            # Even with a Bearer header, endpoints must remain disabled
            # because the server itself was started without --admin-api-key.
            h = self._admin_headers()
            self.assertEqual(self._get_backend(base_url_a, headers=h).status_code, 400)
            self.assertEqual(
                self._attach_backend(base_url_a, "file", headers=h).status_code, 400
            )
            self.assertEqual(
                self._detach_backend(base_url_a, headers=h).status_code, 400
            )
        finally:
            kill_process_tree(process_a.pid)

        # ===================== Phase B: with --admin-api-key =====================
        logging.warning("Phase B: launch server WITH --admin-api-key")
        port_b = find_available_port(21000)
        base_url_b = f"http://127.0.0.1:{port_b}"
        other_args_b = _common_hicache_args(port_b) + [
            "--admin-api-key",
            ADMIN_API_KEY,
        ]
        process_b = popen_launch_server(
            LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
            base_url_b,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args_b,
        )
        try:
            wait_for_http_ready(base_url_b, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

            # 1) Without Authorization header -> 401 for every verb.
            self.assertEqual(self._get_backend(base_url_b).status_code, 401)
            self.assertEqual(self._attach_backend(base_url_b, "file").status_code, 401)
            self.assertEqual(self._detach_backend(base_url_b).status_code, 401)

            admin = self._admin_headers()

            # 2) Fresh server: no backend attached yet.
            resp = self._get_backend(base_url_b, headers=admin)
            self.assertEqual(resp.status_code, 200)
            self.assertIsNone(resp.json().get("hicache_storage_backend"))

            # 3) Attach `file` backend -> 200.
            resp = self._attach_backend(base_url_b, "file", headers=admin)
            self.assertEqual(resp.status_code, 200, resp.text)

            # 4) Try to switch to an unsupported backend (mooncake) -> non-200.
            resp = self._attach_backend(base_url_b, "mooncake", headers=admin)
            self.assertNotEqual(resp.status_code, 200)

            # 5) Detach -> 200, and detaching again is idempotent (still 200).
            resp = self._detach_backend(base_url_b, headers=admin)
            self.assertEqual(resp.status_code, 200, resp.text)
            resp = self._detach_backend(base_url_b, headers=admin)
            self.assertEqual(resp.status_code, 200, resp.text)

            # 6) After detach, GET should report backend == None again.
            resp = self._get_backend(base_url_b, headers=admin)
            self.assertEqual(resp.status_code, 200)
            self.assertIsNone(resp.json().get("hicache_storage_backend"))

            # 7) Re-attach `file` -> 200 (server still healthy after detach).
            resp = self._attach_backend(base_url_b, "file", headers=admin)
            self.assertEqual(resp.status_code, 200, resp.text)
        finally:
            kill_process_tree(process_b.pid)


if __name__ == "__main__":
    unittest.main()
