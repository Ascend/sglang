"""NPU adaptation of test_hicache_storage_runtime_attach_detach.py.

Test target: verify the runtime HTTP endpoints that dynamically attach /
detach the HiCache storage backend, and the `--admin-api-key` based
authentication that gates those endpoints.

[Test Category] Functional
[Test Target] /hicache/storage-backend (GET / PUT / DELETE) + --admin-api-key

Key observation points ported from the GPU test:
  * Phase A - server launched WITHOUT `--admin-api-key`:
      - GET    /hicache/storage-backend  -> 400 (endpoints disabled)
      - PUT    /hicache/storage-backend  -> 400
      - DELETE /hicache/storage-backend  -> 400
  * Phase B - server launched WITH `--admin-api-key`:
      - GET/PUT/DELETE without `Authorization: Bearer <key>` -> 401
      - GET with admin header on a fresh server -> backend == None
      - PUT  attach `file` backend             -> 200
      - PUT  update prefetch policy            -> 200
      - PUT  switch to `mooncake` backend      -> non-200 (rejected)
      - DELETE detach                          -> 200 (and idempotent)
      - PUT  attach again after detach         -> 200

NPU adaptation notes (see report for the full rationale):
  * `--attention-backend ascend` and `--disable-cuda-graph` are mandatory
    on NPU and are added to every server launch.
  * `--mem-fraction-static` is bumped from 0.6 to 0.8 (NPU convention).
  * `--page-size` is forced to 128 (NPU only supports 128).
  * Model: LLAMA_3_2_1B_INSTRUCT (same as `test_npu_hicache.py`, light
    enough that we can launch / kill it twice within the test).
  * `find_available_port` and `popen_launch_server` are reused directly
    from `sglang.test.test_utils` - no NPU-specific helper is needed.
  * The HTTP request / response contract is identical to the GPU test:
    the storage-backend REST API is backend-agnostic, so the same status
    code sequence (400 / 401 / 200) is expected on NPU.
  * `mooncake` backend rejection: kept as-is. If NPU happens to ship a
    mooncake build that accepts attach, this assertion will surface it
    loudly; the rest of the test does not depend on it.
"""

import logging
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
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

    Note: neither `--hicache-storage-backend` nor
    `--hicache-storage-backend-extra-config` is set here, because the
    runtime attach/detach test must start with NO backend attached.
    """
    return [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
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
    def _attach_backend(base_url, backend, extra_cfg=None, headers=None):
        payload = {"backend": backend}
        if extra_cfg is not None:
            payload["extra_config"] = extra_cfg
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
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
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
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
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
            self.assertIsNone(resp.json().get("backend"))

            # 3) Attach `file` backend with an extra config -> 200.
            extra_cfg = {"hicache_storage_pass_prefix_keys": True}
            resp = self._attach_backend(
                base_url_b, "file", extra_cfg=extra_cfg, headers=admin
            )
            self.assertEqual(resp.status_code, 200, resp.text)

            # 4) Update prefetch policy on the attached backend -> 200.
            resp = self._attach_backend(
                base_url_b,
                "file",
                extra_cfg={"hicache_storage_prefetch_policy": "wait_complete"},
                headers=admin,
            )
            self.assertEqual(resp.status_code, 200, resp.text)

            # 5) Try to switch to an unsupported backend (mooncake) -> non-200.
            resp = self._attach_backend(base_url_b, "mooncake", headers=admin)
            self.assertNotEqual(resp.status_code, 200)

            # 6) Detach -> 200, and detaching again is idempotent (still 200).
            resp = self._detach_backend(base_url_b, headers=admin)
            self.assertEqual(resp.status_code, 200, resp.text)
            resp = self._detach_backend(base_url_b, headers=admin)
            self.assertEqual(resp.status_code, 200, resp.text)

            # 7) After detach, GET should report backend == None again.
            resp = self._get_backend(base_url_b, headers=admin)
            self.assertEqual(resp.status_code, 200)
            self.assertIsNone(resp.json().get("backend"))

            # 8) Re-attach `file` -> 200 (server still healthy after detach).
            resp = self._attach_backend(base_url_b, "file", headers=admin)
            self.assertEqual(resp.status_code, 200, resp.text)
        finally:
            kill_process_tree(process_b.pid)


if __name__ == "__main__":
    unittest.main()
