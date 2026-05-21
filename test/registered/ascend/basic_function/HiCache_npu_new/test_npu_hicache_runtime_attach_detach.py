"""
E2E test for HiCache storage runtime attach/detach on Ascend NPU.

This test launches an SGLang server with hierarchical cache enabled but WITHOUT
any storage backend at startup, then attaches/detaches a storage backend via the
HTTP endpoints.

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --hicache-io-backend kernel_ascend
- --hicache-mem-layout page_first_direct

Usage:
    python3 -m pytest test/registered/ascend/basic_function/HiCache/test_npu_hicache_runtime_attach_detach.py -v
"""

import json
import os
import tempfile
import time
import unittest
from urllib import error, request

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHiCacheStorageRuntimeAttachDetach(CustomTestCase):
    """Test runtime attach/detach of HiCache storage backend on NPU.

    [Test Category] HiCache
    [Test Target] Runtime storage backend management via HTTP API
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = QWEN3_8B_WEIGHTS_PATH
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"

        cls.other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            "64",
            "--enable-cache-report",
            "--hicache-io-backend",
            "kernel_ascend",
            "--hicache-mem-layout",
            "page_first_direct",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            # NOTE: do NOT pass --hicache-storage-backend* here
        ]

        cls.env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _wait_for_server_ready(
        cls, base_url: str, timeout: int = 60, process=None
    ) -> bool:
        wait_for_http_ready(
            url=f"{base_url}/health",
            timeout=timeout,
            process=process,
        )
        return True

    @staticmethod
    def _http_get(url: str, timeout: int = 10, headers: dict | None = None):
        try:
            req = request.Request(url, headers=headers or {}, method="GET")
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    @staticmethod
    def _http_put_json_with_headers(
        url: str,
        payload: dict | None = None,
        timeout: int = 30,
        headers: dict | None = None,
    ):
        data = None
        all_headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            all_headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=all_headers, method="PUT")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    @staticmethod
    def _http_delete_with_headers(
        url: str, timeout: int = 30, headers: dict | None = None
    ):
        all_headers = dict(headers or {})
        req = request.Request(url, headers=all_headers, method="DELETE")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    def _get_backend_status(self, base_url: str, headers: dict | None = None):
        code, body = self._http_get(
            f"{base_url}/hicache/storage-backend", timeout=10, headers=headers
        )
        self.assertEqual(code, 200, body)
        return json.loads(body)

    def _attach_backend(
        self,
        base_url: str,
        backend: str,
        extra_cfg: dict,
        prefetch_policy: str = "timeout",
        write_policy: str = "write_through",
        headers: dict | None = None,
    ):
        payload = {
            "hicache_storage_backend": backend,
            "hicache_storage_backend_extra_config_json": json.dumps(extra_cfg),
            "hicache_storage_prefetch_policy": prefetch_policy,
            "hicache_write_policy": write_policy,
        }
        return self._http_put_json_with_headers(
            f"{base_url}/hicache/storage-backend",
            payload,
            timeout=30,
            headers=headers,
        )

    def _detach_backend(self, base_url: str, headers: dict | None = None):
        return self._http_delete_with_headers(
            f"{base_url}/hicache/storage-backend",
            timeout=30,
            headers=headers,
        )

    def test_runtime_attach_detach(self):
        """Test runtime attach/detach lifecycle with admin authentication."""
        # Phase A: WITHOUT --admin-api-key, endpoints must return 400
        process1 = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args,
            env=self.env,
        )
        try:
            self._wait_for_server_ready(self.base_url, process=process1)

            code_info, _body_info = self._http_get(
                f"{self.base_url}/hicache/storage-backend", timeout=10
            )
            self.assertEqual(code_info, 400)

            code_attach_no_admin, _ = self._attach_backend(
                base_url=self.base_url, backend="file", extra_cfg={}
            )
            self.assertEqual(code_attach_no_admin, 400)

            code_detach_no_admin, _ = self._detach_backend(self.base_url)
            self.assertEqual(code_detach_no_admin, 400)
        finally:
            kill_process_tree(process1.pid)
            time.sleep(2)

        # Phase B: WITH --admin-api-key, must provide Authorization: Bearer <admin_key>
        admin_key = "sglang-test-admin-key"
        base_url2 = f"http://127.0.0.1:{find_available_port(int(self.base_url.rsplit(':', 1)[1]) + 1)}"
        other_args2 = list(self.other_args) + ["--admin-api-key", admin_key]
        process2 = popen_launch_server(
            self.model,
            base_url2,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args2,
            env=self.env,
        )
        try:
            self._wait_for_server_ready(base_url2, process=process2)

            # 1) Initially unauthorized
            code_info2_unauth, _ = self._http_get(
                f"{base_url2}/hicache/storage-backend", timeout=10
            )
            self.assertEqual(code_info2_unauth, 401)

            admin_headers = {"Authorization": f"Bearer {admin_key}"}
            status0 = self._get_backend_status(base_url2, headers=admin_headers)
            self.assertIsNone(status0.get("hicache_storage_backend"))

            # 2) Attach should succeed when idle
            extra_cfg = {
                "hicache_storage_pass_prefix_keys": True,
                "prefetch_threshold": 256,
                "prefetch_timeout_base": 3,
                "prefetch_timeout_per_ki_token": 0.01,
            }

            code_attach_unauth, _ = self._attach_backend(
                base_url=base_url2, backend="file", extra_cfg=extra_cfg
            )
            self.assertEqual(code_attach_unauth, 401)

            code_attach, body_attach = self._attach_backend(
                base_url=base_url2,
                backend="file",
                extra_cfg=extra_cfg,
                prefetch_policy="timeout",
                write_policy="write_back",
                headers=admin_headers,
            )
            self.assertEqual(code_attach, 200, f"{code_attach} - {body_attach}")

            status1 = self._get_backend_status(base_url2, headers=admin_headers)
            self.assertEqual(status1.get("hicache_storage_backend"), "file")
            self.assertEqual(
                status1.get("hicache_storage_backend_extra_config"),
                json.dumps(extra_cfg),
            )
            self.assertEqual(status1.get("hicache_storage_prefetch_policy"), "timeout")
            self.assertEqual(status1.get("hicache_write_policy"), "write_back")

            # 3) Attach again with updated policies
            code_attach_again, body_attach_again = self._attach_backend(
                base_url=base_url2,
                backend="file",
                extra_cfg=extra_cfg,
                prefetch_policy="wait_complete",
                write_policy="write_through_selective",
                headers=admin_headers,
            )
            self.assertEqual(
                code_attach_again, 200, f"{code_attach_again} - {body_attach_again}"
            )

            status2 = self._get_backend_status(base_url2, headers=admin_headers)
            self.assertEqual(
                status2.get("hicache_storage_prefetch_policy"), "wait_complete"
            )
            self.assertEqual(
                status2.get("hicache_write_policy"), "write_through_selective"
            )

            # 4) Attach different backend should be rejected
            code_attach_diff, body_attach_diff = self._attach_backend(
                base_url=base_url2,
                backend="mooncake",
                extra_cfg=extra_cfg,
                headers=admin_headers,
            )
            self.assertNotEqual(code_attach_diff, 200, body_attach_diff)

            # 5) Detach should succeed and be idempotent
            code_detach, body_detach = self._detach_backend(
                base_url2, headers=admin_headers
            )
            self.assertEqual(code_detach, 200, f"{code_detach} - {body_detach}")

            status3 = self._get_backend_status(base_url2, headers=admin_headers)
            self.assertIsNone(status3.get("hicache_storage_backend"))

            code_detach_again, _ = self._detach_backend(
                base_url2, headers=admin_headers
            )
            self.assertEqual(code_detach_again, 200)

            # 6) Re-attach after detach should succeed
            code_attach2, body_attach2 = self._attach_backend(
                base_url=base_url2,
                backend="file",
                extra_cfg=extra_cfg,
                headers=admin_headers,
            )
            self.assertEqual(code_attach2, 200, f"{code_attach2} - {body_attach2}")

            status4 = self._get_backend_status(base_url2, headers=admin_headers)
            self.assertEqual(status4.get("hicache_storage_backend"), "file")

            # Cleanup
            self._detach_backend(base_url2, headers=admin_headers)
        finally:
            kill_process_tree(process2.pid)
            time.sleep(2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
