import json
import os
import tempfile
import time
import unittest
from urllib import error, request

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
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
    """Testcase: E2E check for HiCache storage runtime attach/detach on NPU.

    Tests launching server with hierarchical cache enabled but WITHOUT storage backend,
    then attaches/detaches storage backend via HTTP endpoints.

    [Test Category] HiCache
    [Test Target] PUT/DELETE /hicache/storage-backend HTTP API
    """

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"

        cls.other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-size",
            "100",
            "--page-size",
            "128",
            "--enable-cache-report",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        cls.env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
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
    def _http_get(url: str, timeout: int = 10, headers: dict = None):
        try:
            req = request.Request(url, headers=headers or {}, method="GET")
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    @staticmethod
    def _http_post_json(url: str, payload: dict = None, timeout: int = 30):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    @staticmethod
    def _http_put_json_with_headers(
        url: str,
        payload: dict = None,
        timeout: int = 30,
        headers: dict = None,
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
    def _http_delete_with_headers(url: str, timeout: int = 30, headers: dict = None):
        all_headers = dict(headers or {})
        req = request.Request(url, headers=all_headers, method="DELETE")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    def _get_backend_status(self, base_url: str, headers: dict = None):
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
        headers: dict = None,
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

    def _detach_backend(self, base_url: str, headers: dict = None):
        return self._http_delete_with_headers(
            f"{base_url}/hicache/storage-backend",
            timeout=30,
            headers=headers,
        )

    def test_runtime_attach_detach(self):
        """Test runtime attach/detach of HiCache storage backend via HTTP API."""

        phase_a_port = find_available_port(int(self.base_url.rsplit(":", 1)[1]))
        phase_a_url = f"http://127.0.0.1:{phase_a_port}"

        process1 = popen_launch_server(
            self.model,
            phase_a_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args,
            env=self.env,
        )
        try:
            self._wait_for_server_ready(phase_a_url, process=process1)

            code_info, _body_info = self._http_get(
                f"{phase_a_url}/hicache/storage-backend", timeout=10
            )
            self.assertEqual(code_info, 400)
            code_attach_no_admin, _body_attach_no_admin = self._attach_backend(
                base_url=phase_a_url, backend="file", extra_cfg={}
            )
            self.assertEqual(code_attach_no_admin, 400)
            code_detach_no_admin, _body_detach_no_admin = self._detach_backend(
                phase_a_url
            )
            self.assertEqual(code_detach_no_admin, 400)
        finally:
            kill_process_tree(process1.pid)
            time.sleep(2)

        admin_key = "sglang-test-admin-key"
        phase_b_port = find_available_port(phase_a_port + 1)
        phase_b_url = f"http://127.0.0.1:{phase_b_port}"
        other_args2 = list(self.other_args) + ["--admin-api-key", admin_key]
        process2 = popen_launch_server(
            self.model,
            phase_b_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args2,
            env=self.env,
        )
        try:
            self._wait_for_server_ready(phase_b_url, process=process2)

            code_info2_unauth, _ = self._http_get(
                f"{phase_b_url}/hicache/storage-backend", timeout=10
            )
            self.assertEqual(code_info2_unauth, 401)

            admin_headers = {"Authorization": f"Bearer {admin_key}"}
            status0 = self._get_backend_status(phase_b_url, headers=admin_headers)
            self.assertIsNone(status0.get("hicache_storage_backend"))

            extra_cfg = {
                "hicache_storage_pass_prefix_keys": True,
                "prefetch_threshold": 256,
                "prefetch_timeout_base": 3,
                "prefetch_timeout_per_ki_token": 0.01,
            }

            code_attach_unauth, _ = self._attach_backend(
                base_url=phase_b_url, backend="file", extra_cfg=extra_cfg
            )
            self.assertEqual(code_attach_unauth, 401)

            code_attach, body_attach = self._attach_backend(
                base_url=phase_b_url,
                backend="file",
                extra_cfg=extra_cfg,
                prefetch_policy="timeout",
                write_policy="write_back",
                headers=admin_headers,
            )
            self.assertEqual(code_attach, 200, f"{code_attach} - {body_attach}")

            status1 = self._get_backend_status(phase_b_url, headers=admin_headers)
            self.assertEqual(status1.get("hicache_storage_backend"), "file")
            self.assertEqual(
                status1.get("hicache_storage_backend_extra_config"),
                json.dumps(extra_cfg),
            )
            self.assertEqual(status1.get("hicache_storage_prefetch_policy"), "timeout")
            self.assertEqual(status1.get("hicache_write_policy"), "write_back")

            code_attach_again, body_attach_again = self._attach_backend(
                base_url=phase_b_url,
                backend="file",
                extra_cfg=extra_cfg,
                prefetch_policy="wait_complete",
                write_policy="write_through_selective",
                headers=admin_headers,
            )
            self.assertEqual(
                code_attach_again, 200, f"{code_attach_again} - {body_attach_again}"
            )

            status2 = self._get_backend_status(phase_b_url, headers=admin_headers)
            self.assertEqual(
                status2.get("hicache_storage_backend_extra_config"),
                json.dumps(extra_cfg),
            )
            self.assertEqual(
                status2.get("hicache_storage_prefetch_policy"), "wait_complete"
            )
            self.assertEqual(
                status2.get("hicache_write_policy"), "write_through_selective"
            )

            code_attach_again, body_attach_again = self._attach_backend(
                base_url=phase_b_url,
                backend="mooncake",
                extra_cfg=extra_cfg,
                headers=admin_headers,
            )
            self.assertNotEqual(code_attach_again, 200, body_attach_again)

            code_detach, body_detach = self._detach_backend(
                phase_b_url, headers=admin_headers
            )
            self.assertEqual(code_detach, 200, f"{code_detach} - {body_detach}")
            status3 = self._get_backend_status(phase_b_url, headers=admin_headers)
            self.assertIsNone(status3.get("hicache_storage_backend"))
            self.assertEqual(
                status3.get("hicache_storage_prefetch_policy"), "wait_complete"
            )
            self.assertEqual(
                status3.get("hicache_write_policy"), "write_through_selective"
            )

            code_detach_again, body_detach_again = self._detach_backend(
                phase_b_url, headers=admin_headers
            )
            self.assertEqual(
                code_detach_again,
                200,
                f"{code_detach_again} - {body_detach_again}",
            )

            code_attach2, body_attach2 = self._attach_backend(
                base_url=phase_b_url,
                backend="file",
                extra_cfg=extra_cfg,
                headers=admin_headers,
            )
            self.assertEqual(code_attach2, 200, f"{code_attach2} - {body_attach2}")
            status4 = self._get_backend_status(phase_b_url, headers=admin_headers)
            self.assertEqual(status4.get("hicache_storage_backend"), "file")
            self.assertEqual(
                status4.get("hicache_storage_backend_extra_config"),
                json.dumps(extra_cfg),
            )
            self.assertEqual(status4.get("hicache_storage_prefetch_policy"), "timeout")
            self.assertEqual(status4.get("hicache_write_policy"), "write_through")

            code_detach2, body_detach2 = self._detach_backend(
                phase_b_url, headers=admin_headers
            )
            self.assertEqual(code_detach2, 200, f"{code_detach2} - {body_detach2}")
        finally:
            kill_process_tree(process2.pid)
            time.sleep(2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
