import glob
import os
import shutil
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestDownloadDir(CustomTestCase):
    """Testcase：Verify set --download-dir parameter, the parameter take effect and the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --download-dir
    """

    model = "Qwen/Qwen2-0.5B-Instruct"

    @classmethod
    def setUpClass(cls):
        cls.download_dir = tempfile.mkdtemp(prefix="sglang-download-dir-")
        # The CI image pre-populates the default ModelScope cache, so a cached
        # lookup would bypass --download-dir entirely. Point the default cache
        # at an empty directory: resolution then has to download into
        # --download-dir, which is what this test verifies.
        cls.empty_modelscope_cache = tempfile.mkdtemp(prefix="sglang-empty-ms-cache-")
        other_args = [
            "--download-dir",
            cls.download_dir,
            "--attention-backend",
            "ascend",
            "--disable-decode-cuda-graph",
            "--disable-prefill-cuda-graph",
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                **os.environ,
                "SGLANG_USE_MODELSCOPE": "true",
                "MODELSCOPE_CACHE": cls.empty_modelscope_cache,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.download_dir, ignore_errors=True)
        shutil.rmtree(cls.empty_modelscope_cache, ignore_errors=True)

    def test_download_dir(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        # check model weight
        weight_suffixes = ("*.safetensors", "*.bin", "*.pth")
        weight_files = []
        for suffix in weight_suffixes:
            weight_files.extend(
                glob.glob(os.path.join(self.download_dir, "**", suffix), recursive=True)
            )
        self.assertGreater(
            len(weight_files),
            0,
            msg=f"--download-dir {self.download_dir} No model weight",
        )


if __name__ == "__main__":
    unittest.main()
