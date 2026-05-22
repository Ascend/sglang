import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_32B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHiCacheMemLayoutPageFirst(CustomTestCase):
    """Testcase: HiCache with page_first memory layout on NPU.

    [Test Category] HiCache
    [Test Target] --hicache-mem-layout page_first
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-mem-layout",
            "page_first",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_page_first_layout_basic(self):
        """Test basic inference with page_first memory layout."""
        import requests

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def test_page_first_layout_cache_reuse(self):
        """Test cache reuse with page_first memory layout."""
        import requests

        prompt = "What is machine learning? " * 60
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            if i == 0:
                self.assertEqual(cached_tokens, 0)
            else:
                self.assertGreater(cached_tokens, 0)


class TestNPUHiCacheMemLayoutPageFirstDirect(CustomTestCase):
    """Testcase: HiCache with page_first_direct memory layout on NPU.

    [Test Category] HiCache
    [Test Target] --hicache-mem-layout page_first_direct
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-mem-layout",
            "page_first_direct",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_page_first_direct_layout_basic(self):
        """Test basic inference with page_first_direct memory layout."""
        import requests

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def test_page_first_direct_layout_cache_reuse(self):
        """Test cache reuse with page_first_direct memory layout."""
        import requests

        prompt = "Explain quantum computing in detail. " * 60
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            if i == 0:
                self.assertEqual(cached_tokens, 0)
            else:
                self.assertGreater(cached_tokens, 0)


class TestNPUHiCacheMemLayoutPageFirstKVSplit(CustomTestCase):
    """Testcase: HiCache with page_first_kv_split memory layout on NPU.

    [Test Category] HiCache
    [Test Target] --hicache-mem-layout page_first_kv_split
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-mem-layout",
            "page_first_kv_split",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_page_first_kv_split_layout_basic(self):
        """Test basic inference with page_first_kv_split memory layout."""
        import requests

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def test_page_first_kv_split_layout_cache_reuse(self):
        """Test cache reuse with page_first_kv_split memory layout."""
        import requests

        prompt = "What are the benefits of cloud computing? " * 60
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            if i == 0:
                self.assertEqual(cached_tokens, 0)
            else:
                self.assertGreater(cached_tokens, 0)


register_npu_ci(est_time=500, suite="nightly-2-npu-a3", nightly=True)


class TestNPUHiCacheMemLayoutWithTP(CustomTestCase):
    """Testcase: HiCache memory layout with TP=2 on NPU.

    [Test Category] HiCache
    [Test Target] --hicache-mem-layout with --tp-size 2
    """

    model = QWEN3_32B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-mem-layout",
            "page_first_direct",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            "2",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mem_layout_with_tp_basic(self):
        """Test basic inference with memory layout and TP=2."""
        import requests

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_mem_layout_with_tp_cache_reuse(self):
        """Test cache reuse with memory layout and TP=2."""
        import requests

        prompt = "Describe the history of artificial intelligence. " * 60
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            if i == 0:
                self.assertEqual(cached_tokens, 0)
            else:
                self.assertGreater(cached_tokens, 0)


if __name__ == "__main__":
    unittest.main()
