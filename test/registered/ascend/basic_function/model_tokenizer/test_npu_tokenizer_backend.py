"""End-to-end test for --tokenizer-backend on NPU.

Transplanted from: test/registered/unit/utils/test_hf_transformers_fastokens.py (GPU/CPU)
- T1 test_shim_is_applied: verify fastokens injection (removed @unittest.skipUnless)
- T2 test_encode_decode_roundtrip: verify encode → decode roundtrip (removed @unittest.skipUnless)
New:
- T3 test_tokenizer_backend_huggingface: verify huggingface backend server + inference
- T4 test_tokenizer_backend_fastokens: verify fastokens backend server + inference
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)

TOKENIZER_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
SERVER_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH


class TestNpuFastokensBackend(CustomTestCase):
    """Transplanted from GPU test_hf_transformers_fastokens.py — verify fastokens tokenizer injection
    and encode/decode correctness. @unittest.skipUnless has been removed per NPU policy.

    [Test Category] Parameter
    [Test Target] --tokenizer-backend=fastokens (unit-level verification)
    """

    def test_fastokens_shim_is_applied_npu(self):
        """Verify _TokenizerShim is applied when tokenizer_backend='fastokens'.

        GPU origin: test_shim_is_applied in test/registered/unit/utils/test_hf_transformers_fastokens.py
        """
        from fastokens._compat import _TokenizerShim
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            TOKENIZER_MODEL,
            tokenizer_backend="fastokens",
        )
        backend = getattr(tokenizer, "_tokenizer", None)
        self.assertIsInstance(
            backend,
            _TokenizerShim,
            f"Expected tokenizer._tokenizer to be _TokenizerShim, "
            f"got {type(backend).__name__}",
        )

    def test_fastokens_encode_decode_roundtrip_npu(self):
        """Verify encode → decode roundtrip with fastokens backend.

        GPU origin: test_encode_decode_roundtrip in test/registered/unit/utils/test_hf_transformers_fastokens.py
        """
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            TOKENIZER_MODEL,
            tokenizer_backend="fastokens",
        )
        text = "Hello, world!"
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), text)


class TestNpuTokenizerBackendHuggingface(CustomTestCase):
    """Verify --tokenizer-backend=huggingface starts server and inference succeeds.

    [Test Category] Parameter
    [Test Target] --tokenizer-backend=huggingface (default value, E2E)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = SERVER_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tokenizer-backend",
            "huggingface",
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

    def test_tokenizer_backend_huggingface(self):
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


class TestNpuTokenizerBackendFastokens(CustomTestCase):
    """Verify --tokenizer-backend=fastokens starts server and inference succeeds.

    [Test Category] Parameter / Boundary
    [Test Target] --tokenizer-backend=fastokens (E2E)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = SERVER_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tokenizer-backend",
            "fastokens",
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

    def test_tokenizer_backend_fastokens(self):
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


if __name__ == "__main__":
    unittest.main()
