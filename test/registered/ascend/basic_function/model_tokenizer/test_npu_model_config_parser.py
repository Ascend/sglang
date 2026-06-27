"""Test for --model-config-parser on NPU.

Transplanted from: test/registered/unit/configs/test_model_config_parser_registry.py (GPU/CPU)
- T1 test_register_then_get_roundtrip: verify parser registration → retrieval
- T2 test_register_rejects_non_subclass: verify non-subclass rejected
- T3 test_unknown_name_raises_with_registered_list: verify error msg on unknown name
New:
- T4 test_model_config_parser_auto: verify auto parser starts server + inference
- T5 test_model_config_parser_hf: verify hf parser starts server + inference
"""

import unittest

import requests
from transformers import PretrainedConfig

from sglang.srt.configs.model_config_parser_registry import (
    _MODEL_CONFIG_PARSER_REGISTRY,
    ModelConfigParserBase,
    get_model_config_parser,
    register_model_config_parser,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="debug-full-1-npu-a3", nightly=True)


class _FakeParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


class _AnotherFakeParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        return PretrainedConfig()


class TestNpuModelConfigParserRegistry(CustomTestCase):
    """Transplanted from GPU test_model_config_parser_registry.py — verify parser registry API.

    GPU origin: TestModelConfigParserRegistry in test/registered/unit/configs/test_model_config_parser_registry.py

    [Test Category] Parameter / Unit
    [Test Target] --model-config-parser registry API
    """

    def setUp(self):
        self._saved_registry = dict(_MODEL_CONFIG_PARSER_REGISTRY)
        _MODEL_CONFIG_PARSER_REGISTRY.clear()

    def tearDown(self):
        _MODEL_CONFIG_PARSER_REGISTRY.clear()
        _MODEL_CONFIG_PARSER_REGISTRY.update(self._saved_registry)

    def test_register_then_get_roundtrip_npu(self):
        """Verify register → get returns correct parser instance.

        GPU origin: test_register_then_get_roundtrip
        """
        register_model_config_parser("fake")(_FakeParser)
        self.assertIsInstance(get_model_config_parser("fake"), _FakeParser)

    def test_register_rejects_non_subclass_npu(self):
        """Verify registering a non-ModelConfigParserBase class raises ValueError.

        GPU origin: test_register_rejects_non_subclass
        """

        class NotAParser:
            pass

        with self.assertRaises(ValueError) as ctx:
            register_model_config_parser("bad")(NotAParser)
        self.assertIn("ModelConfigParserBase", str(ctx.exception))

    def test_unknown_name_raises_with_registered_list_npu(self):
        """Verify get with unknown name raises ValueError containing registered names.

        GPU origin: test_unknown_name_raises_with_registered_list
        """
        register_model_config_parser("fake")(_FakeParser)
        register_model_config_parser("another")(_AnotherFakeParser)
        with self.assertRaises(ValueError) as ctx:
            get_model_config_parser("does-not-exist")
        msg = str(ctx.exception)
        self.assertIn("does-not-exist", msg)
        self.assertIn("another", msg)
        self.assertIn("fake", msg)


class TestNpuModelConfigParserAuto(CustomTestCase):
    """Verify --model-config-parser=auto starts server and inference succeeds.

    [Test Category] Parameter
    [Test Target] --model-config-parser=auto (default value, E2E)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--model-config-parser",
            "auto",
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

    def test_model_config_parser_auto(self):
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


class TestNpuModelConfigParserHf(CustomTestCase):
    """Verify --model-config-parser=hf starts server and inference succeeds.

    [Test Category] Parameter / Boundary
    [Test Target] --model-config-parser=hf (E2E)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--model-config-parser",
            "hf",
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

    def test_model_config_parser_hf(self):
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
