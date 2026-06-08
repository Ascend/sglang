import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    QWEN3_30B_A3B_W8A8_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=3600, suite="weekly-16-npu-a3", nightly=True)


NPU_ENV = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "2048",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "0",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "PYTHONWARNINGS": "ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning",
}


def _mmlu_eval(base_url, model):
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
    )
    return run_eval(args)


class TestHybridTPOnly(CustomTestCase):
    """Test hybrid parallelism on NPU with TP-only baseline.

    [Test Category] Hybrid Parallelism
    [Test Target] --tp 16 (MLA model baseline)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridDPAttention(CustomTestCase):
    """Test hybrid parallelism on NPU with DP attention (DP=4).

    [Test Category] Hybrid Parallelism
    [Test Target] --enable-dp-attention --dp 4 with TP=16
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridFullDPAttention(CustomTestCase):
    """Test hybrid parallelism on NPU with full DP attention (DP=16).

    [Test Category] Hybrid Parallelism
    [Test Target] --enable-dp-attention --dp 16 (full DP)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--enable-dp-attention",
                "--dp",
                "16",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridDenseTPOne(CustomTestCase):
    """Test hybrid parallelism on NPU with dense FFN TP=1.

    [Test Category] Hybrid Parallelism
    [Test Target] --moe-dense-tp-size 1 (separate dense and sparse TP)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--moe-dense-tp-size",
                "1",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridDPLMHead(CustomTestCase):
    """Test hybrid parallelism on NPU with DP attention + DP LM head.

    [Test Category] Hybrid Parallelism
    [Test Target] --enable-dp-attention --enable-dp-lm-head
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--enable-dp-lm-head",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridEPOnly(CustomTestCase):
    """Test hybrid parallelism on NPU with EP enabled.

    [Test Category] Hybrid Parallelism
    [Test Target] --ep 16 (Expert Parallelism)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--ep-size",
                "16",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridDPEPCombined(CustomTestCase):
    """Test hybrid parallelism on NPU with DP attention + EP.

    [Test Category] Hybrid Parallelism
    [Test Target] --enable-dp-attention --dp 4 --ep 16
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--ep-size",
                "16",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridDeepEPAuto(CustomTestCase):
    """Test hybrid parallelism on NPU with DeepEP backend (auto mode).

    [Test Category] Hybrid Parallelism
    [Test Target] --moe-a2a-backend deepep --deepep-mode auto
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridDeepEPDPAttention(CustomTestCase):
    """Test hybrid parallelism on NPU with DeepEP + DP attention.

    [Test Category] Hybrid Parallelism
    [Test Target] --moe-a2a-backend deepep --enable-dp-attention
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


class TestHybridDeepEPFullStack(CustomTestCase):
    """Test hybrid parallelism on NPU with DeepEP + full hybrid stack.

    [Test Category] Hybrid Parallelism
    [Test Target] DeepEP + DP attention + DP LM head + moe-dense-tp 1
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                0.82,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = _mmlu_eval(self.base_url, self.model)
        self.assertGreater(metrics["score"], 0.48)


if __name__ == "__main__":
    unittest.main()
