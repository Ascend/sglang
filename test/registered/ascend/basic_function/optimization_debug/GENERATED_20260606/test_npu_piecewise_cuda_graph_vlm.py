import os
import unittest
from types import SimpleNamespace

import torch

from sglang import Engine
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


NPU_ENV = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "PYTHONWARNINGS": "ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning",
}


class TestNPUPiecewiseGraphQwen25VL(CustomTestCase):
    """Test piecewise CUDA graph on NPU with Qwen2.5-VL-3B-Instruct VLM.

    [Test Category] Piecewise Graph Optimization (VLM)
    [Test Target] --enforce-piecewise-cuda-graph; --disable-radix-cache; ascend attention backend
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--enforce-piecewise-cuda-graph",
                "--disable-radix-cache",
                "--mem-fraction-static",
                0.8,
            ],
            env={**NPU_ENV, **os.environ},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.60)


class TestNPUPiecewiseGraphQwen25VLEmbedding(CustomTestCase):
    """Test piecewise CUDA graph on NPU with Qwen2.5-VL-3B-Instruct embedding.

    [Test Category] Piecewise Graph Optimization (VLM Embedding)
    [Test Target] enforce_piecewise_cuda_graph vs disable_piecewise_cuda_graph (embedding consistency)
    """

    def test_embedding(self):
        model_path = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
        chat_template = get_chat_template_by_model_path(model_path)
        text = f"{chat_template.image_token}What is in this picture? Answer: "

        # Run with piecewise CUDA graph enabled
        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            attention_backend="ascend",
            enforce_piecewise_cuda_graph=True,
        )
        out = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0]["embedding"]
        engine.shutdown()
        self.assertGreater(len(out), 0)

        # Run with piecewise CUDA graph disabled
        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            attention_backend="ascend",
            disable_piecewise_cuda_graph=True,
        )
        out_without_pcg = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0][
            "embedding"
        ]
        engine.shutdown()
        self.assertGreater(len(out_without_pcg), 0)

        # Verify embedding consistency between modes
        t_out = torch.tensor(out)
        t_out_without_pcg = torch.tensor(out_without_pcg)
        max_abs_diff = (t_out - t_out_without_pcg).abs().max().item()
        max_rel_diff = (
            ((t_out - t_out_without_pcg).abs() / (t_out_without_pcg.abs() + 1e-8))
            .max()
            .item()
        )
        self.assertTrue(
            torch.allclose(
                t_out,
                t_out_without_pcg,
                atol=1e-2,
                rtol=1e-2,
            ),
            f"Piecewise CUDA graph embedding mismatch: max_abs_diff={max_abs_diff}, max_rel_diff={max_rel_diff}",
        )


if __name__ == "__main__":
    unittest.main()
