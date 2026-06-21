import asyncio
import json
import unittest
from types import SimpleNamespace

import torch

import sglang as sgl
from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k_engine import run_eval

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestSRTEngine(unittest.TestCase):
    """Test SRT Engine Python API on NPU.

    [Test Category] Core
    [Test Target] Engine API, Runtime API, sync/async operations
    """

    def test_1_engine_runtime_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            attention_backend="ascend",
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=model_path,
            random_seed=42,
            attention_backend="ascend",
        )
        out2 = json.loads(runtime.generate(prompt, sampling_params))["text"]
        runtime.shutdown()

        self.assertEqual(out1, out2)

    def test_2_engine_runtime_encode_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH

        engine = sgl.Engine(
            model_path=model_path,
            is_embedding=True,
            random_seed=42,
            attention_backend="ascend",
        )
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=model_path,
            is_embedding=True,
            random_seed=42,
            attention_backend="ascend",
        )
        out2 = torch.tensor(json.loads(runtime.encode(prompt))["embedding"])
        runtime.shutdown()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5, rtol=1e-3))

    def test_3_engine_token_ids_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
        )
        out1 = engine.generate(prompt, sampling_params)["text"]

        tokenizer = get_tokenizer(model_path)
        token_ids = tokenizer.encode(prompt)
        out2 = engine.generate(input_ids=token_ids, sampling_params=sampling_params)[
            "text"
        ]

        engine.shutdown()

        self.assertEqual(out1, out2)

    def test_4_sync_async_stream_combination(self):
        prompt = "AI safety is"
        sampling_params = {"temperature": 0.8, "top_p": 0.95}

        llm = sgl.Engine(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            attention_backend="ascend",
        )

        output = llm.generate(prompt, sampling_params)

        output_generator = llm.generate(prompt, sampling_params, stream=True)
        offset = 0
        for output in output_generator:
            offset = len(output["text"])

        loop = asyncio.get_event_loop()
        output = loop.run_until_complete(llm.async_generate(prompt, sampling_params))

        async def async_streaming(engine):
            generator = await engine.async_generate(
                prompt, sampling_params, stream=True
            )

            offset = 0
            async for output in generator:
                offset = len(output["text"])

        loop.run_until_complete(async_streaming(llm))

        llm.shutdown()

    def test_5_gsm8k(self):
        args = SimpleNamespace(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            local_data_path=None,
            num_shots=5,
            num_questions=100,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["accuracy"], 0.15)

    def test_7_engine_offline_throughput(self):
        server_args = ServerArgs(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            attention_backend="ascend",
        )
        bench_args = BenchArgs(num_prompts=10)
        result = throughput_test(server_args=server_args, bench_args=bench_args)
        self.assertGreater(result["total_throughput"], 1000)


if __name__ == "__main__":
    unittest.main()
