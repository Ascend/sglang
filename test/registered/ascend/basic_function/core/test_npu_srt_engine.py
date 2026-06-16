import asyncio
import json
import unittest

import torch

import sglang as sgl
from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_0_6B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

DEFAULT_SMALL_MODEL_NAME_FOR_TEST = QWEN3_0_6B_WEIGHTS_PATH
register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestSRTEngine(CustomTestCase):

    def test_1_engine_runtime_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=model_path,
            random_seed=42,
            attention_backend="ascend",
            disable_cuda_graph=True,
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
            disable_cuda_graph=True,
        )
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=model_path,
            is_embedding=True,
            random_seed=42,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out2 = torch.tensor(json.loads(runtime.encode(prompt))["embedding"])
        runtime.shutdown()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5, rtol=1e-3))

    def test_3_engine_token_ids_consistency(self):
        # just to ensure there is no issue running multiple generate calls
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]

        tokenizer = get_tokenizer(model_path)
        token_ids = tokenizer.encode(prompt)
        out2 = engine.generate(input_ids=token_ids, sampling_params=sampling_params)[
            "text"
        ]

        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_6_engine_cpu_offload(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=128,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=128,
            cpu_offload_gb=3,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_7_engine_offline_throughput(self):
        server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        bench_args = BenchArgs(num_prompts=10)
        result = throughput_test(server_args=server_args, bench_args=bench_args)
        self.assertGreater(result["total_throughput"], 3000)

    def test_8_engine_async_encode_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH

        engine = sgl.Engine(
            model_path=model_path,
            is_embedding=True,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )

        # Get sync and async embeddings
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        loop = asyncio.get_event_loop()
        out2 = torch.tensor(
            loop.run_until_complete(engine.async_encode(prompt))["embedding"]
        )

        engine.shutdown()

        print("\n==== Shapes ====")
        print(f"sync shape: {out1.shape}")
        print(f"async shape: {out2.shape}")

        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-5, rtol=1e-3),
            "Sync and async embeddings are not equal within tolerance",
        )


if __name__ == "__main__":
    unittest.main()
