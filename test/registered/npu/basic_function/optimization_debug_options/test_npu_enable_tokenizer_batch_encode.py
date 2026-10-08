import random
import time
import unittest
from statistics import mean
from types import SimpleNamespace

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=900, suite="full-1-npu-a3", nightly=True)

# 对齐 community v0.5.5 benchmark/benchmark_batch/benchmark_batch.py 的输入规模配置
NUM_REQUESTS = 10  # Total number of requests (each with BATCH_SIZE prompts)
NUM_TOKENS = 32000  # Tokens per prompt
BATCH_SIZE = 8  # Number of prompts per request


class TestTokenizerBatchEncode(CustomTestCase):
    """
    参数最初合入的PR #5141 https://github.com/sgl-project/sglang/pull/5141

    用例修改逻辑为
    1. 用 benchmark_batch.py 同样的随机长 prompt 生成 NUM_REQUESTS 个 BATCH_SIZE 批量；
    2. 逐条编码（关闭参数路径）测出 avg per prompt latency；
    3. 批量编码（开启参数路径）测出 avg per prompt latency；
    4. 断言批量编码耗时更低；
    5. test_gsm8k 复用开启参数的 server 做端到端正确性验证。

    [Test Category] Parameter
    [Test Target] --enable-tokenizer-batch-encode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # 测试进程内加载 tokenizer
        cls.tokenizer = get_tokenizer(cls.model)
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def _launch_server(self, enable_tokenizer_batch_encode):
        other_args = [
            "--attention-backend", "ascend", "--dp", "8",
            "--disable-radix-cache", "--disable-cuda-graph",
            "--max-prefill-tokens", "131072", "--chunked-prefill-size", "131072",
            "--mem-fraction-static", "0.9",
        ]
        if enable_tokenizer_batch_encode:
            other_args.append("--enable-tokenizer-batch-encode")
        return popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def _generate_random_prompt(self, index, num_tokens):
        vocab_size = self.tokenizer.vocab_size
        random_token_ids = [
            random.randint(0, vocab_size - 1) for _ in range(num_tokens)
        ]
        random_text = self.tokenizer.decode(
            random_token_ids, clean_up_tokenization_spaces=True
        )
        return f"Prompt {index}: {random_text}"

    def _build_batched_prompts(self):
        random.seed(0)
        return [
            [
                self._generate_random_prompt(i * BATCH_SIZE + j, NUM_TOKENS)
                for j in range(BATCH_SIZE)
            ]
            for i in range(NUM_REQUESTS)
        ]

    def _benchmark_tokenizer(self, batched_prompts, batch_mode):
        request_latencies = []
        for texts in batched_prompts:
            start_time = time.perf_counter()
            if batch_mode:
                self.tokenizer(texts)
            else:
                for t in texts:
                    self.tokenizer([t])
            request_latencies.append((time.perf_counter() - start_time) * 1000)
        avg_request_latency = mean(request_latencies)
        return avg_request_latency / BATCH_SIZE, request_latencies

    def test_tokenizer_batch_encode_throughput_improvement(self):
        batched_prompts = self._build_batched_prompts()

        # warmup
        self.tokenizer(batched_prompts[0])

        off_avg_per_prompt, off_latencies = self._benchmark_tokenizer(
            batched_prompts, batch_mode=False
        )
        on_avg_per_prompt, on_latencies = self._benchmark_tokenizer(
            batched_prompts, batch_mode=True
        )

        print(
            f"[TEMP] OFF per-request latencies (ms): "
            f"{[f'{v:.1f}' for v in off_latencies]}, avg_per_prompt={off_avg_per_prompt:.2f}ms"
        )
        print(
            f"[TEMP] ON per-request latencies (ms): "
            f"{[f'{v:.1f}' for v in on_latencies]}, avg_per_prompt={on_avg_per_prompt:.2f}ms"
        )
        print(
            f"[TEMP] RESULT off_avg_per_prompt={off_avg_per_prompt:.2f}ms, "
            f"on_avg_per_prompt={on_avg_per_prompt:.2f}ms, "
            f"improvement={(off_avg_per_prompt - on_avg_per_prompt) / off_avg_per_prompt * 100:.1f}%"
        )
        self.assertLess(
            on_avg_per_prompt,
            off_avg_per_prompt,
            f"expected lower avg per prompt latency with --enable-tokenizer-batch-encode: "
            f"off={off_avg_per_prompt:.2f}ms, on={on_avg_per_prompt:.2f}ms",
        )

    def test_gsm8k(self):
        # NOTE: 写回类属性, 保证 tearDownClass 能杀掉 server 进程
        self.__class__.process = self._launch_server(enable_tokenizer_batch_encode=True)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.38)


if __name__ == "__main__":
    unittest.main()
