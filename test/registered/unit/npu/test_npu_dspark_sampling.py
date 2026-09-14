"""Check the different NPU/CUDA random-buffer strides with multiple rows."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.speculative.dspark import dspark_accept
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkSamplingStride(CustomTestCase):
    def test_random_buffer_width(self):
        for is_npu in (False, True):
            for bs, width in ((1, 2), (3, 8)):
                with self.subTest(is_npu=is_npu, bs=bs, width=width):
                    candidates = torch.zeros((bs, width), dtype=torch.int64)
                    buffers = tuple(
                        torch.zeros((bs, width), dtype=torch.int32) for _ in range(5)
                    ) + (
                        torch.zeros(bs, dtype=torch.int32),
                    )
                    sample = Mock()
                    with (
                        patch.object(dspark_accept, "_is_npu", is_npu),
                        patch.object(
                            dspark_accept,
                            "_get_or_create_chain_verify_buffers",
                            return_value=buffers,
                        ),
                        patch.object(
                            dspark_accept.SoftmaxTemp,
                            "execute",
                            return_value=torch.full((bs * width, 4), 0.25),
                        ),
                        patch.object(
                            dspark_accept, "chain_speculative_sampling_triton", sample
                        ),
                    ):
                        dspark_accept._accept_sampling_core(
                            candidates=candidates,
                            target_logits=torch.zeros((bs * width, 4)),
                            draft_probs=torch.full((bs, width, 4), 0.25),
                            sampling_info=SimpleNamespace(
                                need_top_k_sampling=False,
                                need_top_p_sampling=False,
                                temperatures=torch.ones(bs),
                            ),
                            draft_input=SimpleNamespace(),
                            gamma=width - 1,
                            verify_num_draft_tokens=width,
                            cutoff_verify_lens=None,
                        )
                    kwargs = sample.call_args.kwargs
                    uniform = kwargs["uniform_samples"]
                    expected_width = width if is_npu else width - 1
                    self.assertEqual(uniform.shape, (bs, expected_width))
                    self.assertEqual(uniform.stride(0), expected_width)
                    self.assertEqual(
                        kwargs["uniform_samples_for_final_sampling"].shape, (bs,)
                    )
                    self.assertIs(kwargs["candidates"], candidates)


if __name__ == "__main__":
    unittest.main()
