"""NPU integration of DSpark probability construction and chain sampling."""

import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptSampling
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="nightly-1-npu-a3", nightly=True)


class TestDSparkChainSampling(CustomTestCase):
    def test_multiple_rows_full_first_and_partial_rejection(self):
        width, vocab = 8, 16
        candidates = torch.arange(width, dtype=torch.int64).repeat(3, 1).npu()
        next_tokens = torch.arange(1, width + 1).repeat(3, 1)
        next_tokens[1, 0] = 10  # Reject the first draft.
        next_tokens[2, 3] = 10  # Accept three drafts, then reject.
        logits = torch.full((3, width, vocab), -1000.0)
        logits.scatter_(2, next_tokens.unsqueeze(-1), 0.0)
        draft_probs = torch.zeros(3, width - 1, vocab)
        draft_probs.scatter_(2, torch.arange(1, width).repeat(3, 1).unsqueeze(-1), 1.0)
        for _ in range(3):
            correct, bonus, trimmed = AcceptSampling.execute(
                candidates=candidates,
                target_logits=logits.flatten(0, 1).npu(),
                draft_probs=draft_probs.npu(),
                sampling_info=SimpleNamespace(
                    need_top_k_sampling=False,
                    need_top_p_sampling=False,
                    temperatures=torch.ones(3, 1, device="npu"),
                ),
                draft_input=SimpleNamespace(),
                gamma=width - 1,
                verify_num_draft_tokens=width,
            )
            self.assertEqual(correct.cpu().tolist(), [7, 0, 3])
            self.assertEqual(bonus.cpu().tolist(), [8, 10, 10])
            self.assertEqual(trimmed.cpu().tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
