"""CPU checks for DSpark MLA V2 query layout, padding and idle ranks."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.attention import ascend_backend
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3, suite="base-a-test-1-npu-a2")


class TestDSparkFIASV2(CustomTestCase):
    def test_query_layout_and_output_padding(self):
        for graph_mode, active_bs, padded_bs, is_nz in (
            (True, 2, 2, False),
            (True, 2, 2, True),
            (False, 2, 3, False),
            (False, 0, 1, True),
        ):
            with self.subTest(
                graph=graph_mode, active=active_bs, padded=padded_bs, nz=is_nz
            ):
                width, heads = 8, 2
                q = torch.arange(
                    padded_bs * width * heads * 512, dtype=torch.float32
                ).reshape(padded_bs * width, heads, 512)
                q_rope = torch.ones(padded_bs * width, heads, 64)
                backend = object.__new__(AscendAttnBackend)
                backend.use_mla = backend.use_fias_v2_bsnd = True
                backend.graph_mode = graph_mode
                backend.page_size = 128
                backend.kv_lora_rank = 512
                backend.qk_rope_head_dim = 64
                backend.tp_q_head_num = heads
                backend.q_head_num_padding = 4
                backend.model_dtype = torch.float32
                backend.speculative_num_draft_tokens = width
                backend.mtp_mask = torch.ones(16, 16, dtype=torch.bool).triu_(1)
                backend.token_to_kv_pool = SimpleNamespace(
                    get_kv_buffer=lambda _: (
                        torch.zeros(4, 128, 1, 512),
                        torch.zeros(4, 128, 1, 64),
                    ),
                )
                lengths = [127, 129, 256][:padded_bs]
                backend.forward_metadata = SimpleNamespace(
                    seq_lens_cpu_int=None,
                    seq_lens_cpu_list=lengths,
                    block_tables=torch.zeros(padded_bs, 2, dtype=torch.int32),
                )
                layer = SimpleNamespace(
                    layer_id=0,
                    tp_q_head_num=heads,
                    tp_k_head_num=1,
                    tp_v_head_num=1,
                    v_head_dim=512,
                    scaling=0.1,
                )
                batch = SimpleNamespace(num_token_non_padded_cpu=active_bs * width)
                kernel = Mock(
                    side_effect=lambda query, *args, **kwargs: (query + 1, None)
                )
                with (
                    patch.object(ascend_backend, "is_fia_nz", return_value=is_nz),
                    patch.object(
                        ascend_backend.torch_npu,
                        "npu_fused_infer_attention_score_v2",
                        kernel,
                    ),
                ):
                    output = backend.forward_mtp(
                        q, None, None, layer, batch, False, q_rope=q_rope
                    )
                expected = torch.zeros(padded_bs * width, heads * 512)
                expected[: active_bs * width] = (q[: active_bs * width] + 1).flatten(1)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                if active_bs == 0:
                    kernel.assert_not_called()
                else:
                    args, kwargs = kernel.call_args
                    self.assertEqual(args[0].shape, (active_bs, 4, width, 512))
                    self.assertEqual(args[1].ndim, 5 if is_nz else 4)
                    self.assertEqual(kwargs["input_layout"], "BNSD")
                    self.assertEqual(kwargs["actual_seq_qlen"], [width] * active_bs)
                    self.assertEqual(kwargs["actual_seq_kvlen"], lengths[:active_bs])
                    self.assertEqual(kwargs["block_table"].shape[0], active_bs)


if __name__ == "__main__":
    unittest.main()
