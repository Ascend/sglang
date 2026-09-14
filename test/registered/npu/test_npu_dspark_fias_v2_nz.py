"""Device tests for MLA NZ scatter, FIAS V2 and changing graph KV lengths."""

import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import is_fia_nz
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=60, suite="nightly-1-npu-a3", nightly=True)


class TestDSparkFIASV2NZ(CustomTestCase):
    def test_scatter_attention_and_graph_replay(self):
        torch.manual_seed(42)
        page_size, query_len, heads = 128, 8, 16
        kv = (torch.randn(512, 1, 512) * 0.1).bfloat16()
        rope = (torch.randn(512, 1, 64) * 0.1).bfloat16()
        q = (torch.randn(16, heads, 512) * 0.1).bfloat16()
        q_rope = (torch.randn(16, heads, 64) * 0.1).bfloat16()
        q_npu, q_rope_npu = q.npu(), q_rope.npu()
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=heads,
            tp_k_head_num=1,
            tp_v_head_num=1,
            v_head_dim=512,
            scaling=576**-0.5,
        )
        for is_nz in (False, True):
            with (
                self.subTest(is_nz=is_nz),
                envs.SGLANG_USE_FIA_NZ.override(is_nz),
                envs.SGLANG_NPU_USE_MLAPO.override(False),
            ):
                is_fia_nz.cache_clear()
                self.addCleanup(is_fia_nz.cache_clear)
                pool = NPUMLATokenToKVPool(
                    size=512,
                    page_size=page_size,
                    dtype=torch.bfloat16,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    layer_num=1,
                    device="npu",
                    enable_memory_saver=False,
                )
                # The ordinary writer must populate physical NZ rows without MLAPO.
                pool.set_kv_buffer(
                    layer,
                    torch.arange(512, device="npu", dtype=torch.int32),
                    kv.npu(),
                    rope.npu(),
                )
                ids = torch.tensor([3, 0, 2, 1, 3], device="npu", dtype=torch.int32)
                for cache, expected in (
                    (pool.get_key_buffer(0), kv),
                    (pool.get_value_buffer(0), rope),
                ):
                    actual = gather_mla_cache_pages(cache, ids, is_nz=is_nz).cpu()
                    torch.testing.assert_close(
                        actual,
                        expected.reshape(4, 128, 1, -1)[ids.cpu().long()],
                        rtol=0,
                        atol=0,
                    )

                backend = object.__new__(AscendAttnBackend)
                backend.use_mla = backend.use_fias_v2_bsnd = True
                backend.graph_mode = True
                backend.token_to_kv_pool = pool
                backend.page_size = page_size
                backend.kv_lora_rank = 512
                backend.qk_rope_head_dim = 64
                backend.q_head_num_padding = backend.tp_q_head_num = heads
                backend.speculative_num_draft_tokens = query_len
                backend.mtp_mask = torch.ones(
                    (2048, 2048), dtype=torch.bool, device="npu"
                ).triu_(1)
                backend.forward_metadata = SimpleNamespace(
                    seq_lens_cpu_int=None,
                    seq_lens_cpu_list=[127, 129],
                    block_tables=torch.tensor(
                        [[0, 1], [2, 3]], device="npu", dtype=torch.int32
                    ),
                )
                batch = SimpleNamespace(num_token_non_padded_cpu=16)

                def forward():
                    return backend.forward_mtp(
                        q_npu, None, None, layer, batch, False, q_rope=q_rope_npu
                    )

                for _ in range(2):
                    forward()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, auto_dispatch_capture=True):
                    captured = forward()
                for lengths in ([127, 129], [128, 255], [129, 128]):
                    graph.update(cpu_update_input=[{"actual_seq_kvlen": list(lengths)}])
                    graph.replay()
                    torch.npu.synchronize()
                    replay = captured.cpu().clone()
                    backend.forward_metadata.seq_lens_cpu_list = list(lengths)
                    eager = forward().cpu()
                    torch.testing.assert_close(replay, eager, rtol=0.01, atol=0.002)
                    expected = []
                    for b, length in enumerate(lengths):
                        keys = kv[b * 256 : b * 256 + length, 0].float()
                        pos = rope[b * 256 : b * 256 + length, 0].float()
                        query = q[b * 8 : (b + 1) * 8].float().transpose(0, 1)
                        query_rope = q_rope[b * 8 : (b + 1) * 8].float().transpose(0, 1)
                        scores = (query @ keys.T + query_rope @ pos.T) * layer.scaling
                        mask = (
                            torch.arange(length)[None, :]
                            > (length - query_len + torch.arange(query_len))[:, None]
                        )
                        scores.masked_fill_(mask, float("-inf"))
                        expected.append(
                            (scores.softmax(-1) @ keys).transpose(0, 1).reshape(8, -1)
                        )
                    torch.testing.assert_close(
                        eager.float(), torch.cat(expected), rtol=0.02, atol=0.003
                    )


if __name__ == "__main__":
    unittest.main()
