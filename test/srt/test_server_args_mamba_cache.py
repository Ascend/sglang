import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs


class TestMambaCacheServerArgs(unittest.TestCase):
    def _make_server_args(self, arch):
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.model_path = "/models/mock"
        server_args.page_size = 128
        server_args.disable_radix_cache = False
        server_args.disable_overlap_schedule = False
        server_args.mamba_scheduler_strategy = "no_buffer"
        server_args.mamba_track_interval = 256
        server_args.speculative_algorithm = None
        server_args.speculative_num_draft_tokens = None
        server_args.attention_backend = "ascend"
        server_args.enable_flashinfer_allreduce_fusion = False
        server_args.enforce_disable_flashinfer_allreduce_fusion = False
        server_args.tp_size = 1
        server_args.enable_dp_attention = False
        server_args.nnodes = 1
        server_args.moe_a2a_backend = "none"
        server_args.moe_runner_backend = "auto"
        server_args._quantization_explicitly_unset = False
        server_args.quantization = None
        server_args.get_model_config = lambda: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch])
        )
        return server_args

    @patch("sglang.srt.server_args.is_sm100_supported", return_value=False)
    @patch("sglang.srt.server_args.is_sm90_supported", return_value=False)
    @patch("sglang.srt.server_args.is_npu", return_value=True)
    def test_qwen35_text_arches_handle_mamba_cache_page_size(
        self, _mock_npu, _mock_sm90, _mock_sm100
    ):
        for arch in [
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForCausalLMMTP",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ]:
            with self.subTest(arch=arch):
                server_args = self._make_server_args(arch)

                ServerArgs._handle_model_specific_adjustments(server_args)

                self.assertTrue(server_args.uses_mamba_radix_cache)
                self.assertEqual(server_args.mamba_scheduler_strategy, "extra_buffer")
                self.assertEqual(server_args.page_size, 128)
                self.assertFalse(server_args.disable_overlap_schedule)


if __name__ == "__main__":
    unittest.main()
