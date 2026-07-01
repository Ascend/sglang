import unittest
from urllib.parse import urlparse

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MINIMAX_M2_5_W8A8_MODEL_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    write_github_step_summary,
)

register_npu_ci(est_time=1500, suite="full-8-npu-a3", nightly=True)


class TestMiniMaxM25(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify MiniMax-M2.5 (W8A8) end-to-end on Ascend NPU.

    [Test Category] Model
    [Test Target] Eco-Tech/MiniMax-M2.5-w8a8-QuaRot
    """

    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    accuracy = 0.90
    timeout_for_server_launch = 3000

    # NPU EP requires deepep backend (--moe-a2a-backend=deepep).
    # When deepep is enabled, ep_size is auto-set to tp_size (server_args.py
    # _handle_a2a_moe), so --ep-size is not needed.
    # --dp-size / --enable-dp-attention are for DP-Attention mode (not EP),
    # and add extra attention workspace memory (server_args.py:1558). Omit
    # them to avoid OOM on the 230B W8A8 model (CI run 28494887328 OOM with
    # only 244 MiB free when enable-dp-attention was on).
    # W8A8 model MUST declare --quantization modelslim, otherwise the INT32-
    # packed weights are fed to the BF16 GMM kernel and fail with
    # aclnnGroupedMatmulWeightNz error 161002 (CI run 28452151866).
    # DEEP_NORMAL_MODE_USE_INT8_QUANT=1 is required for W8A8 deepep dispatch
    # and is mutually exclusive with SGLANG_DEEPEP_BF16_DISPATCH (the latter
    # is for bf16 unquant models; see qwen3_next_mtp.py override logic where
    # BF16_DISPATCH=True forces INT8_QUANT=False).
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--quantization",
        "modelslim",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--reasoning-parser",
        "minimax-append-think",
        "--tool-call-parser",
        "minimax-m2",
        "--max-running-requests",
        "32",
        "--chunked-prefill-size",
        "-1",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
        "--weight-loader-prefetch-checkpoints",
    ]

    # deepep needs larger HCCL buffer and INT8 quant dispatch for W8A8.
    # DEEP_NORMAL_MODE_USE_INT8_QUANT conflicts with SGLANG_DEEPEP_BF16_DISPATCH;
    # W8A8 quantized model uses the INT8 path (see deepep.py:_update_int8_quant_env
    # where use_fp8=True -> INT8_QUANT=1, and unquant.py where
    # use_fp8 = not SGLANG_DEEPEP_BF16_DISPATCH).
    env = {
        **GSM8KAscendMixin.env,
        "HCCL_BUFFSIZE": "2048",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_WARMUP_TIMEOUT": "3600",
        "TRANSFORMERS_VERBOSITY": "error",
    }

    def test_bs_1_speed(self):
        # NPU TP=8 + deepep: bs=1 MoE all-to-all not amortized (~7.4 token/s baseline).
        url = urlparse(DEFAULT_URL_FOR_TEST)
        args = BenchArgs(
            host=url.hostname,
            port=int(url.port),
            max_new_tokens=2048,
        )
        acc_length, speed = send_one_prompt(args, print_output=False)

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (minimax-m25-w8a8, TP=8 + deepep)\n"
                f"- speed: {speed:.2f} token/s\n"
                f"- accept_length: {acc_length:.2f}\n"
            )

        self.assertGreater(speed, 5, f"bs=1 speed {speed:.2f} below 5 token/s")


if __name__ == "__main__":
    unittest.main()
