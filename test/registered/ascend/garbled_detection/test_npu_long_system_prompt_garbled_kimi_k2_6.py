import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.test_ascend_utils import (
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    KIMI_K2_6_W4A8_MODEL_PATH,
)
from sglang.test.ascend.test_garbled_detection_utils import GarbledDetectionBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="full-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed manually.",
)

MAX_TEST_ROUNDS_NUM = 3

# Environment variables matching the kimi_k2_6 performance test configuration
SERVER_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "DEEPEP_HCCL_BUFFSIZE": "1200",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

# Server arguments matching the kimi_k2_6 performance test configuration
SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    "16",
    "--mem-fraction-static",
    "0.895",
    "--max-running-requests",
    "208",
    "--chunked-prefill-size",
    "32768",
    "--context-length",
    "262144",
    "--max-prefill-tokens",
    "16384",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    "16",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "4",
    "8",
    "12",
    "13",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_6_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "5",
    "--speculative-draft-model-quantization",
    "unquant",
    "--prefill-delayer-max-delay-passes",
    "200",
    "--enable-prefill-delayer",
]

# A representative long Chinese system prompt (extracted from the original test)
# that exercises the model's ability to produce coherent output without garbled text.
_SYSTEM_PROMPT = (
    " #角色\n"
    "你是一个出色的金融产品分析师，擅长深入分析产品说明书，并从中提取关键信息。\n\n"
    "# 目标\n"
    "结合给定的规则材料，判断产品说明书中的金融产品的风险评级等级，并给出判断原因。\n\n"
    "# 技能\n"
    "深入阅读产品说明书，并深入理解规则材料，严格依据规则材料中的内容进行评级。\n\n"
    "# 规则\n"
    "1. 深入阅读产品说明书，并理解规则材料中的内容。\n"
    "2. 严格比较规则材料与产品说明书中的描述，基于规则材料的内容做出审慎的评级。\n"
    "3. 当材料中出现'全部资产类别合计敞口暴露不超过本计划净资产的x%'等内容时，"
    "若全部资产类别包含高波动资产，则表示高波动资产最高为x%，需使用该值给出最终的风险评级。\n"
    "4. 普通产品和FOF产品的R3、R4评级规则有所不同，主要体现在投资范围、投资集中度等方面，"
    "需要严格区分、谨慎分析。\n"
    "5. 将你的评级与管理人评级进行比较，按照孰严原则决定最终评级。\n\n"
    "# 限制\n"
    "1. 思考过程和判断原因需详细记录，不能省略步骤。\n"
    "2. 输出必须严格遵循JSON格式。\n"
)

_USER_PROMPT = (
    "请分析以下金融产品的风险评级：\n"
    "产品名称：稳健增长混合型理财产品A款\n"
    "投资范围：固定收益类资产占比不低于80%，权益类资产占比不超过20%\n"
    "管理人评级：R2\n"
    "请给出你的评级结果和判断原因。"
)


class TestStreamingGarbledDetection(GarbledDetectionBase):
    """Verify that kimi k2.6 model streaming chat completions with speculative decoding + long prompts produce no garbled output.

    [Test Category] API / Streaming Output Correctness
    [Test Target] Streaming chat completions with speculative decoding + long prompts
    """

    # --- Model-specific configuration ---
    model_path = KIMI_K2_6_W4A8_MODEL_PATH
    server_args = SERVER_ARGS
    server_envs = SERVER_ENVS
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _USER_PROMPT
    max_rounds = MAX_TEST_ROUNDS_NUM
    extra_payload = {
        "max_completion_tokens": 10000,
    }

    def test_streaming_no_garbled(self):
        self._run_streaming_no_garbled()


class TestStreamingGarbledDetectionWithReasoningParser(GarbledDetectionBase):
    """Verify that kimi k2.6 model streaming chat completions with explicit --reasoning-parser kimi_k2 and --tool-call-parser kimi_k2 produce no garbled output.

    [Test Category] API / Streaming Output Correctness
    [Test Target] Streaming chat completions with explicit reasoning parser + tool call parser
    """

    model_path = KIMI_K2_6_W4A8_MODEL_PATH

    server_args = SERVER_ARGS + [
        "--reasoning-parser",
        "kimi_k2",
        "--tool-call-parser",
        "kimi_k2",
    ]
    server_envs = SERVER_ENVS
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _USER_PROMPT
    max_rounds = MAX_TEST_ROUNDS_NUM
    extra_payload = {
        "max_tokens": 10000,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    def test_streaming_no_garbled_with_reasoning_parser(self):
        self._run_streaming_no_garbled()


if __name__ == "__main__":
    unittest.main()
