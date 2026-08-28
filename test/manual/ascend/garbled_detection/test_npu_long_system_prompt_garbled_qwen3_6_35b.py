import logging
import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_6_35B_A3B_WEIGHTS_PATH,
)
from sglang.test.ascend.test_garbled_detection_utils import GarbledDetectionBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MAX_TEST_ROUNDS_NUM = 100

# Environment variables matching the original start_server.sh configuration
SERVER_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "3000",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "GDN_ATTN_BACKEND_TRITON": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
}

# Server arguments matching the original start_server.sh configuration
SERVER_ARGS = [
    # "--base-gpu-id",
    # "6",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    "2",
    "--chunked-prefill-size",
    "16384",
    "--max-prefill-tokens",
    "16384",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.75",
    "--enable-prefill-delayer",
    "--prefill-delayer-max-delay-passes",
    "30",
    "--cuda-graph-bs",
    "1",
    "2",
    "4",
    "6",
    "8",
    "10",
    "16",
    "20",
    "24",
    "--max-running-requests",
    "24",
    "--context-length",
    "262144",
    "--mm-attention-backend",
    "ascend_attn",
    "--max-mamba-cache-size",
    "120",
    "--enable-multimodal",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--stream-response-default-include-usage",
    "--enable-metrics",
    # "--disable-radix-cache",
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
    """Verify that GDN model streaming chat completions with speculative decoding + long prompts produce no garbled output.

    [Test Category] API / Streaming Output Correctness
    [Test Target] Streaming chat completions with speculative decoding + long prompts
    """

    # --- Model-specific configuration ---
    model_path = QWEN3_6_35B_A3B_WEIGHTS_PATH
    server_args = SERVER_ARGS
    server_envs = SERVER_ENVS
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _USER_PROMPT
    max_rounds = MAX_TEST_ROUNDS_NUM
    extra_payload = {
        "max_completion_tokens": 10000,
        "frequency_penalty": 1.5,
        "presence_penalty": 1.5,
        "temperature": 0.1,
        "top_k": 10,
        "top_p": 0.6,
    }

    def test_streaming_no_garbled(self):
        self._run_streaming_no_garbled()


class TestStreamingGarbledDetectionWithReasoningParser(GarbledDetectionBase):
    """Verify that GDN model streaming chat completions with explicit --reasoning-parser qwen3 and --tool-call-parser qwen3_coder produce no garbled output.

    [Test Category] API / Streaming Output Correctness
    [Test Target] Streaming chat completions with explicit reasoning parser + tool call parser
    """

    model_path = QWEN3_6_35B_A3B_WEIGHTS_PATH

    server_args = SERVER_ARGS + [
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
    ]
    server_envs = SERVER_ENVS
    system_prompt = _SYSTEM_PROMPT
    user_prompt = _USER_PROMPT
    max_rounds = MAX_TEST_ROUNDS_NUM
    extra_payload = {
        "max_completion_tokens": 10000,
        "frequency_penalty": 1.5,
        "presence_penalty": 1.5,
        "temperature": 0.1,
        "top_k": 10,
        "top_p": 0.6,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    def test_streaming_no_garbled_with_reasoning_parser(self):
        self._run_streaming_no_garbled()


if __name__ == "__main__":
    unittest.main()
