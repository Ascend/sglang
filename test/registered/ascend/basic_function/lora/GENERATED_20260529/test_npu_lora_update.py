import multiprocessing as mp
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)

PROMPTS = [
    "The capital of France is",
    "AI is a field of computer science focused on",
    "What are the main components of a computer?",
]

MEM_FRACTION_STATIC = 0.3


class OperationType(Enum):
    LOAD = "load"
    UNLOAD = "unload"
    FORWARD = "forward"


@dataclass
class Operation:
    type: OperationType
    data: Optional[Any]
    expected_error: Optional[str] = None
    expected_implicit_evictions: Optional[set] = None


@dataclass
class TestCase:
    description: str
    max_loras_per_batch: int
    all_adapters: List[str]
    op_sequence: List[Operation]
    initial_adapters: Optional[List[str]] = None
    enable_lora: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    lora_target_modules: Optional[List] = None
    max_new_tokens: int = 32
    max_loaded_loras: Optional[int] = None


def create_batch_data(adapters: Union[str, list]) -> List[tuple]:
    if not isinstance(adapters, list):
        adapters = [adapters]
    return [(prompt, adapter) for prompt in PROMPTS for adapter in adapters]


LORA_A = "lora_a"
LORA_B = "lora_b"

BASIC_TESTS = [
    TestCase(
        description="dynamic lora update with initial lora_paths",
        max_loras_per_batch=2,
        all_adapters=[LORA_A, LORA_B],
        initial_adapters=[LORA_A, f"{LORA_B}={LORA_B}"],
        op_sequence=[
            Operation(
                type=OperationType.LOAD, data=LORA_A, expected_error="already loaded"
            ),
            Operation(type=OperationType.UNLOAD, data=LORA_A),
            Operation(type=OperationType.LOAD, data=LORA_A),
            Operation(
                type=OperationType.FORWARD, data=create_batch_data([LORA_A, LORA_B])
            ),
            Operation(type=OperationType.UNLOAD, data=LORA_B),
            Operation(type=OperationType.FORWARD, data=create_batch_data(LORA_A)),
            Operation(type=OperationType.FORWARD, data=create_batch_data(LORA_B)),
            Operation(
                type=OperationType.LOAD, data=LORA_B, expected_error="already loaded"
            ),
            Operation(
                type=OperationType.FORWARD, data=create_batch_data([LORA_A, LORA_B])
            ),
            Operation(type=OperationType.UNLOAD, data=LORA_A),
            Operation(type=OperationType.FORWARD, data=create_batch_data(LORA_A)),
            Operation(type=OperationType.UNLOAD, data=LORA_B),
            Operation(type=OperationType.FORWARD, data=create_batch_data(None)),
        ],
    ),
    TestCase(
        description="dynamic lora update without initial lora_paths",
        enable_lora=True,
        max_lora_rank=64,
        lora_target_modules=["all"],
        max_loras_per_batch=2,
        all_adapters=[LORA_A, LORA_B],
        op_sequence=[
            Operation(type=OperationType.LOAD, data=LORA_A),
            Operation(type=OperationType.LOAD, data=LORA_B),
            Operation(
                type=OperationType.LOAD, data=LORA_A, expected_error="already loaded"
            ),
            Operation(type=OperationType.UNLOAD, data=LORA_A),
            Operation(type=OperationType.LOAD, data=LORA_A),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data([LORA_A, LORA_B, None]),
            ),
            Operation(type=OperationType.UNLOAD, data=LORA_A),
            Operation(type=OperationType.FORWARD, data=create_batch_data(LORA_A)),
            Operation(
                type=OperationType.FORWARD, data=create_batch_data([None, LORA_B, None])
            ),
            Operation(type=OperationType.UNLOAD, data=LORA_B),
            Operation(type=OperationType.FORWARD, data=create_batch_data(None)),
        ],
    ),
]

ALL_TESTS = BASIC_TESTS


class TestNPULoRAUpdate(CustomTestCase):
    """Test dynamic LoRA loading/unloading on NPU.

    [Test Category] Feature
    [Test Target] LoRA dynamic update, load_lora_adapter, unload_lora_adapter
    """

    def _run_operation_sequence(
        self,
        base: str,
        initial_adapters: List,
        op_sequence: List[Operation],
        max_loras_per_batch: int,
        max_loaded_loras: Optional[int] = None,
        enable_lora: Optional[bool] = None,
        max_lora_rank: Optional[int] = None,
        lora_target_modules: Optional[List] = None,
        max_new_tokens: int = 32,
    ) -> List:
        forward_outputs = []
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
            "--max-loras-per-batch",
            str(max_loras_per_batch),
        ]
        if enable_lora:
            other_args.append("--enable-lora")
        if max_lora_rank is not None:
            other_args.extend(["--max-lora-rank", str(max_lora_rank)])
        if lora_target_modules is not None:
            other_args.extend(["--lora-target-modules"] + lora_target_modules)
        if max_loaded_loras is not None:
            other_args.extend(["--max-loaded-loras", str(max_loaded_loras)])

        if initial_adapters:
            other_args.append("--lora-path")
            for adapter in initial_adapters:
                if adapter == LORA_A:
                    other_args.append(
                        f"{LORA_A}={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}"
                    )
                elif adapter == LORA_B or adapter.startswith(LORA_B):
                    other_args.append(
                        f"{LORA_B}={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}"
                    )

        process = popen_launch_server(
            base,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            expected_adapters = set()
            if initial_adapters:
                for adapter in initial_adapters:
                    name = adapter.split("=")[0] if "=" in adapter else adapter
                    expected_adapters.add(name)

            for op in op_sequence:
                op_type = op.type
                data = op.data
                expected_error = op.expected_error
                print(f"Running operation: {op_type} --- data: {data}")

                if op_type == OperationType.LOAD:
                    adapter_name = data
                    adapter_path = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
                    response = requests.post(
                        DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
                        json={
                            "lora_name": adapter_name,
                            "lora_path": adapter_path,
                            "pinned": False,
                        },
                    )
                    if expected_error:
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(expected_error, response.text)
                    else:
                        self.assertTrue(
                            response.ok, f"Failed to load adapter: {response.text}"
                        )
                        expected_adapters.add(adapter_name)
                        loaded_adapters = set(response.json()["loaded_adapters"])
                        self.assertEqual(loaded_adapters, expected_adapters)

                elif op_type == OperationType.UNLOAD:
                    expected_adapters.remove(data)
                    response = requests.post(
                        DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
                        json={"lora_name": data},
                    )
                    self.assertTrue(
                        response.ok, f"Failed to unload adapter: {response.text}"
                    )
                    loaded_adapters = set(response.json()["loaded_adapters"])
                    self.assertEqual(loaded_adapters, expected_adapters)

                elif op_type == OperationType.FORWARD:
                    prompts, adapters = zip(*data)
                    lora_paths_list = list(adapters)
                    response = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": list(prompts),
                            "lora_path": lora_paths_list,
                            "sampling_params": {
                                "temperature": 0,
                                "max_new_tokens": max_new_tokens,
                            },
                        },
                    )
                    if expected_error:
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(expected_error, response.text)
                    else:
                        self.assertTrue(
                            response.ok, f"Failed to generate: {response.text}"
                        )
                        output = [r["text"] for r in response.json()]
                        forward_outputs.append(output)
                        expected_adapters.update(
                            [a for a in lora_paths_list if a is not None]
                        )
        finally:
            kill_process_tree(process.pid)

        return forward_outputs

    def test_dynamic_lora_update_server(self):
        test_cases = BASIC_TESTS if is_in_ci() else ALL_TESTS
        for case_idx, test_case in enumerate(test_cases, start=1):
            print(f"Starting test case {case_idx}: {test_case.description}")
            forward_outputs = self._run_operation_sequence(
                base=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
                initial_adapters=test_case.initial_adapters,
                enable_lora=test_case.enable_lora,
                max_loras_per_batch=test_case.max_loras_per_batch,
                max_loaded_loras=test_case.max_loaded_loras,
                op_sequence=test_case.op_sequence,
                max_new_tokens=test_case.max_new_tokens,
                max_lora_rank=test_case.max_lora_rank,
                lora_target_modules=test_case.lora_target_modules,
            )
            forward_ops = [
                x
                for x in test_case.op_sequence
                if x.type == OperationType.FORWARD and x.expected_error is None
            ]
            if not forward_ops:
                continue
            self.assertEqual(len(forward_outputs), len(forward_ops))


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main(warnings="ignore")
