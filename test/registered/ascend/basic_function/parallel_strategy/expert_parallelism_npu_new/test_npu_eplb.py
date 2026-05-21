"""
Test EPLB (Expert Parallel Load Balancing) on Ascend NPU.

Tests dynamic EPLB with expert distribution recording and rebalancing.

GPU reference: test/manual/ep/test_eplb.py

NPU-specific adaptations:
- --attention-backend ascend
- --disable-cuda-graph
- --device npu
- HCCL and NPU-specific environment variables

Usage:
    python3 -m pytest test/registered/ascend/basic_function/parallel_strategy/expert_parallelism_npu_new/test_npu_eplb.py -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang as sgl
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestNPUDynamicEPLB(CustomTestCase):
    """Test dynamic EPLB on Ascend NPU.

    [Test Category] Expert Parallelism
    [Test Target] --enable-eplb, --ep-num-redundant-experts
    """

    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        with (
            envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False),
            envs.SGLANG_EXPERT_LOCATION_UPDATER_CANARY.override(True),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--device",
                    "npu",
                    "--tp-size",
                    "2",
                    "--dp-size",
                    "2",
                    "--ep-size",
                    "2",
                    "--enable-dp-attention",
                    "--mem-fraction-static",
                    "0.7",
                    "--max-running-requests",
                    "64",
                    "--disable-cuda-graph",
                    "--disable-radix-cache",
                    "--chunked-prefill-size",
                    "1024",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--enable-eplb",
                    "--ep-num-redundant-experts",
                    "4",
                    "--eplb-rebalance-num-iterations",
                    "50",
                    "--expert-distribution-recorder-buffer-size",
                    "50",
                    "--expert-distribution-recorder-mode",
                    "stat",
                    "--ep-dispatch-algorithm",
                    "static",
                    *cls.extra_args,
                ],
                env={
                    **os.environ,
                    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                    "STREAMS_PER_DEVICE": "32",
                    "HCCL_BUFFSIZE": "1024",
                    "HCCL_OP_EXPANSION_MODE": "AIV",
                    "SGLANG_NPU_USE_MLAPO": "0",
                    "SGLANG_NPU_USE_MULTI_STREAM": "1",
                    "TASK_QUEUE_ENABLE": "0",
                },
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        """Test MMLU accuracy with dynamic EPLB."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"MMLU score with dynamic EPLB: {metrics['score']}")
        self.assertGreater(
            metrics["score"],
            0.5,
            f"MMLU score {metrics['score']} is below threshold 0.5",
        )


class TestNPUDynamicEPLBMultiChunk(TestNPUDynamicEPLB):
    """Test dynamic EPLB with multi-chunk rebalancing on Ascend NPU.

    [Test Category] Expert Parallelism
    [Test Target] --eplb-rebalance-layers-per-chunk
    """

    extra_args = ["--eplb-rebalance-layers-per-chunk", "1"]


class TestNPUStaticEPLB(CustomTestCase):
    """Test static EPLB with expert distribution recording on Ascend NPU.

    [Test Category] Expert Parallelism
    [Test Target] expert distribution recording and init_expert_location
    """

    def test_save_expert_distribution_and_init_expert_location(self):
        """Test saving expert distribution and using it for initialization."""
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            engine_kwargs = dict(
                model_path=QWEN3_8B_WEIGHTS_PATH,
                trust_remote_code=True,
                ep_num_redundant_experts=4,
                enable_dp_attention=True,
                moe_a2a_backend="deepep",
                disable_cuda_graph=True,
                expert_distribution_recorder_mode="stat",
                tp_size=2,
                dp_size=2,
                log_level="info",
                attention_backend="ascend",
                device="npu",
            )

            print("Action: start engine")
            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)
            envs.SGLANG_EXPERT_LOCATION_UPDATER_CANARY.set(True)
            engine = sgl.Engine(
                **engine_kwargs,
                disable_overlap_schedule=True,
            )
            engine.start_expert_distribution_record()
            self._assert_engine_generate_correct(engine)

            print("Action: dump_expert_distribution_record")
            engine.dump_expert_distribution_record()
            snapshot_path = list(Path(tmp_dir).glob("*.pt"))[0]
            assert snapshot_path is not None
            print(f"{snapshot_path=}")

            print("Action: shutdown engine")
            engine.shutdown()
            del engine

            print("Action: start engine with init_expert_location")
            engine = sgl.Engine(
                **engine_kwargs,
                init_expert_location=str(snapshot_path),
                port=21000,
                ep_dispatch_algorithm="static",
            )
            self._assert_engine_generate_correct(engine)
            print("Action: shutdown engine")
            engine.shutdown()
            del engine

    def _assert_engine_generate_correct(self, engine: sgl.Engine):
        output = engine.generate(
            prompt=["1+1=2, 2+2=4", "One plus one is two, two plus two is four"],
            sampling_params=dict(max_new_tokens=8, temperature=0.0),
        )
        print(f"engine.generate {output=}")
        self.assertEqual(
            [x["text"] for x in output],
            [", 4+4=8,", ", four plus four is eight, eight"],
        )


if __name__ == "__main__":
    unittest.main()
