import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.npu_eval_accuracy_kit import NPUGSM8KMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)

_EAGLE3_BASE_ARGS = [
    "--dtype",
    "float16",
    "--attention-backend",
    "ascend",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
    "--mem-fraction-static",
    "0.7",
    "--disable-piecewise-cuda-graph",
]


class TestBasicSanityEagle3(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    NPUGSM8KMixin,
    CustomTestCase,
):
    """EAGLE3 sanity with draft weights pinned on CPU at load.

    [Test Category] Parameter
    [Test Target] --enable-draft-weights-cpu-backup
    """

    served_model_name = QWEN3_8B_WEIGHTS_PATH

    model = QWEN3_8B_WEIGHTS_PATH
    gsm8k_num_questions = 1400
    gsm8k_accuracy_thres = 0.74

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_8B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_EAGLE3_BASE_ARGS
            + [
                "--cuda-graph-max-bs",
                "4",
                "--enable-memory-saver",
                "--enable-draft-weights-cpu-backup",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestEagle3DraftWeightsCpuBackupRoundtrip(CustomTestCase):
    """Release/resume must restore draft weights: text unchanged and spec still accepts.

    [Test Category] Parameter
    [Test Target] --enable-draft-weights-cpu-backup
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_8B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_EAGLE3_BASE_ARGS
            + [
                "--disable-cuda-graph",
                "--enable-memory-saver",
                "--enable-draft-weights-cpu-backup",
                # WEIGHTS pause unloads target + draft together. Target has no
                # restore path unless this is also on; draft-only backup still
                # covers the draft worker via the OR in load_model_utils.
                "--enable-weights-cpu-backup",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, prompt, max_new_tokens=16):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["text"]

    def test_release_resume_preserves_draft(self):
        info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertTrue(info["enable_draft_weights_cpu_backup"])
        self.assertTrue(info["enable_memory_saver"])

        prompt = "The capital of France is"
        baseline = self._generate(prompt)
        self.assertTrue(baseline)

        requests.post(
            self.base_url + "/release_memory_occupation",
            json={"tags": ["weights"]},
            timeout=120,
        ).raise_for_status()
        requests.post(
            self.base_url + "/resume_memory_occupation",
            json={"tags": ["weights"]},
            timeout=120,
        ).raise_for_status()

        after = self._generate(prompt)
        self.assertEqual(baseline, after, "CPU backup must restore target+draft weights")

        info = requests.get(self.base_url + "/server_info", timeout=30).json()
        accept = info["internal_states"][0].get("avg_spec_accept_length")
        self.assertIsNotNone(accept, "spec metric missing; draft worker may be dead")
        self.assertGreater(
            accept,
            1.0,
            f"draft not speculating after resume (avg_spec_accept_length={accept})",
        )


if __name__ == "__main__":
    unittest.main()
