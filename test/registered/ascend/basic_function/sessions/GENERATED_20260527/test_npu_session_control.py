import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=87, suite="nightly-2-npu-a3", nightly=True)


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


class TestNPUSessionControl(CustomTestCase):
    """Test session control functionality on NPU.

    [Test Category] Feature
    [Test Target] Session branching, backtracking, control operations
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--disable-piecewise-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_session_control(self, gen_len=12):
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
            "A brief history about that city is",
        ]
        tokenizer = get_tokenizer(self.model)
        chunks_ids = [tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()
        rid = None

        ret = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "session_id": session_id},
        )
        self.assertNotEqual(ret.status_code, 200)

        first_rid = None
        outputs_from_session = []
        logprobs_from_session = []
        cur_logprob_start_len = 0
        for i, chunk_ids in enumerate(chunks_ids):
            max_new_tokens = gen_len if i > 0 else 1
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session_params": {
                        "id": session_id,
                        "rid": rid,
                        "offset": -1,
                        "replace": True,
                    },
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                    "return_logprob": True,
                    "logprob_start_len": cur_logprob_start_len - 1,
                },
            ).json()
            rid = response["meta_info"]["id"]
            if i == 0:
                first_rid = rid
            if i > 0:
                outputs_from_session.append(response["text"])
                logprobs_from_session.extend(
                    [
                        round(sublist[0], 2)
                        for sublist in response["meta_info"]["output_token_logprobs"]
                    ]
                )
            cur_logprob_start_len += len(chunk_ids) + max_new_tokens

        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunk_ids,
                "session_params": {
                    "id": session_id,
                    "rid": rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": cur_logprob_start_len + len(chunk_ids),
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        cur_logprob_start_len = 0
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": first_rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
                "logprob_start_len": cur_logprob_start_len,
            },
        ).json()
        outputs_from_session.append(response["text"])
        logprobs_from_session.extend(
            [
                round(sublist[0], 2)
                for sublist in response["meta_info"]["output_token_logprobs"]
            ]
        )

        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(ret.status_code, 200)

        ret = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[-1],
                "session_params": {
                    "id": session_id,
                    "rid": first_rid,
                    "offset": -1,
                    "replace": True,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
            },
        )
        self.assertNotEqual(ret.status_code, 200)

        requests.post(self.base_url + "/flush_cache")

        input_ids_first_req = None
        input_ids = []
        outputs_normal = []
        logprobs_normal = []
        for i, chunk_ids in enumerate(chunks_ids):
            input_ids += chunk_ids
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (gen_len if i > 0 else 1),
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                    "return_logprob": True,
                },
            ).json()
            if i > 0:
                output_ids = tokenizer.encode(response["text"])
                if output_ids[0] == tokenizer.bos_token_id:
                    output_ids = output_ids[1:]
                input_ids += output_ids[:-1]
                outputs_normal.append(response["text"])
                logprobs_normal.extend(
                    [
                        round(sublist[0], 2)
                        for sublist in response["meta_info"]["output_token_logprobs"]
                    ]
                )
            if i == 0:
                input_ids_first_req = input_ids.copy()

        input_ids_first_req += chunks_ids[-1]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids_first_req,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "return_logprob": True,
            },
        ).json()
        outputs_normal.append(response["text"])
        logprobs_normal.extend(
            [
                round(sublist[0], 2)
                for sublist in response["meta_info"]["output_token_logprobs"]
            ]
        )

        self.assertEqual(outputs_from_session, outputs_normal)
        assert len(logprobs_from_session) == len(
            logprobs_normal
        ), "logprobs must have equal length"
        for a, b in zip(logprobs_from_session, logprobs_normal):
            assert abs(a - b) <= 0.15, f"logprobs {a} and {b} differ by more than 0.15"

    def test_session_control_with_branching(self):
        root_prompt = "First, let me explain in one sentence about AI"
        chunks_per_step = [
            [
                "Then, briefly, the positive side of AI is",
                "But, briefly, AI could be harmful to human",
            ],
            ["For example", "For example"],
        ]
        self.run_session_control_with_branching(
            root_prompt=root_prompt, chunks_per_step=chunks_per_step, gen_len=8
        )

        root_prompt = "I have three apples."
        chunks_per_step = [
            ["I then give one apple to my friend", "My friend give me another apple."],
            ["I still have", "I now have"],
        ]
        self.run_session_control_with_branching(
            root_prompt=root_prompt, chunks_per_step=chunks_per_step, gen_len=8
        )

    def run_session_control_with_branching(
        self, root_prompt, chunks_per_step, gen_len=16
    ):
        for x in chunks_per_step:
            assert len(x) == len(chunks_per_step[0])

        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000},
        ).json()

        outputs_from_session = []
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": root_prompt,
                "session_params": {
                    "id": session_id,
                    "rid": None,
                    "offset": 0,
                    "replace": False,
                },
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        rid_per_branch = [response["meta_info"]["id"]] * len(chunks_per_step[0])
        outputs_from_session.append(response["text"])

        for chunks_for_branches in chunks_per_step:
            for j, chunk in enumerate(chunks_for_branches):
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": chunk,
                        "session_params": {
                            "id": session_id,
                            "rid": rid_per_branch[j],
                            "offset": 0,
                            "replace": False,
                        },
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": gen_len,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                    },
                ).json()
                rid = response["meta_info"]["id"]
                rid_per_branch[j] = rid
                outputs_from_session.append(response["text"])

        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        assert ret.status_code == 200

        requests.post(self.base_url + "/flush_cache")

        outputs_normal = []
        input_texts = [root_prompt] * len(chunks_per_step[0])
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": root_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": gen_len,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
            },
        ).json()
        outputs_normal.append(response["text"])
        input_texts = [x + response["text"] for x in input_texts]

        for chunks_for_branches in chunks_per_step:
            for j, chunk in enumerate(chunks_for_branches):
                input_texts[j] += chunk
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": input_texts[j],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": gen_len,
                            "no_stop_trim": True,
                            "skip_special_tokens": False,
                        },
                    },
                ).json()
                outputs_normal.append(response["text"])
                input_texts[j] += response["text"]

        assert (
            outputs_from_session == outputs_normal
        ), f"outputs_from_session: {outputs_from_session}, outputs_normal: {outputs_normal}"


if __name__ == "__main__":
    unittest.main()
