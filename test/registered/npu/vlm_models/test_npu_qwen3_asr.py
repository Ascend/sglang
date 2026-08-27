import io
import re
import string
import unittest

import requests
from datasets import Audio
from modelscope import MsDataset

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    AUDIO_DATASETS_LIBRISPEECH_ASR_PATH,
    QWEN3_ASR_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

try:
    import datasets.features.features as _dff

    _orig_generate_from_dict = _dff.generate_from_dict

    def _generate_from_dict_with_default_dtype(obj):
        if isinstance(obj, dict) and obj.get("_type") == "Value" and "dtype" not in obj:
            obj = {**obj, "dtype": "string"}
        return _orig_generate_from_dict(obj)

    _dff.generate_from_dict = _generate_from_dict_with_default_dtype
except Exception:
    pass

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)

# Test configuration
SUBSET = "clean"
SPLIT = "test"
LIMIT = 20
LANGUAGE = "en"

# ---- Self-implemented WER utilities (replacing evalscope) ----


def _edit_distance(ref: list, hyp: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def normalize_text(text: str, language: str = "en") -> str:
    """Normalize text for WER computation: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    # Remove punctuation except apostrophes within words (e.g. don't, it's)
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(references: list, predictions: list, language: str = "en") -> float:
    """Compute Word Error Rate between reference and prediction texts."""
    total_edit = 0
    total_words = 0
    for ref, pred in zip(references, predictions):
        ref_words = ref.split()
        pred_words = pred.split()
        total_edit += _edit_distance(ref_words, pred_words)
        total_words += len(ref_words)
    if total_words == 0:
        return 0.0
    return total_edit / total_words


class TestQwen3ASR(CustomTestCase):
    """Testcase: Verify the ASR accuracy of Qwen/Qwen3-ASR-1.7B on LibriSpeech dataset.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-ASR-1.7B
    """

    model = QWEN3_ASR_WEIGHTS_PATH
    extra_args = [
        "--tp-size",
        1,
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--trust-remote-code",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.extra_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def transcribe(self, audio_bytes: bytes) -> str:
        """Call sglang /v1/audio/transcriptions endpoint."""
        response = requests.post(
            f"{self.base_url}/audio/transcriptions",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"model": "default"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("text", "")

    def test_librispeech_asr(self):
        """Run ASR evaluation on LibriSpeech dataset and verify WER."""
        print(
            f"Loading dataset: {AUDIO_DATASETS_LIBRISPEECH_ASR_PATH} [{SUBSET}/{SPLIT}]"
        )
        dataset = MsDataset.load(
            dataset_name=AUDIO_DATASETS_LIBRISPEECH_ASR_PATH,
            subset_name=SUBSET,
            split=SPLIT,
            trust_remote_code=True,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()
        dataset = dataset.cast_column("audio", Audio(decode=False))
        total = min(LIMIT, len(dataset))
        print(f"Total samples: {total}")

        references = []
        predictions = []

        for i in range(total):
            record = dataset[i]
            audio_bytes = record["audio"]["bytes"]
            reference = record.get("text", record.get("transcript", ""))

            prediction = self.transcribe(audio_bytes)

            norm_ref = normalize_text(reference, LANGUAGE)
            norm_pred = normalize_text(prediction, LANGUAGE)

            references.append(norm_ref)
            predictions.append(norm_pred)

            sample_wer = wer([norm_ref], [norm_pred], LANGUAGE)
            print(f"[{i + 1}/{total}] WER: {sample_wer:.4f}")

        overall_wer = wer(references, predictions, LANGUAGE)
        print(f"\n{'=' * 60}")
        print(f"Overall WER ({total} samples): {overall_wer:.4f}")
        print(f"{'=' * 60}")

        # Assert WER is within acceptable range
        self.assertLess(
            overall_wer, 0.05, f"WER {overall_wer:.4f} exceeds threshold 0.05"
        )


if __name__ == "__main__":
    unittest.main()
