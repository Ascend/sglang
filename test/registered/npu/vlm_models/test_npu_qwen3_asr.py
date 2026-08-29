import io
import logging
import unittest

import requests
from datasets import Audio
from modelscope import MsDataset
from whisper_normalizer.english import EnglishTextNormalizer

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

logger = logging.getLogger(__name__)

try:
    import datasets.features.features as _dff

    _orig_generate_from_dict = _dff.generate_from_dict

    def _generate_from_dict_with_default_dtype(obj):
        if isinstance(obj, dict) and obj.get("_type") == "Value" and "dtype" not in obj:
            obj = {**obj, "dtype": "string"}
        return _orig_generate_from_dict(obj)

    _dff.generate_from_dict = _generate_from_dict_with_default_dtype
except Exception as e:  # pragma: no cover
    logger.debug("datasets.features.generate_from_dict patch failed: %s", e)

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)

register_npu_ci(
    est_time=400,
    suite="full-test-npu-perf-1",
    nightly=True,
)

# Test configuration
SUBSET = "clean"
SPLIT = "test"
LIMIT = 2620
LANGUAGE = "en"

# Qwen3-ASR-1.7B WER on LibriSpeech test-clean (full set) is 1.63%
# (Qwen3-ASR Technical Report, arXiv:2601.21337). The CI gate asserts
# overall_wer < WER_BASELINE + WER_TOLERANCE
WER_BASELINE = 0.0163
WER_TOLERANCE = 0.000326

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


# Official Qwen3-ASR evaluation normalizes both references and predictions with
# the Whisper EnglishTextNormalizer (the Qwen3-ASR / reference evals use the
# exact same class). It lowercases, strips punctuation, expands contractions
# ("don't" -> "do not", "I'm" -> "I am"), removes filler words (uh/um/mm-hmm),
# normalizes numbers and standardizes British->American spellings, so a correct
# transcription is never penalized for using a contraction, a period, or an
# -our/-re spelling. Naive punctuation removal is NOT aligned with this
# baseline: "I DON'T KNOW" would become "i don t know" (splitting the
# contraction) and inflate WER.
#
# The official Qwen3-ASR / evalscope evaluation uses the whisper_normalizer
# package's EnglishTextNormalizer, which loads the full 1739-entry Whisper
# english.json spelling map at construction time (no argument needed). The CI
# workflow installs whisper_normalizer (see _npu-single-node-test-stage.yml),
# so using it here keeps spelling standardization exactly aligned with the
# reference eval.

# whisper_normalizer's EnglishTextNormalizer needs no argument and loads the
# full 1739-entry Whisper english.json spelling map itself.
_NORMALIZER_SOURCE = "whisper_normalizer+full"

_EN_NORMALIZER = EnglishTextNormalizer()
_SPELLING_MAP_SIZE = len(_EN_NORMALIZER.standardize_spellings.mapping)
logger.info(
    "Built EnglishTextNormalizer (source=%s, %d-entry spelling map)",
    _NORMALIZER_SOURCE,
    _SPELLING_MAP_SIZE,
)

# Log the selected normalizer at import time so CI logs make the active path
# obvious.
logger.info("Normalizer source: %s", _NORMALIZER_SOURCE)


def normalize_text(text: str, language: str = "en") -> str:
    """Whisper-style normalization aligned with the official Qwen3-ASR eval."""
    normalized = _EN_NORMALIZER(text)
    logger.debug("normalize_text(language=%s): %r -> %r", language, text, normalized)
    return normalized


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
        normalizer_class = (
            f"{type(_EN_NORMALIZER).__module__}.{type(_EN_NORMALIZER).__name__}"
        )
        logger.info(
            "Normalizer source: %s (class: %s, map entries: %d)",
            _NORMALIZER_SOURCE,
            normalizer_class,
            _SPELLING_MAP_SIZE,
        )
        logger.info(
            "Loading dataset: %s [%s/%s]",
            AUDIO_DATASETS_LIBRISPEECH_ASR_PATH,
            SUBSET,
            SPLIT,
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
        logger.info("Total samples: %d", total)

        references = []
        predictions = []
        modified_refs = 0
        modified_preds = 0

        for i in range(total):
            record = dataset[i]
            audio_bytes = record["audio"]["bytes"]
            reference = record.get("text", record.get("transcript", ""))

            prediction = self.transcribe(audio_bytes)

            norm_ref = normalize_text(reference, LANGUAGE)
            norm_pred = normalize_text(prediction, LANGUAGE)
            if norm_ref != reference:
                modified_refs += 1
            if norm_pred != prediction:
                modified_preds += 1
            if i == 0:
                logger.info("First sample raw ref:   %r", reference)
                logger.info("First sample norm ref:  %r", norm_ref)
                logger.info("First sample raw pred:  %r", prediction)
                logger.info("First sample norm pred: %r", norm_pred)

            references.append(norm_ref)
            predictions.append(norm_pred)

            sample_wer = wer([norm_ref], [norm_pred], LANGUAGE)
            print(f"[{i + 1}/{total}] WER: {sample_wer:.4f}")

        logger.info(
            "Normalization modified text in %d/%d refs and %d/%d preds",
            modified_refs,
            total,
            modified_preds,
            total,
        )

        overall_wer = wer(references, predictions, LANGUAGE)
        logger.info("=" * 60)
        logger.info("Overall WER (%d samples): %.4f", total, overall_wer)
        logger.info("=" * 60)

        self.assertLess(
            overall_wer,
            WER_BASELINE + WER_TOLERANCE,
            f"WER {overall_wer:.4f} exceeds baseline "
            f"{WER_BASELINE:.4f} + tolerance {WER_TOLERANCE:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
