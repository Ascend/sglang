"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Qwen3-ASR
(served under the model alias "whisper").

Usage:
    python3 test_npu_serving_transcription.py -v
"""

import io
import json
import unittest
from typing import List, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    AUDIO_TRUMP_WEF_PATH,
    QWEN3_ASR_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1600, suite="full-1-npu-a3", nightly=True)
register_npu_ci(est_time=1600, suite="validate-batch-npu", nightly=True)

register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)


def read_audio_bytes(url=AUDIO_TRUMP_WEF_PATH):
    """Read audio file from local path and return raw bytes."""
    with open(url, "rb") as f:
        return f.read()


class TestServingTranscription(CustomTestCase):
    """Test Qwen3-ASR transcription via /v1/audio/transcriptions endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_ASR_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                "whisper",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _transcribe(
        self,
        language: Optional[str] = "en",
        response_format: Optional[str] = None,
    ):
        """Send a non-streaming transcription request and return the JSON response.

        Passing ``language=None`` omits the field entirely.
        """
        audio_bytes = read_audio_bytes()
        data = {"model": "whisper"}
        if language is not None:
            data["language"] = language
        if response_format is not None:
            data["response_format"] = response_format
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data=data,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _transcribe_stream(self, language: Optional[str] = None) -> List[str]:
        """Send a streaming transcription request and return the delta strings."""
        audio_bytes = read_audio_bytes()
        data = {"model": "whisper", "stream": "true"}
        if language is not None:
            data["language"] = language
        with requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data=data,
            stream=True,
            timeout=120,
        ) as response:
            self.assertEqual(response.status_code, 200, response.text)
            deltas: List[str] = []
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        deltas.append(content)
            return deltas

    def test_basic_transcription(self):
        """Test that transcription returns a valid non-empty response."""
        result = self._transcribe()
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0, "Transcription should not be empty")

    def test_transcription_content_quality(self):
        """Test that transcription captures key content from the audio."""
        result = self._transcribe()
        text = result["text"].lower()
        keywords = ["privilege", "leader", "science", "art"]
        matches = [kw for kw in keywords if kw in text]
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 of {keywords} in transcription, "
            f"found {matches}. Full text: {text}",
        )

    # -- language omitted (language=None) ----------------------------------
    # Qwen3-ASR does not run fused language detection; omitting the field
    # must still produce a valid transcription without leaking ASR special
    # tokens.

    def test_language_omitted_verbose_json(self):
        """language omitted + verbose_json returns clean text."""
        result = self._transcribe(language=None, response_format="verbose_json")
        text = result.get("text", "")
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        self.assertNotIn("<|asr|>", text, f"Special token leaked into text: {text!r}")
        # Sanity-check content against the same keywords the English test uses.
        keywords = ["privilege", "leader", "science", "art"]
        matches = [kw for kw in keywords if kw in text.lower()]
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 of {keywords} in transcription, "
            f"found {matches}. Full text: {text!r}",
        )

    def test_language_omitted_streaming(self):
        """language=None + stream=True: deltas are produced and scrubbed of
        ASR special tokens.
        """
        deltas = self._transcribe_stream(language=None)
        self.assertTrue(len(deltas) > 0, "Expected at least one streamed delta")
        for d in deltas:
            self.assertNotIn(
                "<|asr|>", d, f"Special token leaked into streaming delta: {d!r}"
            )


if __name__ == "__main__":
    unittest.main()
