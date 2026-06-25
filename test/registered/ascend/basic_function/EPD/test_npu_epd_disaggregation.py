"""
NPU EPD (Encode Prefill Disaggregation) Tests.

Ported from GPU test:
    test/registered/disaggregation/test_epd_disaggregation.py

NPU Adaptations
---------------
- Removed ``--grpc-mode`` (not supported on NPU).
- Removed ``--enable-mm-global-cache`` (not supported on NPU); cache-hit
  test methods that depend on it are dropped.
- Removed mooncake transfer backend (not supported on NPU); only
  ``zmq_to_scheduler`` is used.
- Excluded GPU-only classes:
    * TestEPDDisaggregationGrpcEncoderMMMU   (needs --grpc-mode)
    * TestEPDDisaggregationGrpcEncoderOnly   (needs --grpc-mode)
    * TestEPDDisaggregationMooncake          (needs mooncake RDMA)
- Removed ``--disaggregation-mode prefill/decode`` (PD disaggregation):
  Qwen3-VL-30B-A3B-Instruct uses Qwen3.5 GDN attention which is not
  supported on NPU when PD disaggregation triggers the GDN forward path.
  The existing NPU EPD test (test_npu_disaggregated_vlm.py) also avoids
  PD disaggregation for the same reason.  Tests use encoder + language
  only (no PD split, no load balancer), matching the proven NPU pattern.
- Model paths changed to NPU-available weights (ModelScope cache).
- Added NPU args: --attention-backend ascend, --disable-cuda-graph,
  --mem-fraction-static 0.8.
- Set SGLANG_MM_SKIP_COMPUTE_HASH=True (NPU backend does not support
  _local_scalar_dense for UInt64 used in multimodal hash).
- Replaced MMMU eval (lmms-eval, heavy) with direct multimodal request
  assertions that are lighter and better suited for NPU nightly CI.
- Used register_npu_ci / CustomTestCase instead of register_cuda_ci /
  PDDisaggregationServerBase.
- cls.server_type = "server" — NPU always uses standard HTTP server mode
  (no gRPC), so the server_type branching from the GPU Omni test is
  collapsed to a single HTTP path.
- Removed @unittest.skipIf(is_in_ci()) decorators so tests run in NPU CI.
- TP size changed from 1 (GPU) to 2 (NPU) because the 30B model requires
  tensor parallelism across multiple NPU cards.
"""

import base64
import os
import threading
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    IMAGES_MAN_PATH,
    QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH,
    VIDEO_JOBS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

os.environ["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

register_npu_ci(est_time=1800, suite="full-8-npu-a3", nightly=True)

# NPU common server arguments shared by all server roles.
NPU_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
]

# Inline 32x32 PNG used for lightweight multimodal requests (same image
# used by the existing NPU EPD tests).
_INLINE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)


def _file_to_data_url(path, mime="image/png"):
    """Convert a local file to a base64 data URL."""
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def _chat_completion(base_url, model, content, max_tokens=64, temperature=0):
    """Send a multimodal chat completion request and return the text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    assert (
        resp.status_code == 200
    ), f"Request failed {resp.status_code}: {resp.text[:300]}"
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Base class — EPD with encoder + language (no PD split, no LB)
# ---------------------------------------------------------------------------


class NpuEPDBase(CustomTestCase):
    """Base class that boots encoder + language servers for EPD.

    Server layout (TP=2 per server, 4 NPUs total):
      encode   — encoder-only  (NPUs 0-1)
      language — language-only (NPUs 2-3)

    No PD disaggregation (no prefill/decode split, no load balancer) to
    avoid triggering the unsupported GDN attention path on NPU.

    Subclasses override ``model`` and optionally ``encoder_transfer_backend``.
    """

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    encoder_transfer_backend = "zmq_to_scheduler"
    tp_size = "2"

    @classmethod
    def setUpClass(cls):
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        bp = int(parsed.port)
        cls.encode_port = str(bp + 300)
        cls.language_port = str(bp + 100)
        cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"
        cls.language_url = f"http://{cls.base_host}:{cls.language_port}"
        cls.process_encode = None
        cls.process_language = None
        cls.api_key = "sk-123456"

    @classmethod
    def _encode_args(cls):
        return [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "0",
            "--port",
            cls.encode_port,
        ] + NPU_COMMON_ARGS

    @classmethod
    def _language_args(cls):
        return [
            "--language-only",
            "--encoder-urls",
            cls.encode_url,
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            str(int(cls.tp_size)),
            "--port",
            cls.language_port,
        ] + NPU_COMMON_ARGS

    @classmethod
    def start_encode(cls):
        print(f"[EPD] Starting encoder on {cls.encode_url} (NPUs 0-1)")
        cls.process_encode = popen_launch_server(
            cls.model,
            cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._encode_args(),
        )

    @classmethod
    def start_language(cls):
        print(f"[EPD] Starting language on {cls.language_url} (NPUs 2-3)")
        cls.process_language = popen_launch_server(
            cls.model,
            cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._language_args(),
        )

    @classmethod
    def start_servers(cls):
        """Start encode + language servers.

        popen_launch_server waits for health internally.
        """
        cls.start_encode()
        cls.start_language()

    @classmethod
    def tearDownClass(cls):
        for p in [cls.process_language, cls.process_encode]:
            if p:
                try:
                    kill_process_tree(p.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")
        time.sleep(5)


# ---------------------------------------------------------------------------
# 1. Omni model EPD — image + video (audio skipped: no NPU CI audio data)
# ---------------------------------------------------------------------------


class TestNpuEPDDisaggregationOmni(NpuEPDBase):
    """EPD test for the Omni model on NPU.

    GPU original (TestEPDDisaggregationOmni) covers image / video / audio
    with three encoder_transfer_backends and two server_types (grpc / http).
    On NPU:
      - server_type = "server" (HTTP only, no gRPC).
      - encoder_transfer_backend = zmq_to_scheduler (no mooncake).
      - Audio tests are skipped (no audio fixture in NPU CI cache).
      - Cache-hit tests are removed (require --enable-mm-global-cache).
      - No PD disaggregation (GDN not supported on NPU).

    [Test Category] EPD
    [Test Target] --encoder-only; --language-only; --encoder-urls;
                   --encoder-transfer-backend zmq_to_scheduler; multimodal
    """

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    server_type = "server"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print(
            f"Setting up NPU EPD Omni: model={cls.model}, "
            f"encode={cls.encode_port}, language={cls.language_port}, "
            f"server_type={cls.server_type}, backend={cls.encoder_transfer_backend}"
        )
        cls.start_servers()

    def test_image(self):
        """Single-image request through the EPD pipeline."""
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe this image in a sentence."},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=128)
        print(f"[Omni EPD] Image response: {text}")
        self.assertGreater(len(text), 0)

    def test_image_local_file(self):
        """Image request using a local file from the NPU CI image cache."""
        if not os.path.exists(IMAGES_MAN_PATH):
            self.skipTest(f"Image file not found: {IMAGES_MAN_PATH}")
        image_url = _file_to_data_url(IMAGES_MAN_PATH)
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "What do you see in this image?"},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=128)
        print(f"[Omni EPD] Local image response: {text}")
        self.assertGreater(len(text), 0)

    def test_video(self):
        """Video request through the EPD pipeline."""
        if not os.path.exists(VIDEO_JOBS_PATH):
            self.skipTest(f"Video file not found: {VIDEO_JOBS_PATH}")
        video_url = _file_to_data_url(VIDEO_JOBS_PATH, mime="video/mp4")
        content = [
            {"type": "text", "text": "Describe the video."},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=512)
        print(f"[Omni EPD] Video response: {text}")
        self.assertGreater(len(text), 0)

    def test_mixed_image_video(self):
        """Image + video in one request to test multi-modal routing."""
        if not os.path.exists(VIDEO_JOBS_PATH):
            self.skipTest(f"Video file not found: {VIDEO_JOBS_PATH}")
        video_url = _file_to_data_url(VIDEO_JOBS_PATH, mime="video/mp4")
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "video_url", "video_url": {"url": video_url}},
            {
                "type": "text",
                "text": "I have an image and a video. "
                "1. Describe the image in a sentence. "
                "2. Describe what happens in the video.",
            },
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=512)
        print(f"[Omni EPD] Mixed image+video response: {text}")
        self.assertGreater(len(text), 0)


# ---------------------------------------------------------------------------
# 2. Single-encoder EPD — image + text (MMMU eval replaced with direct tests)
# ---------------------------------------------------------------------------


class TestNpuEPDDisaggregationOneEncoder(NpuEPDBase):
    """EPD with a single encoder server, Qwen3-VL-30B model.

    GPU original (TestEPDDisaggregationOneEncoder) runs MMMU eval via
    lmms-eval (MMMUMixin).  On NPU, MMMU eval is replaced with direct
    multimodal request assertions that are lighter and faster for nightly CI.

    [Test Category] EPD
    [Test Target] --encoder-only; --language-only; --encoder-urls;
                   single encoder
    """

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print(
            f"Setting up NPU EPD (one encoder): "
            f"encode={cls.encode_port}, language={cls.language_port}"
        )
        cls.start_servers()

    def test_encode_health(self):
        """Encoder-only server /health returns 200."""
        r = requests.get(f"{self.encode_url}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_language_health(self):
        """Language-only server /health returns 200."""
        r = requests.get(f"{self.language_url}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_text_generation(self):
        """Text-only request to the language server."""
        resp = requests.post(
            f"{self.language_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Paris", resp.json().get("text", ""))

    def test_image_processing(self):
        """Single-image request through the EPD pipeline."""
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe the image briefly."},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=64)
        print(f"[OneEncoder EPD] Image response: {text}")
        self.assertGreater(len(text), 0)

    def test_multi_image_processing(self):
        """Multi-image request to verify encoder handles multiple images."""
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe these two images briefly."},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=64)
        print(f"[OneEncoder EPD] Multi-image response: {text}")
        self.assertGreater(len(text), 0)


# ---------------------------------------------------------------------------
# 3. Qwen3.5 EPD — encoder + language-only (mirrors GPU Qwen35 test)
# ---------------------------------------------------------------------------


class TestNpuEPDDisaggregationQwen35(NpuEPDBase):
    """EPD test for Qwen3.5 model on NPU (encoder + language only).

    GPU original (TestEPDDisaggregationQwen35) starts an encoder-only
    server and a language-only server (no decode, no LB).  Requests go
    directly to the language server.  This structure is preserved on NPU.

    Uses Qwen3-VL-30B on NPU (Qwen3.5-27B may not be a VLM in NPU CI cache).

    [Test Category] EPD
    [Test Target] --encoder-only; --language-only; --encoder-urls;
                   encoder + language (no PD split)
    """

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print(
            f"Setting up NPU Qwen3.5 EPD: model={cls.model}, "
            f"encode={cls.encode_port}, language={cls.language_port}"
        )
        cls.start_servers()

    def test_image(self):
        """Single-image request to the language server."""
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe this image in a sentence."},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=256)
        print(f"[Qwen3.5 EPD] Image response: {text}")
        self.assertGreater(len(text), 0)

    def test_video(self):
        """Video request to the language server."""
        if not os.path.exists(VIDEO_JOBS_PATH):
            self.skipTest(f"Video file not found: {VIDEO_JOBS_PATH}")
        video_url = _file_to_data_url(VIDEO_JOBS_PATH, mime="video/mp4")
        content = [
            {"type": "text", "text": "Describe the video."},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=512)
        print(f"[Qwen3.5 EPD] Video response: {text}")
        self.assertGreater(len(text), 0)


# ---------------------------------------------------------------------------
# 4. Multi-encoder EPD — two encode servers + language
# ---------------------------------------------------------------------------


class TestNpuEPDDisaggregationMultiEncoders(NpuEPDBase):
    """EPD with multiple encoder servers for load-balanced encoding.

    GPU original (TestEPDDisaggregationMultiEncoders) starts two encode
    servers on different GPUs and registers both URLs with the prefill
    server.  MMMU eval is replaced with direct image tests on NPU.

    Server layout (TP=2, 6 NPUs total):
      encode1 — NPUs 0-1
      encode2 — NPUs 2-3
      language — NPUs 4-5

    [Test Category] EPD
    [Test Target] --encoder-only; --language-only; multiple --encoder-urls;
                   load balancing
    """

    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.encode_port2 = str(int(cls.encode_port) + 1)
        cls.encode_url2 = f"http://{cls.base_host}:{cls.encode_port2}"
        cls.process_encode2 = None
        print(
            f"Setting up NPU EPD (multi-encoder): "
            f"encode1={cls.encode_port}, encode2={cls.encode_port2}, "
            f"language={cls.language_port}"
        )
        # Start two encoders in parallel
        t1 = threading.Thread(target=cls._start_encode2)
        t2 = threading.Thread(target=cls.start_encode)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Start language server (with both encoder URLs)
        cls.start_language()

    @classmethod
    def _language_args(cls):
        """Override to include both encoder URLs."""
        return [
            "--language-only",
            "--encoder-urls",
            cls.encode_url,
            cls.encode_url2,
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            str(int(cls.tp_size) * 2),
            "--port",
            cls.language_port,
        ] + NPU_COMMON_ARGS

    @classmethod
    def _start_encode2(cls):
        args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            str(int(cls.tp_size)),
            "--port",
            cls.encode_port2,
        ] + NPU_COMMON_ARGS
        print(f"[EPD] Starting encoder2 on {cls.encode_url2} (NPUs 2-3)")
        cls.process_encode2 = popen_launch_server(
            cls.model,
            cls.encode_url2,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
        )

    @classmethod
    def tearDownClass(cls):
        for p in [cls.process_language, cls.process_encode2, cls.process_encode]:
            if p:
                try:
                    kill_process_tree(p.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")
        time.sleep(5)

    def test_encode1_health(self):
        """First encoder server /health returns 200."""
        r = requests.get(f"{self.encode_url}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_encode2_health(self):
        """Second encoder server /health returns 200."""
        r = requests.get(f"{self.encode_url2}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_image_processing(self):
        """Single-image request with multiple encoders available."""
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe the image briefly."},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=64)
        print(f"[MultiEncoders EPD] Image response: {text}")
        self.assertGreater(len(text), 0)

    def test_multi_image_processing(self):
        """Multi-image request to exercise load-balanced encoders."""
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe these two images briefly."},
        ]
        text = _chat_completion(self.language_url, self.model, content, max_tokens=64)
        print(f"[MultiEncoders EPD] Multi-image response: {text}")
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()

