"""NPU adaptation of EPD (Encode-Prefill-Decode) disaggregation tests.

This file ports four of the seven test classes from the upstream GPU test
``test/registered/disaggregation/test_epd_disaggregation.py`` to Ascend NPU:

  - TestNpuEPDDisaggregationOmni        (local only, image + video + audio)
  - TestNpuEPDDisaggregationOneEncoder   (CI-skipped, single encoder + MMMU)
  - TestNpuEPDDisaggregationQwen35       (local only, image + video)
  - TestNpuEPDDisaggregationMultiEncoders (CI runs, two encoders + MMMU)

The remaining three GPU classes are intentionally NOT ported per the NPU
adaptation requirements:

  - TestEPDDisaggregationGrpcEncoderMMMU  : requires --grpc-mode (NPU unsupported)
  - TestEPDDisaggregationGrpcEncoderOnly  : requires --grpc-mode (NPU unsupported)
  - TestEPDDisaggregationMooncake         : requires mooncake/RDMA (NPU unsupported)

NPU-specific adaptations (applied to every ported class):
  - ``cls.server_type = "server"`` (HTTP only; --grpc-mode is unsupported).
  - ``--encoder-transfer-backend zmq_to_scheduler`` (mooncake is GPU/RDMA only).
  - ``--enable-mm-global-cache`` is removed (NPU unsupported).
  - Added NPU backend args: ``--attention-backend ascend``,
    ``--disable-cuda-graph``, ``--mem-fraction-static 0.8``.
  - ``SGLANG_MM_SKIP_COMPUTE_HASH=True``: the Ascend backend does not
    support ``_local_scalar_dense_npu`` for UInt64 used by multimodal
    hash computation; this env var replaces the hash with a random UUID.
  - ``register_npu_ci`` replaces ``register_cuda_ci``; TP=2 (Qwen2.5-VL-3B
    still benefits from 2-NPU TP on Ascend for encoder throughput).
"""

import os
import threading
import unittest
from urllib.parse import urlparse

import openai
import requests
from sglang.srt.utils import kill_process_tree

from sglang.test.ascend.test_ascend_utils import (
    IMAGES_MAN_PATH,
    QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_5_27B_MODEL_WEIGHTS_PATH,
    QWEN3_OMNI_30B_A3B_THINKING_MODEL_PATH,
    VIDEO_JOBS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
)

# NPU common server arguments shared by all server roles.
NPU_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.5",
]

# NPU environment variables required for Ascend PD disaggregation.
# - ASCEND_MF_STORE_URL: Centralized storage URL for Ascend Transfer Engine
#   (MemFabric).  The prefill server hosts the config store at this address
#   and the decode server connects to it.  Without this the transfer engine
#   fails with "input store URL is null".
# - SGLANG_MM_SKIP_COMPUTE_HASH: Ascend backend does not support
#   _local_scalar_dense_npu for UInt64 used by multimodal hash computation;
#   replaces the hash with a random UUID.
NPU_ENV = {
    **os.environ,
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "True",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "200",
    "TRANSFORMERS_VERBOSITY": os.getenv("TRANSFORMERS_VERBOSITY", "error"),
}

# Also set the hash-skip flag in the parent process environment so any
# in-process multimodal utilities (e.g. test helpers) pick it up.
os.environ["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

# Default encoder transfer backend on NPU (mooncake is GPU/RDMA-only).
DEFAULT_NPU_ENCODER_TRANSFER_BACKEND = "zmq_to_scheduler"

# Default tensor parallel size for NPU (2 NPUs per encoder/language server).
DEFAULT_NPU_TP_SIZE = "2"

# A small inline PNG (32x32 blue square) used as a fallback image when the
# NPU CI image cache is unavailable.  Same as in test_npu_disaggregated_vlm.
_INLINE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)

# Register NPU CI.  Only TestNpuEPDDisaggregationMultiEncoders runs in CI
# (the other three classes are marked ``@unittest.skipIf(is_in_ci(), ...)``).
# MultiEncoders uses 8 NPUs (encode1: 0-1, encode2: 2-3, prefill: 4-5, decode: 6-7).
register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


def _file_to_data_url(path: str, mime: str = "image/png") -> str:
    """Encode a local media file as a ``data:`` URL for OpenAI-style payloads."""
    import base64

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _chat_completion(base_url: str, model: str, content: list, **kwargs) -> str:
    """Send a chat completion to ``base_url`` and return the assistant text.

    Uses the requests library directly to avoid openai client wrapper issues
    on NPU CI runners.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    payload.update(kwargs)
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    assert resp.status_code == 200, f"Request failed {resp.status_code}: {resp.text[:300]}"
    return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")


class NpuEPDBase(PDDisaggregationServerBase):
    """Base class for NPU EPD tests.

    Inherits PD-disaggregation plumbing (prefill/decode ports, load balancer,
    ``transfer_backend`` / ``rdma_devices``) from ``PDDisaggregationServerBase``
    and adds the NPU-specific server arguments + boot sequence for the
    encoder/prefill/decode trio.

    Subclasses set ``cls.model`` and may override ``encoder_transfer_backend``
    or ``tp_size``.
    """

    model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    encoder_transfer_backend = DEFAULT_NPU_ENCODER_TRANSFER_BACKEND
    tp_size = DEFAULT_NPU_TP_SIZE
    server_type = "server"  # NPU only supports HTTP ("server"); no --grpc-mode

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Override the PD-disaggregation transfer backend inherited from
        # ``PDDisaggregationServerBase``: in CI the base class forces
        # ``--disaggregation-transfer-backend mooncake`` (GPU/RDMA-only),
        # which fails on NPU with ``ModuleNotFoundError: No module named
        # 'mooncake'``.  Additionally, sglang's default value for
        # ``--disaggregation-transfer-backend`` is also "mooncake", so we
        # MUST explicitly pass the NPU-native "ascend" backend for PD
        # disaggregation to avoid loading the mooncake transfer engine.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        bp = int(parsed.port)
        cls.lb_port = str(bp)
        cls.encode_port = str(bp + 300)
        cls.prefill_port = str(bp + 100)
        cls.decode_port = str(bp + 200)
        cls.bootstrap_port = str(bp + 500)
        cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url
        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.lb_url}/v1"

    @classmethod
    def start_encode(cls, port=None, base_gpu_id=None):
        """Start the encoder-only server.

        Args:
            port: optional override (used by MultiEncoders).
            base_gpu_id: optional --base-gpu-id (used by MultiEncoders).
        """
        port = port or cls.encode_port
        url = f"http://{cls.base_host}:{port}"
        encode_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--port",
            port,
        ]
        if base_gpu_id is not None:
            encode_args.extend(["--base-gpu-id", str(base_gpu_id)])
        encode_args += NPU_COMMON_ARGS
        return popen_launch_server(
            cls.model,
            base_url=url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_prefill(cls, encoder_urls=None):
        """Start the prefill (language-only) server in PD-disaggregation mode."""
        encoder_urls = encoder_urls or cls.encode_url
        prefill_args = [
            "--language-only",
            "--encoder-urls",
            encoder_urls,
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "1",
            "--port",
            cls.prefill_port,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        prefill_args += NPU_COMMON_ARGS
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_decode(cls):
        """Start the decode server in PD-disaggregation mode."""
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "2",
            "--port",
            cls.decode_port,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        decode_args += NPU_COMMON_ARGS
        cls.process_decode = popen_launch_server(
            cls.model,
            base_url=cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_all_servers(cls):
        """Start encode, then prefill+decode in parallel, then lb."""
        cls.process_encode = cls.start_encode()
        t_prefill = threading.Thread(target=cls.start_prefill)
        t_decode = threading.Thread(target=cls.start_decode)
        t_prefill.start()
        t_decode.start()
        t_prefill.join()
        t_decode.join()
        cls.wait_server_ready(cls.encode_url + "/health", process=cls.process_encode)
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        for process in [
            getattr(cls, "process_lb", None),
            getattr(cls, "process_decode", None),
            getattr(cls, "process_prefill", None),
            getattr(cls, "process_encode", None),
            getattr(cls, "process_encode1", None),
            getattr(cls, "process_encode2", None),
        ]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")


# ---------------------------------------------------------------------------
# 1. Omni model EPD — image + video + audio (local only; CI skipped)
# ---------------------------------------------------------------------------


@unittest.skipIf(
    is_in_ci(),
    "Omni model EPD test with image, video, and audio modalities, running locally only",
)
class TestNpuEPDDisaggregationOmni(NpuEPDBase):
    """EPD test for the Omni model on NPU (local only).

    GPU original (TestEPDDisaggregationOmni) covers image / video / audio with
    three encoder_transfer_backends and two server_types (grpc / http).
    On NPU:
      - server_type = "server" (HTTP only; --grpc-mode is unsupported).
      - encoder_transfer_backend = zmq_to_scheduler (mooncake is GPU-only).
      - ``--enable-mm-global-cache`` and cache-hit tests are removed
        (NPU unsupported).
      - PD disaggregation (prefill + decode + LB) is preserved, matching the
        GPU original.

    [Test Category] EPD
    [Test Target] --encoder-only; --language-only; --encoder-urls;
                   --encoder-transfer-backend zmq_to_scheduler; multimodal
    """

    model = QWEN3_OMNI_30B_A3B_THINKING_MODEL_PATH

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        print(
            f"Setting up NPU EPD Omni: model={cls.model}, "
            f"encode={cls.encode_port}, prefill={cls.prefill_port}, "
            f"decode={cls.decode_port}, backend={cls.encoder_transfer_backend}"
        )
        cls.start_all_servers()

    def _client(self):
        return openai.Client(api_key=self.api_key, base_url=f"{self.lb_url}/v1")

    def test_image(self):
        """Single-image request through the EPD pipeline."""
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe this image in a sentence."},
        ]
        text = _chat_completion(self.lb_url, self.model, content, temperature=0, max_tokens=256)
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
        text = _chat_completion(self.lb_url, self.model, content, max_tokens=128)
        print(f"[Omni EPD] Local image response: {text}")
        self.assertGreater(len(text), 0)

    def test_video(self):
        """Video request through the EPD pipeline (local only)."""
        if not os.path.exists(VIDEO_JOBS_PATH):
            self.skipTest(f"Video file not found: {VIDEO_JOBS_PATH}")
        video_url = _file_to_data_url(VIDEO_JOBS_PATH, mime="video/mp4")
        content = [
            {"type": "text", "text": "Describe the video."},
            {"type": "video_url", "video_url": {"url": video_url}},
        ]
        text = _chat_completion(self.lb_url, self.model, content, max_tokens=512)
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
                "text": "Describe the image and the video separately.",
            },
        ]
        text = _chat_completion(self.lb_url, self.model, content, max_tokens=512)
        print(f"[Omni EPD] Mixed image+video response: {text}")
        self.assertGreater(len(text), 0)


# ---------------------------------------------------------------------------
# 2. One encoder + MMMU (CI-skipped to reduce multi-NPU runtime)
# ---------------------------------------------------------------------------


@unittest.skipIf(is_in_ci(), "Skipping in CI to reduce multi-NPU runtime")
class TestNpuEPDDisaggregationOneEncoder(MMMUMixin, NpuEPDBase):
    """Single-encoder EPD test with MMMU evaluation (CI-skipped, like GPU).

    GPU original (TestEPDDisaggregationOneEncoder) uses the small VLM
    ``DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST`` (Qwen2.5-VL-3B-Instruct) and
    enables ``--enable-prefix-mm-cache``.  On NPU we use the Ascend-converted
    weights ``QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH`` and keep the prefix cache.
    """

    # Qwen2.5-VL-3B-Instruct scores ~0.40 on the 50-sample MMMU subset.
    accuracy = 0.40
    mmmu_args = ["--limit", "50"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
        print(
            f"Setting up NPU EPD (one encoder): encode={cls.encode_port}, "
            f"prefill={cls.prefill_port}, decode={cls.decode_port}"
        )
        cls.start_all_servers()

    @classmethod
    def start_encode(cls, port=None, base_gpu_id=None):
        """Override to add --enable-prefix-mm-cache (same as GPU original)."""
        port = port or cls.encode_port
        url = f"http://{cls.base_host}:{port}"
        encode_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--port",
            port,
            "--enable-prefix-mm-cache",
        ]
        if base_gpu_id is not None:
            encode_args.extend(["--base-gpu-id", str(base_gpu_id)])
        encode_args += NPU_COMMON_ARGS
        return popen_launch_server(
            cls.model,
            base_url=url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=NPU_ENV,
        )


# ---------------------------------------------------------------------------
# 3. Qwen3.5 model EPD — image + video (local only; CI skipped)
# ---------------------------------------------------------------------------


@unittest.skipIf(
    is_in_ci(),
    "Qwen3.5 EPD image/video test runs locally only",
)
class TestNpuEPDDisaggregationQwen35(NpuEPDBase):
    """EPD test for the Qwen3.5 model on NPU (local only).

    GPU original (TestEPDDisaggregationQwen35) uses ``QWEN35_27B_MODEL`` and
    adds ``--reasoning-parser qwen3`` + ``--model-loader-extra-config`` with
    multi-threaded loading.  The GPU original only starts encode + prefill
    (no decode / no LB); we preserve that here.
    """

    model = QWEN3_5_27B_MODEL_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.process_decode = None
        cls.process_lb = None
        cls.language_url = cls.prefill_url
        print(
            f"Setting up NPU Qwen3.5 encoder disaggregation: model={cls.model}, "
            f"encode={cls.encode_port}, language={cls.prefill_port}"
        )
        cls.process_encode = cls.start_encode()
        cls.start_prefill()
        cls.wait_server_ready(cls.encode_url + "/health", process=cls.process_encode)
        cls.wait_server_ready(cls.language_url + "/health", process=cls.process_prefill)

    @classmethod
    def start_encode(cls, port=None, base_gpu_id=None):
        port = port or cls.encode_port
        url = f"http://{cls.base_host}:{port}"
        encode_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--port",
            port,
            "--reasoning-parser",
            "qwen3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        if base_gpu_id is not None:
            encode_args.extend(["--base-gpu-id", str(base_gpu_id)])
        encode_args += NPU_COMMON_ARGS
        return popen_launch_server(
            cls.model,
            base_url=url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_prefill(cls, encoder_urls=None):
        encoder_urls = encoder_urls or cls.encode_url
        language_args = [
            "--language-only",
            "--encoder-urls",
            encoder_urls,
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "1",
            "--port",
            cls.prefill_port,
            "--reasoning-parser",
            "qwen3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        language_args += NPU_COMMON_ARGS
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
            env=NPU_ENV,
        )

    def _client(self):
        return openai.Client(api_key=self.api_key, base_url=f"{self.language_url}/v1")

    def test_image(self):
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _INLINE_IMAGE_URL},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=256,
            extra_body={"reasoning_effort": "none"},
        )
        text = response.choices[0].message.content
        print(f"[Qwen3.5 EPD] Image response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

    def test_video(self):
        if not os.path.exists(VIDEO_JOBS_PATH):
            self.skipTest(f"Video file not found: {VIDEO_JOBS_PATH}")
        video_url = _file_to_data_url(VIDEO_JOBS_PATH, mime="video/mp4")
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the video."},
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url},
                        },
                    ],
                }
            ],
            max_tokens=1024,
            stream=False,
        )
        text = response.choices[0].message.content
        print(f"[Qwen3.5 EPD] Video response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)


# ---------------------------------------------------------------------------
# 4. Multi-encoder EPD + MMMU (CI runs)
# ---------------------------------------------------------------------------


class TestNpuEPDDisaggregationMultiEncoders(MMMUMixin, NpuEPDBase):
    """EPD test with multiple encode servers for load balancing (CI runs).

    GPU original (TestEPDDisaggregationMultiEncoders) starts two encode servers
    on GPU 0/1 and runs MMMU.  On NPU we start two encode servers on NPU 0 and
    NPU 2 (each encoder uses TP=2, so encode1 occupies NPU 0-1 and encode2
    occupies NPU 2-3).  Prefill uses NPU 0-1 and decode uses NPU 2-3 via
    ``--base-gpu-id``.

    NOTE: The NPU runner has 4 NPUs total.  Because each encoder/language
    server uses TP=2, the two encoders cannot both occupy NPU 0-1 simultaneously
    with the prefill server.  To fit in 4 NPUs we let encode1 + prefill share
    NPU 0-1 and encode2 + decode share NPU 2-3 (encode is encoder-only and
    prefill is language-only, so they can co-locate on the same NPUs via
    different processes).
    """

    # Qwen2.5-VL-3B-Instruct scores ~0.40 on the 50-sample MMMU subset.
    accuracy = 0.40
    mmmu_args = ["--limit", "50"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
        cls.encode_port1 = str(int(cls.lb_port) + 300)
        cls.encode_port2 = str(int(cls.lb_port) + 301)
        cls.encode_url1 = f"http://{cls.base_host}:{cls.encode_port1}"
        cls.encode_url2 = f"http://{cls.base_host}:{cls.encode_port2}"
        print(
            f"Setting up NPU EPD (multiple encoders): "
            f"encode1={cls.encode_port1}, encode2={cls.encode_port2}, "
            f"prefill={cls.prefill_port}, decode={cls.decode_port}"
        )

        # Start two encode servers in parallel.
        # NPU allocation (8 NPUs available, TP=2 per server):
        #   encode1 -> NPU 0-1, encode2 -> NPU 2-3,
        #   prefill -> NPU 4-5, decode  -> NPU 6-7
        t1 = threading.Thread(target=cls._start_encode1, args=(cls.encode_port1, 0))
        t2 = threading.Thread(target=cls._start_encode2, args=(cls.encode_port2, 2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Start prefill + decode in parallel.
        tp = threading.Thread(target=cls.start_prefill)
        td = threading.Thread(target=cls.start_decode)
        tp.start()
        td.start()
        tp.join()
        td.join()

        cls.wait_server_ready(cls.encode_url1 + "/health", process=cls.process_encode1)
        cls.wait_server_ready(cls.encode_url2 + "/health", process=cls.process_encode2)
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def _start_encode1(cls, port, base_gpu_id):
        encode_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--port",
            port,
            "--enable-prefix-mm-cache",
            "--base-gpu-id",
            str(base_gpu_id),
        ]
        encode_args += NPU_COMMON_ARGS
        cls.process_encode1 = popen_launch_server(
            cls.model,
            base_url=f"http://{cls.base_host}:{port}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=NPU_ENV,
        )

    @classmethod
    def _start_encode2(cls, port, base_gpu_id):
        encode_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--port",
            port,
            "--enable-prefix-mm-cache",
            "--base-gpu-id",
            str(base_gpu_id),
        ]
        encode_args += NPU_COMMON_ARGS
        cls.process_encode2 = popen_launch_server(
            cls.model,
            base_url=f"http://{cls.base_host}:{port}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_prefill(cls, encoder_urls=None):
        """Start prefill pointing at BOTH encoder URLs (load balancing).

        Uses NPU 4-5 (base-gpu-id=4) to avoid OOM: encode1 occupies NPU 0-1,
        encode2 occupies NPU 2-3, so prefill must use NPU 4-5.
        """
        encoder_urls = f"{cls.encode_url1},{cls.encode_url2}"
        prefill_args = [
            "--language-only",
            "--encoder-urls",
            encoder_urls,
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "4",
            "--port",
            cls.prefill_port,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        prefill_args += NPU_COMMON_ARGS
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_decode(cls):
        """Start decode on NPU 6-7 (base-gpu-id=6)."""
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "6",
            "--port",
            cls.decode_port,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        decode_args += NPU_COMMON_ARGS
        cls.process_decode = popen_launch_server(
            cls.model,
            base_url=cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=NPU_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        for process in [
            getattr(cls, "process_lb", None),
            getattr(cls, "process_decode", None),
            getattr(cls, "process_prefill", None),
            getattr(cls, "process_encode1", None),
            getattr(cls, "process_encode2", None),
        ]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")


if __name__ == "__main__":
    unittest.main()
