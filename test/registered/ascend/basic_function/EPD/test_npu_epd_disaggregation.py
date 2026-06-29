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
    popen_with_error_check,
)

NPU_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.5",
]

NPU_ENV = {
    **os.environ,
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "True",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "200",
    "TRANSFORMERS_VERBOSITY": os.getenv("TRANSFORMERS_VERBOSITY", "error"),
}

os.environ["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

DEFAULT_NPU_ENCODER_TRANSFER_BACKEND = "zmq_to_scheduler"

DEFAULT_NPU_TP_SIZE = "2"

_INLINE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4b"
    "AAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGB"
    "cua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR"
    "3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="
)

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


def _file_to_data_url(path: str, mime: str = "image/png") -> str:
    import base64

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _chat_completion(base_url: str, model: str, content: list, **kwargs) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    payload.update(kwargs)
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    assert (
        resp.status_code == 200
    ), f"Request failed {resp.status_code}: {resp.text[:300]}"
    return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")


class NpuEPDBase(PDDisaggregationServerBase):
    model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    encoder_transfer_backend = DEFAULT_NPU_ENCODER_TRANSFER_BACKEND
    tp_size = DEFAULT_NPU_TP_SIZE
    server_type = "server"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
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
    def launch_lb(cls):
        import shlex

        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            cls.prefill_url,
            cls.bootstrap_port,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)

    @classmethod
    def start_all_servers(cls):
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


@unittest.skipIf(
    is_in_ci(),
    "Omni model EPD test with image, video, and audio modalities, running locally only",
)
class TestNpuEPDDisaggregationOmni(NpuEPDBase):
    """
    EPD disaggregation test for omni models on NPU. Covers image and video
    modalities with encoder_transfer_backend: zmq_to_scheduler.
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
        content = [
            {"type": "image_url", "image_url": {"url": _INLINE_IMAGE_URL}},
            {"type": "text", "text": "Describe this image in a sentence."},
        ]
        text = _chat_completion(
            self.lb_url, self.model, content, temperature=0, max_tokens=256
        )
        print(f"[Omni EPD] Image response: {text}")
        self.assertGreater(len(text), 0)

    def test_image_local_file(self):
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


@unittest.skipIf(is_in_ci(), "Skipping in CI to reduce multi-NPU runtime")
class TestNpuEPDDisaggregationOneEncoder(MMMUMixin, NpuEPDBase):
    """
    Single-encoder EPD test with MMMU evaluation.
    Uses --enable-prefix-mm-cache, same as GPU original.
    """

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


@unittest.skipIf(
    is_in_ci(),
    "Qwen3.5 EPD image/video test runs locally only",
)
class TestNpuEPDDisaggregationQwen35(NpuEPDBase):
    """
    EPD test for Qwen3.5 model on NPU (local only).
    Uses --reasoning-parser qwen3 with multi-threaded loading.
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


class TestNpuEPDDisaggregationMultiEncoders(MMMUMixin, NpuEPDBase):
    """
    EPD test with multiple encode servers for load balancing.
    Uses 8 NPUs: 2 encoders (TP=2 each) + prefill (TP=2) + decode (TP=2).
    """

    accuracy = 0.40
    mmmu_args = ["--limit", "50", "--batch_size", "4"]

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

        t1 = threading.Thread(target=cls._start_encode1, args=(cls.encode_port1, 0))
        t2 = threading.Thread(target=cls._start_encode2, args=(cls.encode_port2, 2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

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
        prefill_args = [
            "--language-only",
            "--encoder-urls",
            cls.encode_url1,
            cls.encode_url2,
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
