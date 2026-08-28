import unittest

import openai

from sglang.test.ascend.test_ascend_utils import (
    AUDIOS_BIRD_PATH,
    AUDIOS_TRUMP_PATH,
    IMAGES_LOGO_PATH,
    IMAGES_MAN_PATH,
    QWEN3_OMNI_30B_A3B_INSTRUCT_WEIGHTS_PATH,
    VIDEO_JOBS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.vlm_utils import (
    AudioOpenAITestMixin,
    ImageOpenAITestMixin,
    OmniOpenAITestMixin,
    TestOpenAIMLLMServerBase,
    VideoOpenAITestMixin,
)

register_npu_ci(est_time=400, suite="full-4-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="full-test-npu", nightly=True)


class TestQwen3Omni30bA3bInstruct(OmniOpenAITestMixin):
    model = QWEN3_OMNI_30B_A3B_INSTRUCT_WEIGHTS_PATH
    extra_args = [
        "--tp-size",
        4,
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        0.5,
        "--disable-cuda-graph",
        "--disable-fast-image-processor",
        "--grammar-backend=none",
        "--mm-process-config",
        '{"image":{"max_pixels":262144},"video":{"fps":1,"max_pixels":262144,"max_frames":4}}',
    ]

    # ---- ImageOpenAITestMixin overrides ----

    def run_decode_with_image(self, image_id):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        content = []
        if image_id == 0:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGES_MAN_PATH},
                }
            )
        elif image_id == 1:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGES_LOGO_PATH},
                }
            )
        else:
            pass

        content.append(
            {
                "type": "text",
                "text": "Describe this image in a sentence.",
            }
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": content},
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGES_MAN_PATH},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        print("-" * 30)
        print(f"Single image response:\n{response.choices[0].message.content}")
        print("-" * 30)

        self.verify_single_image_response(response)

    def test_multi_turn_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGES_MAN_PATH},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "There is a man at the back of a yellow cab ironing his clothes.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Repeat your previous answer."}
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert (
            "man" in text or "cab" in text
        ), f"text: {text}, should contain man or cab"
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_multi_images_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGES_MAN_PATH},
                            "modalities": "multi-images",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGES_LOGO_PATH},
                            "modalities": "multi-images",
                        },
                        {
                            "type": "text",
                            "text": "I have two very different images. Please describe them.",
                        },
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        print("-" * 30)
        print(f"Multi images response:\n{text}")
        print("-" * 30)
        assert (
            "man" in text
            or "cab" in text
            or "SUV" in text
            or "taxi" in text
            or "car" in text
        ), f"text: {text}, should contain man, cab, SUV, taxi or car"
        assert (
            "logo" in text or '"S"' in text or "SG" in text or "graphic" in text
        ), f"text: {text}, should contain logo, S or SG or graphic"
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_video_images_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        messages = self.prepare_video_images_messages(VIDEO_JOBS_PATH)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            stream=False,
        )

        video_response = response.choices[0].message.content

        print("-" * 30)
        print(f"Video images response:\n{video_response}")
        print("-" * 30)

        assert (
            "iPod" in video_response
            or "device" in video_response
            or "microphone" in video_response
        ), f"""
        ====================== video_images response =====================
        {video_response}
        ===========================================================
        should contain 'iPod' or 'device' or 'microphone'
        """
        assert (
            "man" in video_response
            or "person" in video_response
            or "individual" in video_response
            or "speaker" in video_response
            or "presenter" in video_response
            or "Steve" in video_response
            or "hand" in video_response
        ), f"""
        ====================== video_images response =====================
        {video_response}
        ===========================================================
        should contain 'man' or 'person' or 'individual' or 'speaker' or 'presenter' or 'Steve' or 'hand'
        """
        assert (
            "present" in video_response
            or "examine" in video_response
            or "display" in video_response
            or "hold" in video_response
        ), f"""
        ====================== video_images response =====================
        {video_response}
        ===========================================================
        should contain 'present' or 'examine' or 'display' or 'hold'
        """
        self.assertIsNotNone(video_response)
        self.assertGreater(len(video_response), 0)

    # ---- VideoOpenAITestMixin overrides ----

    def test_video_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        messages = self.prepare_video_messages(VIDEO_JOBS_PATH)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            stream=False,
            **(self.get_vision_request_kwargs()),
        )

        video_response = response.choices[0].message.content.lower()

        print("-" * 30)
        print(f"Video response:\n{video_response}")
        print("-" * 30)

        assert (
            "ipod" in video_response
            or "device" in video_response
            or "microphone" in video_response
            or "phone" in video_response
        ), f"video_response: {video_response}, should contain 'iPod' or 'device'"
        assert (
            "man" in video_response
            or "person" in video_response
            or "individual" in video_response
            or "speaker" in video_response
            or "presenter" in video_response
            or "hand" in video_response
        ), f"video_response: {video_response}, should either have 'man' in video_response, or 'person' in video_response, or 'individual' in video_response or 'speaker' in video_response or 'presenter' or 'hand' in video_response"
        assert (
            "present" in video_response
            or "examine" in video_response
            or "display" in video_response
            or "hold" in video_response
        ), f"video_response: {video_response}, should contain 'present', 'examine', 'display', or 'hold'"
        assert (
            "black" in video_response or "dark" in video_response
        ), f"video_response: {video_response}, should contain 'black' or 'dark'"
        self.assertIsNotNone(video_response)
        self.assertGreater(len(video_response), 0)

    # ---- AudioOpenAITestMixin overrides ----

    def test_audio_speech_completion(self):
        client = openai.Client(api_key="sk-123456", base_url=self.base_url)

        messages = self.prepare_audio_messages(
            "Listen to this audio and write down the audio transcription in English.",
            AUDIOS_TRUMP_PATH,
        )

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
            **(self.get_audio_request_kwargs()),
        )

        audio_response = response.choices[0].message.content

        print("-" * 30)
        print(f"audio speech response:\n{audio_response}")
        print("-" * 30)

        audio_response = audio_response.lower()

        self.assertIsNotNone(audio_response)
        self.assertGreater(len(audio_response), 0)

        self.verify_speech_recognition_response(audio_response)

    def test_audio_ambient_completion(self):
        client = openai.Client(api_key="sk-123456", base_url=self.base_url)

        messages = self.prepare_audio_messages(
            "Please listen to the audio snippet carefully and transcribe the content in English.",
            AUDIOS_BIRD_PATH,
        )

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
            **(self.get_audio_request_kwargs()),
        )

        audio_response = response.choices[0].message.content

        print("-" * 30)
        print(f"audio ambient response:\n{audio_response}")
        print("-" * 30)

        audio_response = audio_response.lower()

        self.assertIsNotNone(audio_response)
        self.assertGreater(len(audio_response), 0)

        assert "bird" in audio_response

    # ---- OmniOpenAITestMixin overrides ----

    def test_mixed_modality_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGES_MAN_PATH},
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": AUDIOS_TRUMP_PATH},
                    },
                    {
                        "type": "text",
                        "text": "I have an image and audio, which are not related at all. Please:  1. Describe the image in a sentence, 2. Repeat the exact words from the audio I provided. Be exact",
                    },
                ],
            },
        ]
        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
        )

        text = response.choices[0].message.content

        print("-" * 30)
        print(f"Mixed modality response:\n{text}")
        print("-" * 30)

        self.verify_single_image_response(response=response)
        self.verify_speech_recognition_response(text=text)

    def test_image_prefix_cache_reuse(self):
        """Override the mixin test to use local images (NPU CI has no access
        to raw.githubusercontent.com). Mirrors vlm_utils.py behavior."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        def describe(image_path: str) -> str:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_path}},
                            {
                                "type": "text",
                                "text": "Describe this image in one sentence.",
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=32,
                **(self.get_vision_request_kwargs()),
            )
            assert response.usage.prompt_tokens > 0
            content = response.choices[0].message.content
            assert isinstance(content, str) and content
            return content

        # miss -> compute, then hit -> reuse the identical image's prefix
        first = describe(IMAGES_MAN_PATH)
        repeat = describe(IMAGES_MAN_PATH)
        # a different image must be computed on its own, not reuse `first`'s KV
        other = describe(IMAGES_LOGO_PATH)
        # the original image again, after a different one occupied the cache
        first_again = describe(IMAGES_MAN_PATH)

        self.assertEqual(
            first,
            repeat,
            "Repeating an identical image changed the output; image prefix "
            "reuse broke greedy determinism.",
        )
        self.assertEqual(
            first,
            first_again,
            "The identical image after a different one changed the output; "
            "image KV was cross-contaminated across requests.",
        )
        self.assertNotEqual(
            first,
            other,
            "A different image produced an identical description; a wrong "
            "image's KV may have been reused from the prefix cache.",
        )


# Delete the mixin classes so that they are not collected by pytest
del (
    TestOpenAIMLLMServerBase,
    ImageOpenAITestMixin,
    VideoOpenAITestMixin,
    AudioOpenAITestMixin,
    OmniOpenAITestMixin,
)


if __name__ == "__main__":
    unittest.main()
