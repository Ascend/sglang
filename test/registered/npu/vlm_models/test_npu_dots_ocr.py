import json
import logging
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DOTS_OCR_WEIGHTS_PATH,
    INVOICE_WITH_BARCODE_LOGO_IMAGES_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-1-npu-a3",
    nightly=True,
)

PROMPT_TEXT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

    1. Bbox format: [x1, y1, x2, y2]

    2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

    3. Text Extraction & Formatting Rules:
        - Picture: For the 'Picture' category, the text field should be omitted.
        - Formula: Format its text as LaTeX.
        - Table: Format its text as HTML.
        - All Others (Text, Title, etc.): Format their text as Markdown.

    4. Constraints:
        - The output text must be the original text from the image, with no translation.
        - All layout elements must be sorted according to human reading order.

    5. Final Output: The entire output must be a single JSON object.
    """

PAYLOAD = {
    "model": "dots_ocr",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"{INVOICE_WITH_BARCODE_LOGO_IMAGES_PATH}"},
                },
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ],
    "max_tokens": 2000,
    "temperature": 0,
}


class TestDotsOcr(CustomTestCase):
    """Testcase: Tests dots.ocr model OCR accuracy on invoice image.

    [Test Category] Model
    [Test Target] rednote-hilab/dots.ocr
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DOTS_OCR_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--max-prefill-tokens",
            "40960",
            "--chunked-prefill-size",
            "40960",
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.7",
            "--mm-attention-backend",
            "ascend_attn",
            "--max-running-requests",
            "80",
            "--disable-radix-cache",
            "--cuda-graph-bs",
            1,
            4,
            8,
            16,
            24,
            32,
            40,
            60,
            80,
            "--enable-multimodal",
            "--sampling-backend",
            "ascend",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_npu_ocr(self):
        """Test OCR model accuracy on invoice image."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=PAYLOAD)
        logging.warning(response.json())
        self.assertEqual(response.status_code, 200)

        ocr_result = json.loads(response.json()["choices"][0]["message"]["content"])

        detected_texts_set = {
            item["text"]
            for item in ocr_result
            if item.get("category") in ["Text", "Section-header"] and "text" in item
        }
        logging.warning(detected_texts_set)

        expected_texts_set = {
            "8:36",
            "Flight",
            "Upto Rs/- 300 discount per pax on round trips, use APPVIA coupon code and Pay through Mobikwik, Get Up to 100% cashback (Maximum Rs. 500) on your booking.",
            "From",
            "DEL",
            "Delhi",
            "To",
            "BLR",
            "Bangalore",
            "Depart",
            "6 FEB",
            "Mon, 2017",
            "Add Return",
            "Adults",
            "12+ Years",
            "1",
            "Children",
            "2 - 11 Years",
            "0",
            "Infants",
            "Below 2 Years",
            "More Options",
            "Direct flights only.",
            "SEARCH FLIGHTS",
        }

        self.assertEqual(
            detected_texts_set,
            expected_texts_set,
            f"Missing texts: {expected_texts_set - detected_texts_set}",
        )


if __name__ == "__main__":
    unittest.main()
