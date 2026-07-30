"""Shared garbled text detection utilities.

The model's garbled output is NOT encoding corruption. All characters are valid
UTF-8. The model enters a degenerate generation loop producing semantically
meaningless text with characteristic patterns.

Each signal below has a threshold high enough to avoid false positives on
legitimate text (e.g. code blocks, URLs, repeated punctuation for formatting)
but low enough to catch the observed garbled patterns.
"""

import json
import logging
import re

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logger = logging.getLogger(__name__)


def find_garbled_signals(text: str) -> list[str]:
    """Analyze text for garbled output signals.

    Returns a list of human-readable descriptions of each signal found.
    Empty list means no garbled signals detected.

    The detection uses multiple independent signals. Each signal alone
    is a strong indicator; multiple signals together are definitive.
    """
    if not text:
        return []

    signals = []

    # --- Signal 1: Encoding-level corruption -------------------------------
    # These catch actual byte-level corruption (unlikely for this bug but
    # important as a safety net).
    for ch in text:
        if ch == "\x00":
            signals.append("NUL byte found in output")
            break
        if "\ud800" <= ch <= "\udfff":
            signals.append(f"Unpaired surrogate U+{ord(ch):04X} in output")
            break
        cp = ord(ch)
        if cp < 0x20 and ch not in ("\t", "\n", "\r"):
            signals.append(f"C0 control character U+{cp:04X} in output")
            break
    if "\ufffd" in text:
        signals.append("Unicode replacement character (U+FFFD) in output")

    # --- Signal 2: Run-on concatenated words --------------------------------
    # The garbled output contains massive runs of concatenated English words
    # without spaces, e.g. "DoneOutputGenerationInitiatedImmediately..."
    # Normal text: longest English word is ~30 chars. URLs/code may be longer
    # but rarely exceed 80 chars without any whitespace/punctuation break.
    # Threshold: 100+ alphabetic chars with no whitespace/punctuation.
    long_runs = re.findall(r"[A-Za-z]{100,}", text)
    if long_runs:
        longest = max(long_runs, key=len)
        signals.append(
            f"Run-on concatenated words: {len(long_runs)} run(s) found, "
            f"longest is {len(longest)} chars: {longest[:80]}..."
        )

    # --- Signal 3: Excessive word repetition --------------------------------
    # The garbled output repeats words many times consecutively,
    # e.g. "realistically realistically realistically realistically realistically..."
    # Normal emphasis uses 2-3 repetitions. Creative writing may use 4-8
    # repetitions for dramatic effect. Threshold: 9+ consecutive repeats
    # to avoid false positives on intentional wordplay.
    repeated = re.findall(r"\b(\w{3,})\b((?:\s+\1\b){8,})", text)
    if repeated:
        total_bursts = len(repeated)
        examples = []
        for word, _ in repeated[:3]:
            count = 1 + _.count(word)
            examples.append(f"'{word}' x{count}")
        signals.append(
            f"Excessive word repetition: {total_bursts} burst(s) — "
            f"{', '.join(examples)}"
        )

    # --- Signal 4: Excessive character repetition ---------------------------
    # The garbled output contains long runs of the same letter,
    # e.g. "AAAAAA", "ppppppppppp", "LLLLLOOOOOKKKKK"
    # Normal text: "..." (3 dots) or "——" (2 dashes). Threshold: 12+.
    char_runs = re.findall(r"([A-Za-z])\1{11,}", text)
    if char_runs:
        unique_chars = set(char_runs)
        longest_run = max(
            len(m.group(0))
            for ch in unique_chars
            for m in re.finditer(rf"({re.escape(ch)})\1{{11,}}", text)
        )
        signals.append(
            f"Excessive character repetition: chars {sorted(unique_chars)}, "
            f"max run length {longest_run}"
        )

    # --- Signal 5: Alphabetical enumeration pattern -------------------------
    # The garbled output enumerates letter combinations alphabetically,
    # e.g. "Ba Ba Ba Be Be Be Bi Bi Bi Bo Bo Bo"
    # This is a very specific garbled-output fingerprint.
    alphabetic_seq = re.findall(r"(?:\b[A-Z][a-z]\b\s*){8,}", text)
    if alphabetic_seq:
        signals.append(
            f"Alphabetical enumeration pattern: " f"{alphabetic_seq[0][:80].strip()}..."
        )

    # --- Signal 6: Single-char spaced-out pattern ---------------------------
    # e.g. "p p p p p p" or "t w i n e w i n e w i n e"
    single_char_spaced = re.findall(r"(?:(?:^|\s)[A-Za-z](?=\s|$)){8,}", text)
    if single_char_spaced:
        signals.append(
            f"Single-char spaced-out pattern: "
            f"{single_char_spaced[0][:80].strip()}..."
        )

    # --- Signal 7: Identical line repetition (block-level loop) --------------
    # When the model enters a degenerate loop producing repeated structured
    # blocks (e.g., the same test case definition cloned dozens of times),
    # individual lines recur excessively.
    #
    # Handle both plain text and JSON-embedded strings (e.g. tool call
    # arguments where newlines are JSON-escaped, appearing as a single line).

    def _collect_json_strings(obj):
        """Recursively collect all leaf string values from a JSON object."""
        strings = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(_collect_json_strings(v))
        elif isinstance(obj, list):
            for item in obj:
                strings.extend(_collect_json_strings(item))
        return strings

    def _check_line_repetition(s: str):
        """Check a single text block for repeated lines."""
        line_counts = {}
        for line in s.split("\n"):
            stripped = line.strip()
            if len(stripped) >= 30:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        if not line_counts:
            return None
        most_repeated = max(line_counts, key=line_counts.get)
        max_count = line_counts[most_repeated]
        if max_count >= 5:
            return (max_count, most_repeated)
        return None

    # Direct check on the raw text
    result = _check_line_repetition(text)
    if result:
        max_count, most_repeated = result
        signals.append(
            f"Identical line repeated {max_count} times: " f'"{most_repeated[:100]}"'
        )
    else:
        # Also try JSON nesting: parse text as JSON and check all leaf strings
        try:
            parsed = json.loads(text)
            nested_strings = _collect_json_strings(parsed)
            for ns in nested_strings:
                result = _check_line_repetition(ns)
                if result:
                    max_count, most_repeated = result
                    signals.append(
                        f"Identical line repeated {max_count} times in nested JSON: "
                        f'"{most_repeated[:100]}"'
                    )
                    break  # one match is enough
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return signals


class GarbledDetectionBase(CustomTestCase):
    """Base class for garbled detection tests.

    Subclasses set class attributes for model-specific configuration.
    setUpClass / tearDownClass are inherited.

    Reusable test logic lives in ``_run_*`` helpers. Subclasses must
    explicitly define ``test_*`` methods that delegate to these helpers,
    choosing which scenarios are relevant.

    Configurable attributes
    -----------------------
    model_path : str
        Path to model weights (required).
    server_args : list[str]
        CLI arguments passed to ``popen_launch_server``.
    server_envs : dict[str, str]
        Environment variables for the server process.
    system_prompt : str
        System prompt for streaming test. Empty string omits the
        system message from the request.
    user_prompt : str
        User prompt for streaming test.
    max_rounds : int
        Number of stress-test iterations (default: 100).
    api_key : str
        API key for server auth (default: "sk-1234").
    extra_payload : dict or None
        Extra fields for streaming requests; also merged into non-streaming
        request body via ``data.update()``. ``None`` (default) adds nothing.
    request_body : str or None
        Raw JSON body for ``_run_non_streaming_tool_call_scenario``.
        Set before calling the method (e.g. read from a file).
    """

    # --- Subclasses override these as needed ---
    model_path = None
    server_args = []
    server_envs = {}
    system_prompt = ""
    user_prompt = ""
    max_rounds = 100
    api_key = "sk-1234"
    # Extra fields merged into payload (streaming via **, non-streaming via update).
    extra_payload = None
    # Raw JSON body string for _run_non_streaming_tool_call_scenario.
    request_body = None

    @classmethod
    def setUpClass(cls):
        cls.model = cls.model_path
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=cls.server_args,
            env=cls.server_envs,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _send_request(self, payload, stream=True, timeout=None):
        """Send a chat completion request.

        Args:
            payload: Request payload dict.
            stream: Whether to stream the response.
            timeout: Request timeout in seconds (None for default).
        """
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = requests.post(
            self.base_url + "/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            stream=stream,
            **kwargs,
        )
        return response

    def _parse_streaming_response(self, response):
        """Parse SSE streaming response and return collected content.

        Returns a dict with keys: status_code, error, reasoning_content,
        content, chunk_count, chunks_with_errors, finish_reason_seen.
        """
        result = {
            "status_code": response.status_code,
            "error": None,
            "reasoning_content": "",
            "content": "",
            "chunk_count": 0,
            "chunks_with_errors": 0,
            "finish_reason_seen": False,
            "finish_reason": "",
        }

        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
            return result

        line_count = 0
        for line in response.iter_lines():
            line_count += 1
            if not line:
                continue

            try:
                line_str = line.decode("utf-8")
            except UnicodeDecodeError as e:
                raw_bytes_snippet = line[:200]
                result["chunks_with_errors"] += 1
                result["error"] = (
                    (result["error"] or "")
                    + f"UTF-8 decode error at line {line_count}: {e}; "
                    + f"raw_bytes={raw_bytes_snippet!r}; "
                )
                logger.error(
                    f"\n[GARBLED] UTF-8 decode error at stream line {line_count}: {e}"
                )
                logger.error(f"[GARBLED] raw bytes (first 200): {raw_bytes_snippet!r}")
                continue

            if not line_str.strip():
                continue

            # Handle "data: " SSE prefix
            if line_str.startswith("data:"):
                json_str = line_str[5:].strip()
            else:
                json_str = line_str.strip()

            if not json_str:
                continue

            # Skip SSE stream termination marker
            if json_str == "[DONE]":
                continue

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                raw_snippet = json_str[:500]
                result["chunks_with_errors"] += 1
                result["error"] = (
                    (result["error"] or "")
                    + f"JSON parse error at line {line_count}: {e}; "
                    + f"raw={raw_snippet!r}; "
                )
                logger.error(
                    f"\n[GARBLED] JSON parse error at stream line {line_count}: {e}"
                )
                logger.error(
                    f"[GARBLED] raw content (first 500 chars): {raw_snippet!r}"
                )
                continue

            if "choices" not in data or len(data["choices"]) == 0:
                continue

            choice = data["choices"][0]
            result["chunk_count"] += 1

            if "delta" in choice:
                delta = choice["delta"]
                if delta.get("reasoning_content"):
                    result["reasoning_content"] += delta["reasoning_content"]
                if delta.get("content"):
                    result["content"] += delta["content"]

            if choice.get("finish_reason") is not None:
                result["finish_reason_seen"] = True
                result["finish_reason"] = choice["finish_reason"]

        return result

    def _check_text_signals(self, text, context=""):
        """Assert text is free of garbled signals.  Shared by streaming and non-streaming paths."""
        if not text:
            return
        prefix = f"[{self._testMethodName}] {context}: " if context else ""
        signals = find_garbled_signals(text)
        if signals:
            self.fail(
                f"{prefix}Garbled output detected!\n"
                f"Signals: {'; '.join(signals)}\n"
                f"\n--- Text (first 2000 chars) Begin---\n"
                f"{text[:2000]}"
                f"\n--- Text End---\n"
            )

    def _assert_no_garbled(self, result, context=""):
        """Run all garbled checks on a parsed streaming response and assert cleanliness."""
        prefix = f"[{context}] " if context else ""

        # HTTP must succeed
        self.assertEqual(
            result["status_code"],
            200,
            f"{prefix}Expected HTTP 200, got {result['status_code']}. "
            f"Error: {result.get('error', '')}",
        )

        # No SSE/JSON parsing errors at the transport level
        self.assertEqual(
            result["chunks_with_errors"],
            0,
            f"{prefix}Found {result['chunks_with_errors']} SSE/JSON errors: "
            f"{result.get('error', '')}",
        )

        # Must receive stream chunks
        self.assertGreater(
            result["chunk_count"],
            0,
            f"{prefix}No chunks received in streaming response",
        )

        # Must have some output (content or reasoning_content)
        total_len = len(result["content"]) + len(result["reasoning_content"])
        self.assertGreater(
            total_len,
            0,
            f"{prefix}Response empty — no content or reasoning. Result: {result}",
        )

        # Must complete normally (have finish_reason)
        self.assertTrue(
            result["finish_reason_seen"],
            f"{prefix}Response did not complete — no finish_reason received",
        )

        # Detect model stuck in thinking loop: reasoning but no content
        if len(result["reasoning_content"]) > 0 and len(result["content"]) == 0:
            self.fail(
                f"{prefix}Model stuck in thinking loop: "
                f"reasoning_content is {len(result['reasoning_content'])} chars "
                f"but content is empty — model never exited thinking mode.\n"
                f"\n--- FULL Reasoning Content (last 500 chars) Begin---\n"
                f"{result['reasoning_content'][-500:]}"
                f"\n--- FULL Reasoning Content End---\n"
            )

        # --- Garbled signal detection ---
        self._check_text_signals(result["content"], f"{context} content")
        self._check_text_signals(result["reasoning_content"], f"{context} reasoning")

    def _run_streaming_no_garbled(self):
        """Reusable streaming garbled-detection test logic.

        Sends ``system_prompt`` + ``user_prompt`` as messages and validates
        the streaming response is clean.  When ``system_prompt`` is empty
        the request is sent without a system message.

        Subclasses should define a ``test_*`` method that calls this, e.g.::

            def test_streaming_no_garbled(self):
                self._run_streaming_no_garbled()
        """
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": self.user_prompt}]

        for i in range(self.max_rounds):
            logger.info(f"===== Iteration {i}/{self.max_rounds} =====")
            response = self._send_request(
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    **(self.extra_payload or {}),
                },
                stream=True,
            )
            result = self._parse_streaming_response(response)
            logger.info(f"finish_reason: {result['finish_reason']}")
            self._assert_no_garbled(result, context=f"run_streaming_no_garbled[{i}]")

    def _run_non_streaming_tool_call_scenario(self):
        """Reusable non-streaming tool-call garbled-detection test logic.

        Uses ``self.request_body`` as the raw JSON request body string,
        sends a non-streaming POST, then checks the response content,
        reasoning and tool-call arguments for garbled signals.

        The caller is responsible for setting ``self.request_body`` before
        calling this method (e.g. reading from a file).

        Subclasses should define a ``test_*`` method that calls this, e.g.::

            def test_tool_call_scenario(self):
                with open("tool_calls.json", "r") as f:
                    self.request_body = f.read()
                self._run_non_streaming_tool_call_scenario()
        """

        raw_body = self.request_body
        self.assertIsNotNone(raw_body, "request_body not set")

        logger.info(
            f"[{self._testMethodName}] ======== Start: tool call scenario ========"
        )

        logger.info(
            f"[{self._testMethodName}] Request body length: {len(raw_body)} chars"
        )

        data = json.loads(raw_body)
        data.update(self.extra_payload or {})
        logger.info(
            f"[{self._testMethodName}] Local parse OK: "
            f"messages={len(data.get('messages', []))}, "
            f"tools={len(data.get('tools', []))}"
        )

        for iteration in range(1, self.max_rounds + 1):
            logger.info(
                f"[{self._testMethodName}] ======== Iteration {iteration}/{self.max_rounds} ========"
            )

            try:
                resp = self._send_request(data, stream=False, timeout=300)
            except requests.exceptions.ConnectionError:
                self.fail(f"Cannot connect to {self.base_url}/chat/completions")
            except Exception as e:
                self.fail(f"Request error: {e}")

            logger.info(f"[{self._testMethodName}] Status code: {resp.status_code}")
            self.assertEqual(
                resp.status_code,
                200,
                f"Expected 200, got {resp.status_code}: {resp.text[:1000]}",
            )

            resp_data = resp.json()
            choice = resp_data.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason")
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            reasoning_content = msg.get("reasoning_content") or ""
            tool_calls = msg.get("tool_calls")

            logger.info(f"[{self._testMethodName}] finish_reason: {finish_reason}")

            self._check_text_signals(content, f"Iter {iteration} content")
            self._check_text_signals(
                reasoning_content, f"Iter {iteration} reasoning_content"
            )

            if tool_calls:
                self.assertIsInstance(tool_calls, list)
                self.assertGreater(len(tool_calls), 0)
                logger.info(
                    f"[{self._testMethodName}] tool_calls count: {len(tool_calls)}"
                )

                for i, tc in enumerate(tool_calls):
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    args_raw = fn.get("arguments", "")
                    logger.info(f"[{self._testMethodName}]   [{i}] function: {name}")

                    self._check_text_signals(args_raw, f"Iter {iteration} tc[{i}] args")

                    try:
                        parsed = json.loads(args_raw)
                        self.assertIsInstance(
                            parsed,
                            dict,
                            f"Arguments should parse to dict, got {type(parsed)}",
                        )
                    except json.JSONDecodeError as e:
                        self.fail(f"Tool call arguments not valid JSON: {e}")

            if finish_reason not in ("tool_calls", "stop"):
                self.fail(
                    f"Iteration {iteration}: Expected finish_reason='tool_calls' or 'stop', got '{finish_reason}'"
                )
        logger.info(
            f"[{self._testMethodName}] ======== All {self.max_rounds} iterations passed ========"
        )
