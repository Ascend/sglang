#!/usr/bin/env python3
"""
对运行中的 SGLang server 跑推理冒烟测试。

用法:
    python run_tests.py --port 8000 --wait 300 --mode quick --output /path/to/result.json

mode:
    quick  - 单条 chat completion 请求(<10 tokens)
    full   - chat + completion 两种,且断言响应非空、HTTP 200

退出码:
    0  - 通过(json status="passed")
    1  - 失败(json status="failed")
    2  - 不可达(server 没起来/等不到)
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def http_get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return None, str(e)


def http_post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return None, str(e)


def wait_server(port, max_wait):
    """轮询 /v1/models,直到 200 或超时。"""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + max_wait
    last_err = None
    while time.time() < deadline:
        code, body = http_get(url, timeout=5)
        if code == 200:
            return True, body
        last_err = (code, body)
        time.sleep(5)
    return False, last_err


def test_chat_quick(port):
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    code, body = http_post_json(url, payload, timeout=60)
    if code != 200:
        return False, {"http_code": code, "body": body[:500]}
    try:
        resp = json.loads(body)
        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not isinstance(text, str) or len(text) == 0:
            return False, {"reason": "empty_response", "body": body[:500]}
        return True, {"text": text, "usage": resp.get("usage")}
    except Exception as e:
        return False, {"reason": "parse_error", "error": str(e), "body": body[:500]}


def test_completions(port):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": "default",
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0,
    }
    code, body = http_post_json(url, payload, timeout=60)
    if code != 200:
        return False, {"http_code": code, "body": body[:500]}
    try:
        resp = json.loads(body)
        text = resp.get("choices", [{}])[0].get("text", "")
        if not isinstance(text, str) or len(text) == 0:
            return False, {"reason": "empty_response", "body": body[:500]}
        return True, {"text": text}
    except Exception as e:
        return False, {"reason": "parse_error", "error": str(e), "body": body[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--wait", type=int, default=300, help="启动等待秒数")
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    result = {
        "status": "unknown",
        "port": args.port,
        "mode": args.mode,
        "phases": {},
    }

    # Phase: 等启动
    ready, ready_info = wait_server(args.port, args.wait)
    result["phases"]["wait_server"] = {
        "ready": ready,
        "info": str(ready_info)[:500] if ready_info else None,
    }
    if not ready:
        result["status"] = "failed"
        result["reason"] = f"server not ready within {args.wait}s"
        _write(args.output, result)
        sys.exit(2)

    # Phase: chat quick
    ok, info = test_chat_quick(args.port)
    result["phases"]["chat_quick"] = {"ok": ok, **info}
    if not ok:
        result["status"] = "failed"
        result["reason"] = "chat_quick failed"
        _write(args.output, result)
        sys.exit(1)

    # Phase: completions (full 模式)
    if args.mode == "full":
        ok, info = test_completions(args.port)
        result["phases"]["completions"] = {"ok": ok, **info}
        if not ok:
            result["status"] = "failed"
            result["reason"] = "completions failed"
            _write(args.output, result)
            sys.exit(1)

    result["status"] = "passed"
    _write(args.output, result)
    sys.exit(0)


def _write(path, obj):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
