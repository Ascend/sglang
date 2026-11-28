#!/usr/bin/env python3
"""
GSM8K 精度测试脚本(NPU 适配专用)。

通过 OpenAI 兼容接口向 SGLang server 发送 few-shot GSM8K 请求,
提取预测答案并计算 accuracy / invalid 比例。

执行模式:
  - 本机执行(默认):直接向 --port 发 HTTP 请求,支持 --parallel 并发
  - 容器内执行(可选):加 --docker-container <name>,通过 docker exec 在容器内发请求

判断标准:
  - accuracy >= threshold 且 invalid == 0 -> 通过
  - 否则 -> 不通过

用法:
    python gsm8k_accuracy_test.py \
        --port 8000 \
        --workspace-dir /path/to/workspace \
        --data-path /path/to/gsm8k.jsonl \
        --num-questions 10 \
        --min-accuracy auto \
        --parallel 4

退出码:
    0 - 精度通过
    1 - 精度不通过
    2 - 服务器不可达、数据缺失或测试执行失败
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def read_jsonl(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def http_json(url, payload=None, timeout=120):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:  # noqa: BLE001
        return None, {"error": str(exc)}


def fetch_model_name(port, docker_container=None):
    """从 /v1/models 取真实 model id,失败时退回 'default'。"""
    url = f"http://127.0.0.1:{port}/v1/models"
    if docker_container:
        try:
            proc = subprocess.run(
                ["docker", "exec", docker_container, "curl", "-s", url],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                items = data.get("data") or data.get("models") or []
                if items and items[0].get("id"):
                    return items[0]["id"]
        except Exception:  # noqa: BLE001
            pass
        return "default"

    status, data = http_json(url, timeout=5)
    if status == 200:
        items = data.get("data") or data.get("models") or []
        if items and items[0].get("id"):
            return items[0]["id"]
    return "default"


def wait_server(port, max_wait=120, docker_container=None):
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if docker_container:
            try:
                proc = subprocess.run(
                    ["docker", "exec", docker_container, "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                    capture_output=True, text=True, timeout=5,
                )
                if proc.returncode == 0 and proc.stdout.strip() == "200":
                    return True
            except Exception:  # noqa: BLE001
                pass
        else:
            status, _ = http_json(url, timeout=5)
            if status == 200:
                return True
        time.sleep(5)
    return False


def resolve_min_accuracy(raw, workspace_dir):
    """auto: >=5B 用 0.8,<5B 用 0.3。无法判断参数量时退回 0.3 并告警。"""
    if raw != "auto":
        return float(raw)

    params_b = None
    summary_path = os.path.join(workspace_dir, "output", "output_summary.json")
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        params_b = summary.get("params_b")
        if params_b is None:
            params_b = (summary.get("resource") or {}).get("params_b")
    except Exception:  # noqa: BLE001
        params_b = None

    if params_b is None:
        print("[gsm8k] 警告: output_summary.json 中没有 params_b,auto 阈值退回 0.3", file=sys.stderr)
        return 0.3
    threshold = 0.8 if float(params_b) >= 5 else 0.3
    print(f"[gsm8k] auto 阈值: params_b={params_b} -> {threshold}", file=sys.stderr)
    return threshold


REMOTE_CODE = r"""
import ast
import json
import re
import sys
import urllib.request

INVALID = -9999999


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


payload = json.loads(sys.argv[1])
url = payload.pop("__url__")
timeout = payload.pop("__timeout__", 120)
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())
        answer_text = result["choices"][0]["message"]["content"]
        print(json.dumps({
            "pred": get_answer_value(answer_text),
            "usage": result.get("usage"),
        }))
except Exception as exc:
    print(json.dumps({"pred": INVALID, "error": str(exc)}))
"""


def ask_docker(container, url, model_name, prompt, max_new_tokens, timeout):
    payload = {
        "__url__": url,
        "__timeout__": timeout,
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    try:
        proc = subprocess.run(
            ["docker", "exec", container, "python3", "-c", REMOTE_CODE, json.dumps(payload)],
            capture_output=True, text=True, timeout=timeout + 30,
        )
        if not proc.stdout.strip():
            return INVALID, 0
        parsed = json.loads(proc.stdout)
        usage = parsed.get("usage") or {}
        return parsed.get("pred", INVALID), int(usage.get("completion_tokens") or 0)
    except Exception:  # noqa: BLE001
        return INVALID, 0


def ask_local(url, model_name, prompt, max_new_tokens, timeout):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }
    status, data = http_json(url, payload=payload, timeout=timeout)
    if status != 200:
        return INVALID, 0
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return INVALID, 0
    usage = data.get("usage") or {}
    return get_answer_value(text), int(usage.get("completion_tokens") or 0)


def run_gsm8k(port, data_path, num_questions, num_shots, max_new_tokens,
              docker_container, model_name, parallel, http_timeout):
    lines = read_jsonl(data_path)
    if num_questions < 0:
        num_questions = len(lines) - num_shots
    if num_questions <= 0:
        raise ValueError(f"num_questions={num_questions} 无效")
    if len(lines) < num_shots + num_questions:
        raise ValueError(
            f"数据不足: 需要 {num_shots} shots + {num_questions} 题,实际只有 {len(lines)} 行"
        )

    few_shot_examples = get_few_shot_examples(lines, num_shots)
    questions = []
    labels = []
    for i in range(num_questions):
        questions.append(get_one_example(lines, i + num_shots, False))
        labels.append(get_answer_value(lines[i + num_shots]["answer"]))

    prompts = [few_shot_examples + q for q in questions]
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    preds = [INVALID] * num_questions
    completion_tokens = 0
    tic = time.perf_counter()

    if docker_container:
        for i, prompt in enumerate(prompts):
            preds[i], tokens = ask_docker(
                docker_container, url, model_name, prompt, max_new_tokens, http_timeout
            )
            completion_tokens += tokens
    else:
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futures = {
                pool.submit(ask_local, url, model_name, prompt, max_new_tokens, http_timeout): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                preds[idx], tokens = future.result()
                completion_tokens += tokens

    latency = time.perf_counter() - tic

    correct = sum(1 for p, label in zip(preds, labels) if p == label)
    invalid_count = sum(1 for p in preds if p == INVALID)

    for i in range(min(10, num_questions)):
        print(f"  Q{i}: pred={preds[i]}, label={labels[i]}, match={preds[i] == labels[i]}", file=sys.stderr)

    return {
        "accuracy": correct / num_questions if num_questions else 0.0,
        "invalid": invalid_count / num_questions if num_questions else 1.0,
        "correct": correct,
        "invalid_count": invalid_count,
        "num_questions": num_questions,
        "latency": latency,
        "completion_tokens": completion_tokens,
        "output_throughput": (completion_tokens / latency) if latency > 0 else 0.0,
        "mode": "docker" if docker_container else "local",
        "parallel": parallel if not docker_container else 1,
    }


def main():
    ap = argparse.ArgumentParser(description="GSM8K accuracy test for NPU model adaptation")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-path", type=str, default=None,
                    help="GSM8K jsonl 路径;默认取 workspace/data/gsm8k.jsonl 或 adapter_state.paths.gsm8k_data_path")
    ap.add_argument("--num-questions", type=int, default=10, help="测试题数;-1 表示使用数据中全部剩余题目")
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=1, help="并发请求数(本机模式;docker 模式固定串行)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--workspace-dir", required=True)
    ap.add_argument("--min-accuracy", type=str, default="auto", help="最低精度阈值,或 'auto'(按参数量选择)")
    ap.add_argument("--max-invalid", type=float, default=0.0)
    ap.add_argument("--wait", type=int, default=120)
    ap.add_argument("--http-timeout", type=int, default=120)
    ap.add_argument("--model-name", type=str, default=None, help="请求中使用的 model id;默认从 /v1/models 读取")
    ap.add_argument("--docker-container", type=str, default=None,
                    help="可选:在该容器内发请求;不传则本机直接请求 --port")
    args = ap.parse_args()

    result = {
        "status": "unknown",
        "port": args.port,
        "min_accuracy": None,
        "max_invalid": args.max_invalid,
        "data_path": None,
        "metrics": None,
        "passed": False,
        "reason": None,
    }

    # 解析数据路径
    data_path = args.data_path
    if not data_path:
        try:
            with open(os.path.join(args.workspace_dir, "adapter_state.json"), encoding="utf-8") as f:
                state = json.load(f)
            data_path = (state.get("paths") or {}).get("gsm8k_data_path")
        except Exception:  # noqa: BLE001
            state = {}
        if not data_path:
            data_path = os.path.join(args.workspace_dir, "data", "gsm8k.jsonl")
    result["data_path"] = data_path

    if not os.path.isfile(data_path):
        result["status"] = "failed"
        result["reason"] = f"gsm8k_data_not_found:{data_path}"
        _write_result(args.workspace_dir, result)
        print(f"[gsm8k] 错误: 未找到数据文件 {data_path}", file=sys.stderr)
        print("[gsm8k] 请下载 GSM8K jsonl(字段: question, answer)后重试,"
              "或通过 --data-path / adapter_state.paths.gsm8k_data_path 指定", file=sys.stderr)
        sys.exit(2)

    min_accuracy = resolve_min_accuracy(args.min_accuracy, args.workspace_dir)
    result["min_accuracy"] = min_accuracy

    print(f"Waiting for server on port {args.port}...", file=sys.stderr)
    if not wait_server(args.port, args.wait, args.docker_container):
        result["status"] = "failed"
        result["reason"] = f"server not ready within {args.wait}s"
        _write_result(args.workspace_dir, result)
        sys.exit(2)

    model_name = args.model_name or fetch_model_name(args.port, args.docker_container)
    print(f"[gsm8k] model={model_name} mode={'docker:' + args.docker_container if args.docker_container else 'local'}", file=sys.stderr)

    try:
        metrics = run_gsm8k(
            args.port, data_path, args.num_questions, args.num_shots, args.max_new_tokens,
            args.docker_container, model_name, args.parallel, args.http_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["reason"] = f"gsm8k_test_execution_error:{exc}"
        _write_result(args.workspace_dir, result)
        sys.exit(2)

    result["metrics"] = metrics
    accuracy = metrics.get("accuracy", 0)
    invalid = metrics.get("invalid", 1)

    print(f"Accuracy: {accuracy:.3f} (threshold: {min_accuracy})", file=sys.stderr)
    print(f"Invalid: {invalid:.3f} (threshold: {args.max_invalid})", file=sys.stderr)

    if accuracy >= min_accuracy and invalid <= args.max_invalid:
        result["status"] = "passed"
        result["passed"] = True
        _write_result(args.workspace_dir, result)
        print(f"PASS: accuracy={accuracy:.3f} >= {min_accuracy}, invalid={invalid:.3f} <= {args.max_invalid}", file=sys.stderr)
        sys.exit(0)

    result["status"] = "failed"
    result["passed"] = False
    if accuracy < min_accuracy:
        result["reason"] = f"accuracy_{accuracy:.3f}_below_threshold_{min_accuracy}"
    else:
        result["reason"] = f"invalid_{invalid:.3f}_above_threshold_{args.max_invalid}"
    _write_result(args.workspace_dir, result)
    print(f"FAIL: {result['reason']}", file=sys.stderr)
    sys.exit(1)


def _write_result(workspace_dir, result):
    output_dir = os.path.join(workspace_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gsm8k_accuracy_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Result written to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
