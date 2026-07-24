#!/usr/bin/env python3
"""
GSM8K 精度测试脚本（NPU 适配专用）。

通过 chat completions API 向 SGLang server 发送 few-shot GSM8K 请求，
使用 SGLang 的 get_answer_value 提取预测答案并计算精度。

支持两种执行模式：
  - 容器内直接执行（推荐）：通过 --docker-container 参数指定容器名
  - 容器外子进程执行（备选）：通过 --use-subprocess 标志

判断标准：
  - accuracy >= threshold 且 invalid_count == 0 -> 通过
  - 否则 -> 不通过，需要重新适配算子

用法:
    python gsm8k_accuracy_test.py \
        --port 8000 \
        --data-path /home/w00937173/run_file/gsm8k.jsonl \
        --num-questions 10 \
        --workspace-dir /path/to/workspace \
        --docker-container sgl_wm \
        [--parallel 1] \
        [--max-new-tokens 512] \
        [--min-accuracy 0.3]

退出码:
    0 - 精度通过
    1 - 精度不通过
    2 - 服务器不可达或测试执行失败
"""

import ast
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

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


def wait_server(port, max_wait=120, docker_container=None):
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            cmd_prefix = []
            if docker_container:
                cmd_prefix = ["docker", "exec", docker_container]
            check_cmd = cmd_prefix + ["curl", "-s", url]
            result = subprocess.run(check_cmd, capture_output=True, timeout=5)
            if result.returncode == 0 and result.stdout:
                return True
        except (subprocess.TimeoutExpired, Exception):
            pass
        time.sleep(5)
    return False


def run_gsm8k_in_container(port, data_path, num_questions, num_shots, max_new_tokens, docker_container, workspace_dir):
    lines = read_jsonl(data_path)
    few_shot_examples = get_few_shot_examples(lines, num_shots)
    questions = []
    labels = []
    for i in range(num_questions):
        questions.append(get_one_example(lines, i + num_shots, False))
        labels.append(get_answer_value(lines[i + num_shots]["answer"]))

    model_name = "/home/weight/granite-4.0-h-micro"
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    preds = []
    tic = time.perf_counter()
    for i in range(num_questions):
        prompt = few_shot_examples + questions[i]
        data = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0
        }).encode()
        cmd = ["docker", "exec", docker_container, "python3", "-u", "-c", """
import json, urllib.request, sys, ast, re

INVALID = -9999999

def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\\d+\\.?\\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID

url = "%s"
data = json.dumps({
    "model": "%s",
    "messages": [{"role": "user", "content": %s}],
    "max_tokens": %d,
    "temperature": 0
}).encode()
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())
        answer_text = result["choices"][0]["message"]["content"]
        pred = get_answer_value(answer_text)
        print(json.dumps({"pred": pred, "answer_len": len(answer_text)}))
except Exception as e:
    print(json.dumps({"pred": INVALID, "error": str(e)}))
""" % (url, model_name, json.dumps(prompt), max_new_tokens)]

        try:
            result_proc = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
            output = result_proc.stdout.strip()
            if output:
                parsed = json.loads(output)
                pred = parsed.get("pred", INVALID)
            else:
                pred = INVALID
        except Exception as e:
            pred = INVALID

        preds.append(pred)
        correct = pred == labels[i]
        if i < 10:
            print(f"  Q{i}: pred={pred}, label={labels[i]}, match={correct}", file=sys.stderr)

    latency = time.perf_counter() - tic

    import numpy as np
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid_ratio = np.mean(np.array(preds) == INVALID)

    return {
        "accuracy": float(acc),
        "invalid": float(invalid_ratio),
        "latency": float(latency),
        "output_throughput": 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="GSM8K accuracy test for NPU model adaptation")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-path", type=str, default="/home/w00937173/run_file/gsm8k.jsonl")
    ap.add_argument("--num-questions", type=int, default=10)
    ap.add_argument("--num-shots", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--workspace-dir", required=True)
    ap.add_argument("--min-accuracy", type=float, default=0.3)
    ap.add_argument("--max-invalid", type=float, default=0.0)
    ap.add_argument("--wait", type=int, default=120)
    ap.add_argument("--docker-container", type=str, default="sgl_wm", help="Docker container name for NPU execution")
    args = ap.parse_args()

    result = {
        "status": "unknown",
        "port": args.port,
        "min_accuracy": args.min_accuracy,
        "max_invalid": args.max_invalid,
        "metrics": None,
        "passed": False,
        "reason": None,
    }

    print(f"Waiting for server on port {args.port}...", file=sys.stderr)
    ready = wait_server(args.port, args.wait, args.docker_container)
    if not ready:
        result["status"] = "failed"
        result["reason"] = f"server not ready within {args.wait}s"
        _write_result(args.workspace_dir, result)
        sys.exit(2)

    print(f"Server ready. Running GSM8K test via container {args.docker_container}...", file=sys.stderr)

    metrics = run_gsm8k_in_container(
        args.port, args.data_path, args.num_questions,
        args.num_shots, args.max_new_tokens,
        args.docker_container, args.workspace_dir
    )

    if metrics is None:
        result["status"] = "failed"
        result["reason"] = "gsm8k_test_execution_error"
        _write_result(args.workspace_dir, result)
        sys.exit(2)

    result["metrics"] = metrics
    accuracy = metrics.get("accuracy", 0)
    invalid = metrics.get("invalid", 1)

    print(f"Accuracy: {accuracy:.3f} (threshold: {args.min_accuracy})", file=sys.stderr)
    print(f"Invalid: {invalid:.3f} (threshold: {args.max_invalid})", file=sys.stderr)

    if accuracy >= args.min_accuracy and invalid <= args.max_invalid:
        result["status"] = "passed"
        result["passed"] = True
        _write_result(args.workspace_dir, result)
        print(f"PASS: accuracy={accuracy:.3f} >= {args.min_accuracy}, invalid={invalid:.3f} <= {args.max_invalid}", file=sys.stderr)
        sys.exit(0)
    else:
        result["status"] = "failed"
        result["passed"] = False
        if accuracy < args.min_accuracy:
            result["reason"] = f"accuracy_{accuracy:.3f}_below_threshold_{args.min_accuracy}"
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