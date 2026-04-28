#!/usr/bin/env python3
"""
测试验证脚本
发送测试请求并验证模型推理
"""

import json
import time
import argparse
import sys
from typing import Optional, Dict, Any, List


def send_request(url: str, messages: List[Dict], max_tokens: int = 50, timeout: int = 180) -> Dict[str, Any]:
    """发送推理请求"""
    import requests
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        latency_ms = int((time.time() - start_time) * 1000)
        
        if response.status_code == 200:
            result = response.json()
            return {
                "status": "success",
                "latency_ms": latency_ms,
                "output": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "raw_response": result
            }
        else:
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": int((time.time() - start_time) * 1000),
            "error": str(e)
        }


def check_service_ready(port: int, timeout: int = 300) -> bool:
    """检查服务是否就绪"""
    import requests
    
    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    
    return False


def run_test_case(port: int, test_case: Dict) -> Dict:
    """运行单个测试用例"""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    case_id = test_case.get("case_id", 0)
    case_name = test_case.get("case_name", "unknown")
    messages = test_case.get("messages", [])
    max_tokens = test_case.get("max_tokens", 50)
    expected_pattern = test_case.get("expected_pattern")
    expected_min_length = test_case.get("expected_min_length", 0)
    
    result = {
        "case_id": case_id,
        "case_name": case_name,
        "status": "pending",
        "input": str(messages),
        "output": "",
        "latency_ms": 0,
        "error_message": None
    }
    
    if not messages:
        result["status"] = "skipped"
        result["error_message"] = "无输入消息"
        return result
    
    response = send_request(url, messages, max_tokens)
    
    if response["status"] == "success":
        output = response["output"]
        result["output"] = output
        result["latency_ms"] = response["latency_ms"]
        
        passed = True
        if expected_pattern:
            import re
            if not re.search(expected_pattern, output):
                passed = False
                result["error_message"] = f"输出不匹配预期模式: {expected_pattern}"
        
        if expected_min_length > 0 and len(output) < expected_min_length:
            passed = False
            result["error_message"] = f"输出长度不足: {len(output)} < {expected_min_length}"
        
        result["status"] = "passed" if passed else "failed"
    else:
        result["status"] = "failed"
        result["error_message"] = response.get("error", "未知错误")
    
    return result


def run_all_tests(port: int, test_mode: str = "quick") -> Dict:
    """运行所有测试用例"""
    
    test_cases = [
        {
            "case_id": 1,
            "case_name": "short_text_inference",
            "messages": [{"role": "user", "content": "1+1=?"}],
            "max_tokens": 10,
            "expected_pattern": ".*2.*"
        },
        {
            "case_id": 2,
            "case_name": "long_text_inference",
            "messages": [{"role": "user", "content": "请写一篇关于人工智能的短文"}],
            "max_tokens": 200,
            "expected_min_length": 50
        },
        {
            "case_id": 3,
            "case_name": "multi_turn_dialog",
            "messages": [
                {"role": "user", "content": "我叫张三"},
                {"role": "assistant", "content": "你好，张三！"},
                {"role": "user", "content": "我叫什么名字？"}
            ],
            "max_tokens": 20,
            "expected_pattern": ".*张三.*"
        }
    ]
    
    results = {
        "status": "pending",
        "overall_result": "pending",
        "test_mode": test_mode,
        "test_cases": [],
        "passed_count": 0,
        "failed_count": 0,
        "total_count": len(test_cases),
        "issues": [],
        "recommendations": []
    }
    
    print(f"开始测试 (模式: {test_mode})")
    print(f"服务端口: {port}")
    print("-" * 40)
    
    for case in test_cases:
        print(f"运行用例 {case['case_id']}: {case['case_name']}...")
        result = run_test_case(port, case)
        results["test_cases"].append(result)
        
        if result["status"] == "passed":
            results["passed_count"] += 1
            print(f"  ✓ 通过 ({result['latency_ms']}ms)")
        else:
            results["failed_count"] += 1
            print(f"  ✗ 失败: {result['error_message']}")
            results["issues"].append({
                "severity": "high",
                "category": "correctness",
                "description": f"测试用例{case['case_id']}失败: {result['error_message']}",
                "suggestion": "检查模型输出或调整测试预期"
            })
    
    print("-" * 40)
    
    if results["failed_count"] == 0:
        results["status"] = "passed"
        results["overall_result"] = "passed"
        print(f"结果: 全部通过 ({results['passed_count']}/{results['total_count']})")
    else:
        results["status"] = "failed"
        results["overall_result"] = "failed"
        print(f"结果: 部分失败 ({results['passed_count']}/{results['total_count']})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="测试验证脚本")
    parser.add_argument("--port", "-p", type=int, default=8000, help="服务端口")
    parser.add_argument("--mode", "-m", choices=["quick", "standard", "full"], default="quick", help="测试模式")
    parser.add_argument("--output", "-o", help="输出JSON文件路径")
    parser.add_argument("--wait", "-w", type=int, default=0, help="等待服务就绪的超时时间(秒)")
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"等待服务就绪 (超时: {args.wait}秒)...")
        if not check_service_ready(args.port, args.wait):
            print("错误: 服务未能在指定时间内就绪")
            results = {
                "status": "error",
                "overall_result": "service_not_ready",
                "error_message": "服务未能在指定时间内就绪"
            }
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            sys.exit(1)
        print("服务已就绪")
    
    results = run_all_tests(args.port, args.mode)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")
    
    sys.exit(0 if results["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
