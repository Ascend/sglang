#!/usr/bin/env python3
"""
聚合 workspace 下的产物,生成最终中文教程到 output/<ModelName>.md。

用法:
    python generate_report.py --workspace <WORKSPACE_DIR> --model <ModelName> --output <out_path>
"""

import argparse
import datetime
import json
import os
import sys


def load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def section(title, body):
    return f"## {title}\n\n{body}\n"


def fmt_feature_matrix(matrix):
    if not matrix:
        return "(无)"
    lines = ["| 特性 | 状态 | 证据 |", "|------|------|------|"]
    for k, v in matrix.items():
        status = v.get("status", "unknown") if isinstance(v, dict) else str(v)
        evidence = v.get("evidence", "") if isinstance(v, dict) else ""
        lines.append(f"| {k} | {status} | {evidence} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ws = args.workspace
    state = load_json(os.path.join(ws, "adapter_state.json"), {})
    arch = load_json(os.path.join(ws, "output", "output_summary.json"), {})
    test = load_json(os.path.join(ws, "output", "test_result.json"), {})

    sections = []

    # 标题
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(f"# {args.model} NPU 适配报告\n\n> 生成时间: {now}\n> 工作目录: {ws}\n")

    # 1. 模型概况
    body = f"""- 模型名称: {arch.get('model_name', args.model)}
- HF 架构: `{arch.get('hf_architecture', 'unknown')}`
- 模型类型: {arch.get('model_class', 'unknown')}
- 参数量: {arch.get('params_b', '?')}B
- 隐藏维度: {arch.get('hidden_size', '?')}
- 层数: {arch.get('num_layers', '?')}
- attention heads: {arch.get('num_attention_heads', '?')} (KV: {arch.get('num_kv_heads', '?')})
- intermediate_size: {arch.get('intermediate_size', '?')}
- vocab_size: {arch.get('vocab_size', '?')}
- max_position: {arch.get('max_position', '?')}
- rope_theta: {arch.get('rope_theta', '?')}
- torch_dtype: {arch.get('torch_dtype', '?')}
"""
    sections.append(section("1. 模型概况", body))

    # 2. 适配策略
    body = f"""- 参考实现: `{arch.get('reference_file', 'unknown')}` ({arch.get('reference_model', '?')})
- 相似度: **{arch.get('similarity', 'unknown')}**
- 差异摘要: {arch.get('diff_summary', '(无)')}
- 适配策略: **{arch.get('adapter_strategy', 'unknown')}**
- 是否需要新适配器: {arch.get('requires_new_adapter', '?')}
"""
    sections.append(section("2. 适配策略", body))

    # 3. 并行配置建议
    pcs = arch.get("parallel_config_suggestion", {})
    body = f"""- tp_size: {pcs.get('tp_size', '?')}
- dp_size: {pcs.get('dp_size', '?')}
- ep_size: {pcs.get('ep_size', '?')}
- 理由: {pcs.get('rationale', '(无)')}
"""
    sections.append(section("3. 并行配置建议", body))

    # 4. 特性兼容性(架构分析师预判)
    body = fmt_feature_matrix(arch.get("feature_compatibility", {}))
    sections.append(section("4. 特性兼容性(架构分析预判)", body))

    # 5. 测试结果(test-validator 实测)
    body = f"**总体状态:** {test.get('status', 'unknown')}\n\n"
    body += "**实测矩阵:**\n\n"
    body += fmt_feature_matrix(test.get("feature_matrix", {}))
    cb = test.get("capacity_baseline") or {}
    if cb:
        body += "\n\n**容量基线:**\n\n"
        body += f"- max_context_len: {cb.get('max_context_len', '?')}\n"
        body += f"- max_batch_size: {cb.get('max_batch_size', '?')}\n"
        body += f"- throughput (tok/s): {cb.get('throughput_tokens_per_sec', '?')}\n"
        body += f"- latency p50/p99 (ms): {cb.get('latency_p50_ms', '?')} / {cb.get('latency_p99_ms', '?')}\n"
        body += f"- 测试命令: `{cb.get('test_command', '?')}`\n"
        body += f"- 证据: `{cb.get('evidence', '?')}`\n"
    regs = test.get("regressions") or []
    if regs:
        body += "\n**已知 regression:**\n\n"
        for r in regs:
            body += f"- {r.get('feature', '?')}: {r.get('phenomenon', '?')}\n"
    sections.append(section("5. 测试结果", body))

    # 6. 推荐启动命令
    cmd = arch.get("launch_command_template", "(无)")
    sections.append(section("6. 推荐启动命令", f"```bash\n{cmd}\n```"))

    # 7. 风险与遗留
    risks = arch.get("risks", []) or []
    body = "\n".join(f"- {r}" for r in risks) if risks else "(无)"
    sections.append(section("7. 风险与遗留", body))

    # 8. 验证状态
    v = state.get("validation", {})
    body = f"""- Dummy 权重验证: {'✓ passed' if v.get('dummy_passed') else '✗ not passed'}
- 真实权重验证: {'✓ passed' if v.get('real_weight_passed') else '✗ not passed'}
- precision_suspect: {state.get('precision_suspect', False)}
"""
    if state.get("precision_suspect"):
        rca = load_json(os.path.join(ws, "output", "root_cause.json"), {})
        body += f"- 精度 RCA status: {rca.get('status', 'missing')}\n"
        body += f"- 候选根因数: {len(rca.get('hypotheses', []))}\n"
    sections.append(section("8. 验证状态", body))

    # 9. 文件清单
    body = f"""- 架构分析: `output/output_summary.json`, `output/analysis_report.md`
- 测试结果: `output/test_result.json`, `output/test_report.md`
- 调试历史: `output/fix_instructions.json`, `output/debug_report.md`(若有错误发生)
- 验证日志: `logs/dummy_run.log`, `logs/dummy_inference.json`, `logs/real_run.log`, `logs/real_inference.json`
- Agent 调用审计: `logs/agent_calls/index.jsonl`
"""
    sections.append(section("9. 产物清单", body))

    # 输出
    content = "\n".join(sections)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[generate_report] 写入 {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
