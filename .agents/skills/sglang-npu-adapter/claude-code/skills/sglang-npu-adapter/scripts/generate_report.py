#!/usr/bin/env python3
"""
聚合 workspace 下的产物,生成最终中文教程到 output/<ModelName>.md。

用法:
    python generate_report.py --workspace <WORKSPACE_DIR> --model <ModelName> --output <out_path>

兼容两代 output_summary.json:
  - 扁平 schema(prompts/model_analyzer.md / agents/architecture-analyst.md)
  - 嵌套 schema(templates/output_summary.json)
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
        if isinstance(v, dict):
            status = v.get("status", "unknown")
            evidence = v.get("evidence", "")
        else:
            status = str(v)
            evidence = ""
        lines.append(f"| {k} | {status} | {evidence} |")
    return "\n".join(lines)


def fmt_risks(risks):
    if not risks:
        return "(无)"
    lines = []
    for r in risks:
        if isinstance(r, dict):
            level = r.get("level", "")
            desc = r.get("description") or r.get("risk") or json.dumps(r, ensure_ascii=False)
            mitigation = r.get("mitigation")
            prefix = f"[{level}] " if level else ""
            lines.append(f"- {prefix}{desc}" + (f"(缓解: {mitigation})" if mitigation else ""))
        else:
            lines.append(f"- {r}")
    return "\n".join(lines)


def normalize_arch(arch):
    """把嵌套/扁平两种 schema 归一化成扁平字段。"""
    arch = arch or {}
    nested_arch = arch.get("architecture") if isinstance(arch.get("architecture"), dict) else {}
    cfg = arch.get("model_config") or {}
    par = arch.get("parallel_config") or arch.get("parallel_config_suggestion") or {}
    npu = arch.get("npu_compatibility") or {}
    feature = arch.get("feature_compatibility") or {}

    def pick(*keys, default=None):
        for key in keys:
            for src in (arch, nested_arch, cfg):
                if isinstance(src, dict) and src.get(key) is not None:
                    return src[key]
        return default

    return {
        "model_name": arch.get("model_name"),
        "hf_architecture": pick("hf_architecture", "architecture_name")
        or (nested_arch.get("name") if isinstance(nested_arch, dict) else None),
        "architecture": arch.get("architecture") if not isinstance(arch.get("architecture"), dict) else nested_arch.get("type"),
        "model_class": arch.get("model_class") or (nested_arch.get("type") if isinstance(nested_arch, dict) else None),
        "params_b": pick("params_b") or ((arch.get("resource") or {}).get("params_b")),
        "hidden_size": pick("hidden_size"),
        "num_layers": pick("num_layers", "num_hidden_layers"),
        "num_attention_heads": pick("num_attention_heads"),
        "num_kv_heads": pick("num_kv_heads", "num_key_value_heads"),
        "intermediate_size": pick("intermediate_size"),
        "vocab_size": pick("vocab_size"),
        "max_position": pick("max_position", "max_position_embeddings"),
        "rope_theta": pick("rope_theta"),
        "torch_dtype": pick("torch_dtype", "dtype"),
        "reference_model": arch.get("reference_model") or (nested_arch.get("reference_model") if isinstance(nested_arch, dict) else None),
        "reference_file": arch.get("reference_file") or (nested_arch.get("reference_file") if isinstance(nested_arch, dict) else None),
        "similarity": arch.get("similarity") or (nested_arch.get("similarity") if isinstance(nested_arch, dict) else None),
        "diff_summary": arch.get("diff_summary"),
        "adapter_strategy": arch.get("adapter_strategy"),
        "requires_new_adapter": arch.get("requires_new_adapter"),
        "parallel": par,
        "feature_compatibility": feature,
        "npu_compatibility": npu,
        "launch_command_template": arch.get("launch_command_template"),
        "risks": arch.get("risks") or [],
        # probe_model_arch.py 的精简产物
        "model_type": arch.get("model_type"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ws = args.workspace
    state = load_json(os.path.join(ws, "adapter_state.json"), {})
    raw_arch = load_json(os.path.join(ws, "output", "output_summary.json"), {})
    arch = normalize_arch(raw_arch)
    test = load_json(os.path.join(ws, "output", "test_result.json"), {})
    triton = load_json(os.path.join(ws, "output", "triton_replace_result.json"), {})
    gsm8k = load_json(os.path.join(ws, "output", "gsm8k_accuracy_result.json"), {})
    rca = load_json(os.path.join(ws, "output", "root_cause.json"), {})

    sections = []

    # 标题
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(f"# {args.model} NPU 适配报告\n\n> 生成时间: {now}\n> 工作目录: {ws}\n")

    # 1. 模型概况
    body = f"""- 模型名称: {arch.get('model_name') or args.model}
- HF 架构: `{arch.get('hf_architecture') or arch.get('model_type') or 'unknown'}`
- 模型类型: {arch.get('model_class') or 'unknown'}
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
    body = f"""- 参考实现: `{arch.get('reference_file') or 'unknown'}` ({arch.get('reference_model') or '?'})
- 相似度: **{arch.get('similarity') or 'unknown'}**
- 差异摘要: {arch.get('diff_summary') or '(无)'}
- 适配策略: **{arch.get('adapter_strategy') or 'unknown'}**
- 是否需要新适配器: {arch.get('requires_new_adapter', '?')}
"""
    sections.append(section("2. 适配策略", body))

    # 3. 并行配置建议
    pcs = arch.get("parallel") or {}
    if pcs.get("derivation_reasoning") and not pcs.get("rationale"):
        pcs["rationale"] = pcs["derivation_reasoning"]
    body = f"""- tp_size: {pcs.get('tp_size', '?')}
- dp_size: {pcs.get('dp_size', '?')}
- ep_size: {pcs.get('ep_size', '?')}
- pp_size: {pcs.get('pp_size', '?')}
- 理由: {pcs.get('rationale') or '(无)'}
"""
    sections.append(section("3. 并行配置建议", body))

    # 4. 特性兼容性(架构分析师预判)
    body = fmt_feature_matrix(arch.get("feature_compatibility") or {})
    npu = arch.get("npu_compatibility") or {}
    if npu:
        body += f"\n\nNPU 兼容性: `{npu.get('compatible', 'unknown')}`(attention_backend={npu.get('attention_backend', '?')}, moe_backend={npu.get('moe_backend', '?')})"
    sections.append(section("4. 特性兼容性(架构分析预判)", body))

    # 5. 测试结果(test-validator 实测)
    body = f"**总体状态:** {test.get('status', test.get('overall_result', 'unknown'))}\n\n"
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
            if isinstance(r, dict):
                body += f"- {r.get('feature', '?')}: {r.get('phenomenon', '?')}\n"
            else:
                body += f"- {r}\n"
    sections.append(section("5. 测试结果", body))

    # 5.1 Triton 算子替换
    body = f"**dry_run:** {triton.get('dry_run', '?')}\n\n"
    results = triton.get("replace_results") or []
    if results:
        body += "| 文件 | 动作 | 来源 | 备份 |\n|------|------|------|------|\n"
        for r in results:
            body += (
                f"| {r.get('rel_dir', '')}/{r.get('file', '?')} "
                f"| {r.get('action', '?')} "
                f"| {r.get('source', '')} "
                f"| {r.get('backup', '')} |\n"
            )
    else:
        body += "(无替换记录)"
    sections.append(section("5.1 Triton 算子替换", body))

    # 5.2 GSM8K 精度
    metrics = gsm8k.get("metrics") or {}
    body = f"""- status: **{gsm8k.get('status', 'missing')}**
- passed: **{gsm8k.get('passed', False)}**
- accuracy: {metrics.get('accuracy', '?')} (threshold {gsm8k.get('min_accuracy', '?')})
- invalid: {metrics.get('invalid', '?')} (threshold {gsm8k.get('max_invalid', '?')})
- reason: {gsm8k.get('reason') or '(无)'}
"""
    sections.append(section("5.2 GSM8K 精度", body))

    # 6. 推荐启动命令
    cmd = arch.get("launch_command_template") or "(无)"
    sections.append(section("6. 推荐启动命令", f"```bash\n{cmd}\n```"))

    # 7. 风险与遗留
    sections.append(section("7. 风险与遗留", fmt_risks(arch.get("risks"))))

    # 8. 验证状态
    v = state.get("validation", {})
    body = f"""- Dummy 权重验证: {'✓ passed' if v.get('dummy_passed') else '✗ not passed'}
- 真实权重验证: {'✓ passed' if v.get('real_weight_passed') else '✗ not passed'}
- GSM8K 精度: {'✓ passed' if v.get('gsm8k_passed') else '✗ not passed'}
- precision_suspect: {state.get('precision_suspect', False)}
"""
    if state.get("precision_suspect"):
        replacements = rca.get("replacements") or []
        body += f"- 精度 RCA status: {rca.get('status', 'missing')}\n"
        body += f"- 候选替换/根因数: {len(replacements)}\n"
        if rca.get("first_bad_layer") is not None:
            body += f"- 首坏层: {rca.get('first_bad_layer')}\n"
        if rca.get("first_bad_module"):
            body += f"- 首坏模块: {rca.get('first_bad_module')}\n"
        if rca.get("drift_after_fix"):
            body += f"- drift_after_fix: {json.dumps(rca.get('drift_after_fix'), ensure_ascii=False)}\n"
    sections.append(section("8. 验证状态", body))

    # 9. 文件清单
    body = f"""- 架构分析: `output/output_summary.json`, `output/analysis_report.md`
- Triton 替换: `output/triton_replace_result.json`
- GSM8K 精度: `output/gsm8k_accuracy_result.json`
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
