#!/usr/bin/env python3
"""
报告生成脚本
汇总分析报告、测试结果、提交信息，生成最终模型适配报告
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def read_json_file(filepath: str) -> Optional[Dict]:
    """读取JSON文件，不存在返回 None"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def read_markdown_file(filepath: str) -> str:
    """读取 Markdown 文件，不存在返回空串"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def _git_repo_root(start_dir: str) -> Optional[str]:
    """从 start_dir 向上查找 git 仓库根目录"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start_dir,
            stderr=subprocess.DEVNULL,
        ).decode().strip() or None
    except Exception:
        return None


def _git(cwd: str, *args) -> str:
    """在指定 cwd 下运行 git，返回 stdout（出错返回空串）"""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def get_git_info(repo_root: str) -> Dict[str, str]:
    """获取当前 HEAD 的 Git 信息（路径与命令均锚定到 repo_root）"""
    info = {
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit_hash": _git(repo_root, "rev-parse", "HEAD")[:8],
        "commit_message": _git(repo_root, "log", "-1", "--pretty=%s"),
        "commit_author": _git(repo_root, "log", "-1", "--pretty=%an"),
        "commit_date": _git(repo_root, "log", "-1", "--pretty=%ci"),
        "status": "",
    }
    status_output = _git(repo_root, "status", "--short")
    info["status"] = "clean" if not status_output else "dirty"
    return info


# 工作区临时产物（agent 自己的输入/输出/日志/快照），不算"代码改动"，从 diff 中过滤
_WORKSPACE_NOISE_PREFIX = ".trae/workspace/"


def _is_workspace_noise(path: str) -> bool:
    return path.startswith(_WORKSPACE_NOISE_PREFIX)


def get_changed_files(repo_root: str, base_commit: Optional[str]) -> Dict[str, Any]:
    """
    获取本任务涉及的修改文件。
    优先用 adapter_state.base_commit..HEAD 计算（任务起点之后的所有改动），
    否则退回 HEAD~1..HEAD 并标记为不可靠。所有 git 命令均运行在 repo_root，
    确保路径以仓库根为基准（避免被调用方 cwd 影响）。
    """
    result: Dict[str, Any] = {"files": [], "diff_range": "", "reliable": False}

    if base_commit:
        # 校验 base_commit 在当前仓库是否存在
        verified = subprocess.run(
            ["git", "rev-parse", "--verify", base_commit],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0

        if not verified:
            result["diff_range"] = f"(base_commit {base_commit[:8]} 在当前仓库找不到)"
        else:
            committed = _git(repo_root, "diff", "--name-only", f"{base_commit}..HEAD").splitlines()
            worktree = _git(repo_root, "diff", "--name-only", "HEAD").splitlines()
            staged = _git(repo_root, "diff", "--name-only", "--cached").splitlines()
            untracked = _git(repo_root, "ls-files", "--others", "--exclude-standard").splitlines()

            merged: List[str] = []
            filtered_count = 0
            for group in (committed, worktree, staged, untracked):
                for f in group:
                    if not f or f in merged:
                        continue
                    if _is_workspace_noise(f):
                        filtered_count += 1
                        continue
                    merged.append(f)
            merged.sort()
            result["files"] = merged
            result["filtered_workspace_count"] = filtered_count
            result["diff_range"] = f"{base_commit[:8]}..HEAD (含未提交改动)"
            result["reliable"] = True
            return result

    # 回退路径：base_commit 缺失或不可用
    output = _git(repo_root, "diff", "--name-only", "HEAD~1")
    if output:
        all_files = output.split("\n")
        result["files"] = [f for f in all_files if not _is_workspace_noise(f)]
        result["filtered_workspace_count"] = len(all_files) - len(result["files"])
    result["diff_range"] = "HEAD~1..HEAD（无 base_commit，可能包含与本任务无关的改动）"
    return result


def fmt_or_na(value):
    """格式化字段，None / 空串显示 N/A"""
    if value is None or value == "" or value == []:
        return "N/A"
    return value


def generate_final_report(
    workspace_dir: str,
    model_name: str,
    output_file: str
) -> str:
    """生成最终适配报告"""

    summary = read_json_file(f"{workspace_dir}/output/output_summary.json") or {}
    test_result = read_json_file(f"{workspace_dir}/output/test_result.json") or {}
    adapter_state = read_json_file(f"{workspace_dir}/adapter_state.json") or {}

    # 嵌套字段抽取
    arch = summary.get("architecture", {}) or {}
    model_cfg = summary.get("model_config", {}) or {}
    parallel = summary.get("parallel_config", {}) or {}
    resource = summary.get("resource", {}) or {}
    npu_compat = summary.get("npu_compatibility", {}) or {}
    risks = summary.get("risks", []) or []
    recommendations = summary.get("recommendations", {}) or {}
    initial_cfg = recommendations.get("initial_test_config", {}) or {}

    test_summary = test_result.get("summary", {}) or {}
    issues_fixed = test_result.get("issues_fixed", []) or []
    validation_results = test_result.get("validation_results", {}) or {}
    stage_b = validation_results.get("stage_b_real_weight", {}) or {}

    repo_root = _git_repo_root(workspace_dir) or os.getcwd()
    git_info = get_git_info(repo_root)
    base_commit = adapter_state.get("base_commit")
    diff_info = get_changed_files(repo_root, base_commit)

    lines: List[str] = []
    lines.append(f"# {model_name} 模型适配报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ===== 1. 基本信息 =====
    lines.append("\n---\n")
    lines.append("## 1. 基本信息")
    arch_type = arch.get("type", "")
    arch_subtype = arch.get("subtype", "")
    type_display = arch_type
    if arch_subtype:
        type_display = f"{arch_type} ({arch_subtype})" if arch_type else arch_subtype
    lines.append(f"\n- **模型架构**: {fmt_or_na(arch.get('name'))}")
    lines.append(f"- **架构类型**: {fmt_or_na(type_display)}")
    lines.append(f"- **参考模型**: {fmt_or_na(arch.get('reference_model'))}")
    lines.append(f"- **参考实现**: {fmt_or_na(arch.get('reference_file'))}")
    lines.append(f"- **相似度**: {fmt_or_na(arch.get('similarity'))}")
    lines.append(f"- **NPU 兼容**: {'是' if npu_compat.get('compatible') else '否'}")
    if arch.get("notes"):
        lines.append(f"- **备注**: {arch['notes']}")

    # ===== 2. 配置建议 =====
    lines.append("\n---\n")
    lines.append("## 2. 配置建议")
    lines.append(f"\n- **推荐 TP**: {fmt_or_na(parallel.get('tp_size'))}")
    lines.append(f"- **推荐 EP**: {fmt_or_na(parallel.get('ep_size'))}")
    lines.append(f"- **推荐 PP**: {fmt_or_na(parallel.get('pp_size'))}")
    lines.append(f"- **推荐 DP**: {fmt_or_na(parallel.get('dp_size'))}")
    lines.append(f"- **总设备数**: {fmt_or_na(parallel.get('total_devices_needed'))}")
    # 上下文长度优先用初始测试配置；缺失则用模型最大值
    ctx_recommended = initial_cfg.get("max_length") or model_cfg.get("max_position_embeddings")
    lines.append(f"- **推荐上下文长度**: {fmt_or_na(ctx_recommended)}")
    lines.append(f"- **模型最大上下文**: {fmt_or_na(model_cfg.get('max_position_embeddings'))}")
    lines.append(f"- **权重大小**: {fmt_or_na(resource.get('weight_size_gb'))} GB")
    if resource.get("estimated_params"):
        lines.append(f"- **参数量**: {resource['estimated_params']}")
    if parallel.get("derivation_reasoning"):
        lines.append(f"- **推导依据**: {parallel['derivation_reasoning']}")

    # ===== 3. 测试结果 =====
    lines.append("\n---\n")
    lines.append("## 3. 测试结果")
    overall = test_result.get("overall_status", "N/A")
    overall_emoji = "✓" if overall == "passed" else "✗"
    lines.append(f"\n**总体状态**: {overall_emoji} {overall}")
    lines.append(
        f"\n- **通过用例**: {test_summary.get('passed', 0)}/{test_summary.get('total_tests', 0)}"
    )
    lines.append(f"- **失败用例**: {test_summary.get('failed', 0)}")
    lines.append(f"- **跳过用例**: {test_summary.get('skipped', 0)}")

    # 各阶段详情
    if validation_results:
        lines.append("\n**分阶段结果：**")
        for stage_key, stage_label in [
            ("stage_a_dummy_weight", "Stage A (Dummy 权重)"),
            ("stage_b_real_weight", "Stage B (真实权重)"),
        ]:
            stage = validation_results.get(stage_key, {}) or {}
            if not stage:
                continue
            status = stage.get("status", "N/A")
            emoji = "✓" if status == "passed" else "✗"
            lines.append(f"- **{stage_label}**: {emoji} {status}")
            if stage.get("notes"):
                lines.append(f"  - 备注: {stage['notes']}")

    # 调试中已修复的问题
    if issues_fixed:
        lines.append("\n**调试中修复的问题：**")
        for issue in issues_fixed:
            iter_n = issue.get("iteration", "?")
            etype = issue.get("error_type", "")
            desc = issue.get("description", "")
            fix = issue.get("fix", "")
            lines.append(f"- 第 {iter_n} 次迭代 — `{etype}`: {desc}")
            if fix:
                lines.append(f"  - 修复: {fix}")
            if issue.get("reference"):
                lines.append(f"  - 参考: `{issue['reference']}`")

    # ===== 4. 功能状态矩阵 =====
    lines.append("\n---\n")
    lines.append("## 4. 功能状态矩阵")
    lines.append("\n| 功能 | 状态 | 说明 |")
    lines.append("|------|------|------|")

    verified = npu_compat.get("verified_components") or npu_compat.get("verified_features") or []
    workarounds = npu_compat.get("workarounds_applied") or npu_compat.get("workarounds") or []
    fallback = npu_compat.get("fallback_components") or []

    # ACLGraph
    aclgraph_disabled = any("disable-cuda-graph" in str(w) or "ACLGraph" in str(w) for w in workarounds)
    if aclgraph_disabled:
        lines.append("| ACLGraph | ⚠ 已禁用 | NPU Graph 捕获存在兼容性问题，使用 --disable-cuda-graph 规避 |")
    else:
        lines.append("| ACLGraph | 待验证 | - |")

    # DeepEP
    if "MoE" in arch_type:
        lines.append("| DeepEP | 待验证 | MoE 模型 |")
    else:
        lines.append("| DeepEP | 不适用 | 非 MoE 模型 |")

    # 多模态
    if "VLM" in arch_type:
        verified_vision = any("Vision" in str(v) or "VLM" in str(v) or "multimodal" in str(v).lower()
                              for v in verified)
        status_label = "✓ 已验证" if verified_vision else "待验证"
        notes = "; ".join([str(v) for v in verified if "Vision" in str(v) or "VLM" in str(v) or "multimodal" in str(v).lower()]) or "VLM 模型"
        lines.append(f"| 多模态 | {status_label} | {notes} |")
    else:
        lines.append("| 多模态 | 不适用 | 非 VLM 模型 |")

    # Linear Attention（如果有）
    if any("linear" in str(f).lower() or "GatedDeltaNet" in str(f) for f in fallback):
        lines.append("| Linear Attention | ⚠ Fallback | 走 NPU fallback 路径，性能可能与 CUDA 有差异 |")

    # MTP / DP-Attention 等等占位（数据中没有时不显示）
    # 留给将来扩展

    # ===== 5. 验证矩阵 =====
    lines.append("\n---\n")
    lines.append("## 5. 验证矩阵")
    lines.append("\n| 阶段 | 状态 | 说明 |")
    lines.append("|------|------|------|")

    validation = adapter_state.get("validation", {}) or {}
    dummy_passed = validation.get("dummy_passed", False)
    real_passed = validation.get("real_weight_passed", False)
    dummy_status = "✓ 通过" if dummy_passed else "✗ 未通过"
    real_status = "✓ 通过" if real_passed else "✗ 未通过"
    lines.append(f"| Dummy 验证 | {dummy_status} | 架构/算子加载验证 |")
    lines.append(f"| 真实权重验证 | {real_status} | 功能/精度验证 |")
    if dummy_passed and not real_passed:
        lines.append("\n**注意**: Dummy 验证通过但真实权重验证未通过，请检查权重映射。")

    # ===== 6. 修改文件 =====
    lines.append("\n---\n")
    lines.append("## 6. 修改文件")
    if diff_info.get("diff_range"):
        lines.append(f"\n*Diff 范围*: `{diff_info['diff_range']}`")
        if not diff_info.get("reliable"):
            lines.append("\n> ⚠ 该列表可能包含与本任务无关的改动；建议手工筛选。")
        if diff_info.get("filtered_workspace_count"):
            lines.append(
                f"\n*已过滤 {diff_info['filtered_workspace_count']} 个 `.trae/workspace/` 下的工作区产物*"
            )
    if diff_info.get("files"):
        lines.append("\n```")
        lines.extend(diff_info["files"])
        lines.append("```")
    else:
        lines.append("\n*无修改文件记录*")

    # ===== 7. 提交信息 =====
    lines.append("\n---\n")
    lines.append("## 7. 提交信息")
    lines.append(f"\n- **分支**: {fmt_or_na(git_info.get('branch'))}")
    if base_commit:
        lines.append(f"- **任务起点 commit**: `{base_commit[:8]}`")
    lines.append(f"- **当前提交哈希**: {fmt_or_na(git_info.get('commit_hash'))}")
    lines.append(f"- **当前提交信息**: {fmt_or_na(git_info.get('commit_message'))}")
    lines.append(f"- **当前提交者**: {fmt_or_na(git_info.get('commit_author'))}")
    lines.append(f"- **当前提交时间**: {fmt_or_na(git_info.get('commit_date'))}")
    lines.append(f"- **工作区状态**: {fmt_or_na(git_info.get('status'))}")

    # ===== 8. 关键发现 =====
    lines.append("\n---\n")
    lines.append("## 8. 关键发现")
    findings_added = False
    if arch.get("notes"):
        lines.append(f"\n- **架构识别**: {arch['notes']}")
        findings_added = True
    if parallel.get("derivation_reasoning"):
        lines.append(f"- **并行配置推导**: {parallel['derivation_reasoning']}")
        findings_added = True
    layer_pat = (summary.get("layer_analysis") or {}).get("pattern")
    if layer_pat:
        lines.append(f"- **层结构**: {layer_pat}")
        findings_added = True
    if verified:
        lines.append(f"- **已验证组件**: {', '.join(str(v) for v in verified)}")
        findings_added = True
    if workarounds:
        lines.append(f"- **应用的规避方案**: {', '.join(str(w) for w in workarounds)}")
        findings_added = True
    if fallback:
        lines.append(f"- **走 fallback 的组件**: {', '.join(str(f) for f in fallback)}")
        findings_added = True

    if risks:
        lines.append("\n**风险与缓解：**")
        for r in risks:
            level = r.get("level", "?")
            cat = r.get("category", "")
            desc = r.get("description", "")
            mit = r.get("mitigation", "")
            lines.append(f"- [{level}] **{cat}** — {desc}")
            if mit:
                lines.append(f"  - 缓解: {mit}")
        findings_added = True

    if not findings_added:
        lines.append("\n*无关键发现记录*")

    # ===== 9. 运行命令 =====
    lines.append("\n---\n")
    lines.append("## 9. 运行命令")
    # 优先使用真实权重验证阶段实际跑通的启动命令
    real_launch = stage_b.get("launch_command")
    if real_launch:
        lines.append("\n*以下为真实权重验证阶段实际跑通的启动命令：*")
        lines.append("\n```bash")
        lines.append("export PYTHONPATH=${PWD}/python:$PYTHONPATH")
        lines.append(real_launch)
        lines.append("```")
    else:
        # 兜底：根据推导出的并行配置组装一条
        tp = parallel.get("tp_size", 1)
        ctx = ctx_recommended or 4096
        lines.append("\n```bash")
        lines.append("export PYTHONPATH=${PWD}/python:$PYTHONPATH")
        lines.append(f"python -m sglang.launch_server \\")
        lines.append(f"    --model-path /path/to/{model_name} \\")
        lines.append(f"    --port 8000 \\")
        lines.append(f"    --tp {tp} \\")
        lines.append(f"    --context-length {ctx} \\")
        lines.append(f"    --device npu \\")
        lines.append(f"    --attention-backend ascend")
        lines.append("```")

    lines.append("\n---")
    lines.append("\n*报告由 SGLang NPU 适配技能套件自动生成*")

    content = "\n".join(lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    return content


def main():
    import argparse

    parser = argparse.ArgumentParser(description="报告生成脚本")
    parser.add_argument("--workspace", "-w", required=True, help="工作目录路径")
    parser.add_argument("--model", "-m", required=True, help="模型名称")
    parser.add_argument("--output", "-o", required=True, help="输出报告文件路径")
    args = parser.parse_args()

    report = generate_final_report(args.workspace, args.model, args.output)
    print(f"报告已生成: {args.output}")
    print("\n" + "=" * 60)
    print(report[:500] + "..." if len(report) > 500 else report)


if __name__ == "__main__":
    main()
