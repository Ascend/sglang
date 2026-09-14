# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.parallel_hook import (
    handle_context_parallel_kernel_compatibility,
)
from sglang.srt.hardware_backend.npu.kda_kernel_capabilities import (
    check_kda_fla_cp_kernel_compatibility,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _compatible_module():
    return SimpleNamespace(
        chunk_gated_delta_rule_fwd_affine_npu=lambda: None,
        merge_kda_cp_affine_states=lambda: None,
    )


def test_kda_fla_cp_kernel_capability_requires_affine_operators():
    with patch("importlib.import_module", return_value=_compatible_module()):
        assert check_kda_fla_cp_kernel_compatibility() == (True, "compatible")

    missing_operator = _compatible_module()
    del missing_operator.merge_kda_cp_affine_states
    with patch("importlib.import_module", return_value=missing_operator):
        compatible, reason = check_kda_fla_cp_kernel_compatibility()
    assert not compatible
    assert "missing KDA FLA PCP operators" in reason


def test_incompatible_kimi_k3_kernel_disables_prefill_cp(monkeypatch):
    args = SimpleNamespace(
        _resolved_overrides=[],
        attn_cp_size=2,
        cp_strategy="zigzag",
        device="npu",
        enable_dsa_prefill_context_parallel=False,
        model_path="/model",
        enable_prefill_context_parallel=True,
        enable_prefill_cp=True,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.parallel_hook.model_config_of",
        lambda _server_args: model_config,
    )
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.npu.kda_kernel_capabilities."
        "check_kda_fla_cp_kernel_compatibility",
        lambda: (False, "missing affine operator"),
    )

    handle_context_parallel_kernel_compatibility(args)

    cfg = resolving_view(args)
    assert cfg.attn_cp_size == 1
    assert cfg.cp_strategy is None
    assert not cfg.enable_dsa_prefill_context_parallel
    assert not cfg.enable_prefill_context_parallel
    assert not cfg.enable_prefill_cp


def test_compatible_kimi_k3_kernel_keeps_prefill_cp(monkeypatch):
    args = SimpleNamespace(
        _resolved_overrides=[],
        attn_cp_size=2,
        cp_strategy="zigzag",
        device="npu",
        enable_dsa_prefill_context_parallel=False,
        model_path="/model",
        enable_prefill_context_parallel=True,
        enable_prefill_cp=True,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
    )
    monkeypatch.setattr(
        "sglang.srt.arg_groups.parallel_hook.model_config_of",
        lambda _server_args: model_config,
    )
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.npu.kda_kernel_capabilities."
        "check_kda_fla_cp_kernel_compatibility",
        lambda: (True, "compatible"),
    )

    handle_context_parallel_kernel_compatibility(args)

    cfg = resolving_view(args)
    assert cfg.attn_cp_size == 2
    assert cfg.cp_strategy == "zigzag"
    assert cfg.enable_prefill_cp
