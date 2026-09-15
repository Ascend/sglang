# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

_KDA_FLA_CP_OPERATOR_CONTRACTS = (
    (
        "sgl_kernel_npu.fla.kda_chunk_delta_h",
        (
            "chunk_gated_delta_rule_fwd_affine_npu",
            "merge_kda_cp_affine_states",
        ),
    ),
    (
        "sgl_kernel_npu.fla.kda_scaled_dot_kkt",
        ("chunk_kda_scaled_dot_kkt_fwd_npu",),
    ),
)
_KDA_CHUNK_MODULE = _KDA_FLA_CP_OPERATOR_CONTRACTS[0][0]
_KDA_FLA_CP_STATE_KERNEL_ARGUMENTS = (
    "initial_state_key_value_layout",
    "block_value",
)


def _load_module(module_name: str) -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, f"cannot import {module_name}: {type(exc).__name__}: {exc}"


def check_kda_fla_cp_kernel_compatibility() -> tuple[bool, str]:
    """Check the complete external-kernel contract required by Kimi-K3 PCP."""
    modules = {}
    for module_name, operator_names in _KDA_FLA_CP_OPERATOR_CONTRACTS:
        module, error = _load_module(module_name)
        if module is None:
            return False, error or f"cannot import {module_name}"
        missing = [
            name for name in operator_names if not callable(getattr(module, name, None))
        ]
        if missing:
            return False, "missing KDA FLA PCP operators: " + ", ".join(missing)
        modules[module_name] = module

    module = modules[_KDA_CHUNK_MODULE]
    state_kernel = getattr(module, "chunk_gated_delta_rule_fwd_h_npu", None)
    if not callable(state_kernel):
        return False, "missing KDA FLA PCP state kernel"
    state_kernel_parameters = inspect.signature(state_kernel).parameters
    missing_arguments = [
        name
        for name in _KDA_FLA_CP_STATE_KERNEL_ARGUMENTS
        if name not in state_kernel_parameters
    ]
    if missing_arguments:
        return False, "KDA FLA PCP state kernel is missing arguments: " + ", ".join(
            missing_arguments
        )
    return True, "compatible"
