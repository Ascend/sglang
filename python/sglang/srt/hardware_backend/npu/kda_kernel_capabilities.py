# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

_KDA_CHUNK_MODULE = "sgl_kernel_npu.fla.kda_chunk_delta_h"
_KDA_FLA_CP_OPERATORS = (
    "chunk_gated_delta_rule_fwd_affine_npu",
    "merge_kda_cp_affine_states",
)
_KDA_FLA_CP_STATE_KERNEL_ARGUMENTS = (
    "initial_state_key_value_layout",
    "block_value",
)


def _load_kda_chunk_module() -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(_KDA_CHUNK_MODULE), None
    except Exception as exc:
        return None, f"cannot import {_KDA_CHUNK_MODULE}: {type(exc).__name__}: {exc}"


def check_kda_fla_cp_kernel_compatibility() -> tuple[bool, str]:
    """Check the complete external-kernel contract required by Kimi-K3 PCP."""
    module, error = _load_kda_chunk_module()
    if module is None:
        return False, error or f"cannot import {_KDA_CHUNK_MODULE}"

    missing = [
        name
        for name in _KDA_FLA_CP_OPERATORS
        if not callable(getattr(module, name, None))
    ]
    if missing:
        return False, "missing KDA FLA PCP operators: " + ", ".join(missing)

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
