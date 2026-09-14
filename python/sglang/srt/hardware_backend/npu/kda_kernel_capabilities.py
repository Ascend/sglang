# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from types import ModuleType

_KDA_CHUNK_MODULE = "sgl_kernel_npu.fla.kda_chunk_delta_h"
_KDA_FLA_CP_OPERATORS = (
    "chunk_gated_delta_rule_fwd_affine_npu",
    "merge_kda_cp_affine_states",
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
    return True, "compatible"
