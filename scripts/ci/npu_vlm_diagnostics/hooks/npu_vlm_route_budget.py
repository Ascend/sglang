"""Isolated, opt-in CANN routing experiment; never writes an installed file.

This exact-library diagnostic changes the 9392 branch's internal MatMul routing
budget from 192 to 96 MiB. It does not change physical cache, available cores,
weights or precision. Both arms require the original 24/48 resource limits.
"""

import atexit
import ctypes
import functools
import hashlib
import json
import os
import platform
from collections import Counter
from pathlib import Path

LIBRARY = Path(
    "/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/"
    "op_host/lib/linux/aarch64/libophost_comm_legacy.so"
)
LIBRARY_SHA256 = "8cd4d9b43b8f004643faf297e7fcfe7995b8cfe34a77014b14a7caecb7165087"
SYMBOL_OFFSET = 0x4151D0
INSTRUCTION_OFFSET = 0x413998
ORIGINAL = bytes.fromhex("00188052")  # mov w0, #192
REPLACEMENT = bytes.fromhex("000c8052")  # mov w0, #96
_APPLIED = {}
_HANDLES = []


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _rewrite(address, expected, replacement):
    _require(ctypes.string_at(address, 4) == expected, "Unexpected live instruction")
    size = os.sysconf("SC_PAGE_SIZE")
    page = address // size * size
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int
    gcc = ctypes.CDLL("libgcc_s.so.1")
    clear = gcc.__clear_cache
    clear.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    clear.restype = None
    _require(libc.mprotect(page, size, 3) == 0, "mprotect RW failed")
    try:
        ctypes.memmove(address, replacement, 4)
        clear(address, address + 4)
    finally:
        _require(libc.mprotect(page, size, 5) == 0, "mprotect RX failed")
    _require(ctypes.string_at(address, 4) == replacement, "Instruction write failed")


def patch_library(mode):
    """Separate from NPU checks so the exact native branch can be CPU-audited."""
    _require(mode in ("observe", "96"), f"Unsupported routing mode: {mode}")
    _require(
        platform.machine() == "aarch64", "Only the audited AArch64 binary is supported"
    )
    _require(
        hashlib.sha256(LIBRARY.read_bytes()).hexdigest() == LIBRARY_SHA256,
        "CANN binary differs from the audited version",
    )
    lib = ctypes.CDLL(str(LIBRARY))
    _HANDLES.append(lib)
    symbol = ctypes.cast(lib.LegacyMmCheckHitV3Shape, ctypes.c_void_p).value
    base = symbol - SYMBOL_OFFSET
    address = base + INSTRUCTION_OFFSET
    _require(ctypes.string_at(address, 4) == ORIGINAL, "Budget instruction mismatch")
    _require(
        ctypes.string_at(address + 4, 8) == bytes.fromhex("fd7bc3a8c0035fd6"),
        "Budget function epilogue mismatch",
    )
    if mode == "96":
        _rewrite(address, ORIGINAL, REPLACEMENT)
        atexit.register(_rewrite, address, REPLACEMENT, ORIGINAL)
    _require(
        hashlib.sha256(LIBRARY.read_bytes()).hexdigest() == LIBRARY_SHA256,
        "Installed file changed",
    )
    return {
        "mode": mode,
        "library_sha256": LIBRARY_SHA256,
        "instruction_offset": hex(INSTRUCTION_OFFSET),
        "before_instruction": ORIGINAL.hex(),
        "after_instruction": ctypes.string_at(address, 4).hex(),
        "budget_before_mib": 192,
        "budget_after_mib": 96 if mode == "96" else 192,
        "installed_file_unchanged": True,
    }


def apply(mode, torch, torch_npu):
    _require(
        not any(
            os.environ.get(name)
            for name in ("VLM_DEVICE_LIMIT_CUBE", "VLM_DEVICE_LIMIT_VECTOR")
        ),
        "Routing-only experiment cannot include core-limit settings",
    )
    device = torch.npu.current_device()
    before = dict(torch_npu.npu.get_device_limit(device))
    props = torch.npu.get_device_properties(device)
    expected = {"cube_core_num": 24, "vector_core_num": 48}
    _require(props.name == "Ascend910_9392", f"Expected CI 9392, observed {props}")
    _require(
        all(before[k] == v for k, v in expected.items()),
        f"Expected untouched 24/48 core resources, observed {before}",
    )
    key = (os.getpid(), device)
    if key in _APPLIED:
        _require(_APPLIED[key]["mode"] == mode, "Conflicting routing interventions")
        return _APPLIED[key]
    record = patch_library(mode)
    after = dict(torch_npu.npu.get_device_limit(device))
    _require(before == after, "Routing intervention changed device limits")
    record.update(
        {
            "pid": os.getpid(),
            "device": device,
            "properties": str(props),
            "core_limits_before": before,
            "core_limits_after": after,
        }
    )
    _APPLIED[key] = record
    return record


def install(cls):
    mode = os.environ.get("VLM_CANN_ROUTE_BUDGET")
    if mode is None:
        return
    _require(mode in ("observe", "96"), f"Unsupported routing mode: {mode}")
    original = cls.forward
    done = False

    @functools.wraps(original)
    def forward(owner, *args, **kwargs):
        nonlocal done
        if not done:
            import torch
            import torch_npu

            before = tuple((id(p), str(p.dtype)) for p in owner.parameters())
            record = dict(apply(mode, torch, torch_npu))
            after = tuple((id(p), str(p.dtype)) for p in owner.parameters())
            _require(
                before == after,
                "Routing intervention changed parameter identities/dtypes",
            )
            _require(
                all(dtype == "torch.bfloat16" for _, dtype in before),
                "Expected original BF16 model parameters",
            )
            record.update(
                {
                    "parameter_dtypes": dict(Counter(dtype for _, dtype in before)),
                    "parameter_identities_and_dtypes_unchanged": True,
                    "applied_before_first_forward_including_warmup": True,
                }
            )
            print("NPU_VLM_ROUTE_BUDGET " + json.dumps(record), flush=True)
            done = True
        return original(owner, *args, **kwargs)

    cls.forward = forward
