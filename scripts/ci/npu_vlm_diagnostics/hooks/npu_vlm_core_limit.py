"""Opt-in causal intervention on the Qwen worker's available NPU cores.

Unlike the observational diagnostics, this experiment changes process-local
kernel resource limits. It preserves tensors, model dtype and sampling settings.
With neither environment variable set, install() leaves forward unchanged and
does not import torch. The limit is applied before the first forward, including
warmup forwards that do not yet have a diagnostic request ID.
"""

import functools
import json
import os
from collections import Counter

_APPLIED = {}
_ENV_NAMES = ("VLM_DEVICE_LIMIT_CUBE", "VLM_DEVICE_LIMIT_VECTOR")


def _positive_limit(value, name):
    if value is None:
        return -1
    try:
        limit = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error
    if limit <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return limit


def _parameter_dtypes(owner):
    return tuple((id(value), str(value.dtype)) for value in owner.parameters())


def install(cls):
    """Wrap this model only when the caller explicitly requests core limits."""
    configured = tuple(os.environ.get(name) for name in _ENV_NAMES)
    if configured == (None, None):
        return
    original = cls.forward
    if getattr(original, "_vlm_device_limit", False):
        return
    runtime = None

    @functools.wraps(original)
    def forward(owner, *args, **kwargs):
        nonlocal runtime
        # Import and API failures intentionally propagate: an unapplied
        # intervention must never look like a successful control experiment.
        if runtime is None:
            import torch
            import torch_npu

            runtime = (torch, torch_npu)
        else:
            torch, torch_npu = runtime

        requested = tuple(
            _positive_limit(value, name) for value, name in zip(configured, _ENV_NAMES)
        )
        pid = os.getpid()
        device = torch.npu.current_device()
        key = (pid, device)
        previous = _APPLIED.get(key)
        if previous is not None and previous != requested:
            raise RuntimeError(
                f"Conflicting VLM core limits in pid={pid}, device={device}: "
                f"already applied {previous}, requested {requested}"
            )
        if previous is None:
            before_limits = torch_npu.npu.get_device_limit(device)
            before_dtypes = _parameter_dtypes(owner)
            before_default_dtype = str(torch.get_default_dtype())
            torch_npu.npu.set_device_limit(
                device, cube_num=requested[0], vector_num=requested[1]
            )
            actual = torch_npu.npu.get_device_limit(device)
            for field, limit in zip(("cube_core_num", "vector_core_num"), requested):
                expected = before_limits[field] if limit == -1 else limit
                if actual[field] != expected:
                    raise RuntimeError(
                        f"VLM core limit was not applied on device={device}: "
                        f"{field} expected {expected}, observed {actual[field]}"
                    )
            after_dtypes = _parameter_dtypes(owner)
            after_default_dtype = str(torch.get_default_dtype())
            if (
                before_dtypes != after_dtypes
                or before_default_dtype != after_default_dtype
            ):
                raise RuntimeError(
                    "VLM core limit intervention changed model/default dtype"
                )
            print(
                "NPU_VLM_CORE_LIMIT "
                + json.dumps(
                    {
                        "intervention": "process_device_core_limit",
                        "pid": pid,
                        "device": device,
                        "requested": {
                            "cube_num": requested[0],
                            "vector_num": requested[1],
                        },
                        "before": before_limits,
                        "actual": actual,
                        "parameter_dtypes_before": dict(
                            Counter(dtype for _, dtype in before_dtypes)
                        ),
                        "parameter_dtypes_after": dict(
                            Counter(dtype for _, dtype in after_dtypes)
                        ),
                        "default_dtype_before": before_default_dtype,
                        "default_dtype_after": after_default_dtype,
                        "dtype_unchanged": True,
                        "applied_before_first_forward_including_warmup": True,
                    }
                ),
                flush=True,
            )
            _APPLIED[key] = requested
        return original(owner, *args, **kwargs)

    forward._vlm_device_limit = True
    cls.forward = forward
