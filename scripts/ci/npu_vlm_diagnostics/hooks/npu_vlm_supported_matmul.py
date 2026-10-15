"""Compare documented torch_npu APIs without modifying CANN binaries.

Operator alternatives affect only bias-free Qwen decoder gate/up prefill
projections. Decode, lm_head, other models, and available NPU cores are retained.
The options are diagnostic candidates, not accepted fixes until CI validates them.
"""

import copy
import functools
import json
import os
import types

MODES = ("default", "gmm", "nz", "addmm", "deterministic", "high_precision")


def configure(mode):
    import torch
    import torch_npu

    if mode not in MODES:
        raise ValueError(f"Unsupported candidate: {mode}")
    if any(os.environ.get(name) for name in (
        "VLM_CANN_ROUTE_BUDGET", "VLM_DEVICE_LIMIT_CUBE", "VLM_DEVICE_LIMIT_VECTOR"
    )):
        raise RuntimeError("Binary routing or core-limit experiment contaminates this run")
    limits = torch_npu.npu.get_device_limit(torch.npu.current_device())
    if os.environ.get("VLM_REQUIRE_24_CORES") == "1" and limits != {
        "cube_core_num": 24, "vector_core_num": 48
    }:
        raise RuntimeError(f"Expected default 24/48 cores: {limits}")
    if mode == "deterministic":
        torch.use_deterministic_algorithms(True)
    elif mode == "high_precision":
        torch.npu.set_option({
            "ACL_OP_SELECT_IMPL_MODE": "high_precision",
            "ACL_OPTYPELIST_FOR_IMPLMODE": "MatMul:MatMulV2:MatMulV3",
        })
    return {"mode": mode, "limits": limits,
            "properties": str(torch.npu.get_device_properties(torch.npu.current_device())),
            "pid": os.getpid(), "binary_patch": False}


def compute(x, weight, mode):
    import torch
    import torch_npu

    if mode == "gmm":
        return torch_npu.npu_grouped_matmul(
            [x], [weight.t()], split_item=0, group_type=-1
        )[0]
    if mode == "nz":
        return torch.nn.functional.linear(x, torch_npu.npu_format_cast(weight, 29))
    if mode == "addmm":
        return torch.addmm(torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype), x, weight.t())
    return torch.nn.functional.linear(x, weight)


def install(cls):
    mode = os.environ.get("VLM_SUPPORTED_MATMUL")
    if mode is None:
        return
    if mode not in MODES:
        raise ValueError(mode)
    original = cls.forward

    @functools.wraps(original)
    def forward(owner, *args, **kwargs):
        if not getattr(owner, "_vlm_supported_matmul_installed", False):
            import torch

            before = [(id(p), str(p.dtype)) for p in owner.parameters()]
            record = configure(mode)
            selected = []
            for name, layer in owner.named_modules():
                if not name.startswith("model.layers.") or not name.endswith(".mlp.gate_up_proj"):
                    continue
                if layer.bias is not None or layer.weight.dtype != torch.bfloat16:
                    raise RuntimeError(f"Unexpected gate/up parameter: {name}")
                method = copy.copy(layer.quant_method)
                original_apply = method.apply

                def apply(method_self, actual_layer, x, bias=None, original_apply=original_apply):
                    if mode in ("gmm", "nz", "addmm") and x.ndim == 2 and x.shape[0] > 1:
                        if bias is not None or x.dtype != torch.bfloat16:
                            raise RuntimeError("Expected bias-free BF16 prefill")
                        return compute(x, actual_layer.weight, mode)
                    return original_apply(actual_layer, x, bias)

                method.apply = types.MethodType(apply, method)
                layer.quant_method = method
                selected.append(name)
            if len(selected) != 36:
                raise RuntimeError(f"Expected 36 Qwen decoder gate/up layers, found {selected}")
            if before != [(id(p), str(p.dtype)) for p in owner.parameters()]:
                raise RuntimeError("Candidate altered parameter identities or dtype")
            record.update(layers=selected, parameter_dtypes=sorted({d for _, d in before}),
                          parameter_identities_and_dtypes_unchanged=True)
            print("NPU_VLM_SUPPORTED_MATMUL " + json.dumps(record), flush=True)
            owner._vlm_supported_matmul_installed = True
        return original(owner, *args, **kwargs)

    cls.forward = forward
