"""Read-only Qwen2.5-VL observations; imported without loading torch/SGLang."""

import contextvars
import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import itertools
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

_RID = contextvars.ContextVar("vlm_diagnostic_rid", default=None)
_SEQUENCE = itertools.count()
_MODULES = {
    "sglang.srt.multimodal.processors.base_processor",
    "sglang.srt.multimodal.processors.qwen_vl",
    "sglang.srt.hardware_backend.npu.modules.qwen_vl_processor",
}
_FIELDS = (
    "input_ids",
    "padded_input_ids",
    "mm_items",
    "offsets",
    "feature",
    "precomputed_embeddings",
    "model_specific_data",
    "modality",
    "image_grid_thw",
    "mrope_positions",
    "mrope_position_delta",
    "hash",
)


def _summarize(value, tensors, key="value", depth=0):
    import torch

    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        tensors[key] = cpu
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        result = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if cpu.numel() and cpu.is_floating_point():
            floats = cpu.float()
            result.update(
                min=float(floats.min()),
                max=float(floats.max()),
                finite=bool(torch.isfinite(floats).all()),
            )
        if cpu.numel() <= 32:
            result["values"] = cpu.tolist()
        return result
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 5:
        return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        return {
            str(k): _summarize(v, tensors, f"{key}.{k}", depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 64 and all(isinstance(v, int) for v in value):
            tensors[key] = torch.tensor(value, dtype=torch.int64, device="cpu")
            return {
                "length": len(value),
                "sha256": hashlib.sha256(
                    json.dumps(value, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        return [
            _summarize(v, tensors, f"{key}.{i}", depth + 1) for i, v in enumerate(value)
        ]
    if hasattr(value, "size") and hasattr(value, "mode"):
        return {
            "type": type(value).__name__,
            "size": list(value.size),
            "mode": value.mode,
        }
    fields = {name: getattr(value, name) for name in _FIELDS if hasattr(value, name)}
    if fields:
        return _summarize(fields, tensors, key, depth + 1)
    return {"type": type(value).__name__, "repr": str(value)[:200]}


def emit(stage, payload, owner=None):
    """Diagnostic failures never replace the original test/runtime exception."""
    if not os.environ.get("VLM_DIAGNOSTICS_DIR"):
        return
    try:
        import torch

        root = Path(os.environ["VLM_DIAGNOSTICS_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time_ns()}-{os.getpid()}-{next(_SEQUENCE)}"
        tensors = {}
        record = {
            "stage": stage,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "rid": _RID.get(),
            "time_ns": time.time_ns(),
            "payload": _summarize(payload, tensors),
        }
        if owner is not None:
            processor = getattr(owner, "_processor", None)
            record["processor"] = type(processor).__name__
            record["image_processor"] = type(
                getattr(processor, "image_processor", None)
            ).__name__
            record["model_type"] = getattr(owner, "model_type", None)
            # Whole-record provenance, never use this to change resolved options.
            server_args = getattr(owner, "server_args", None)
            if server_args is not None:
                record["server_args"] = vars(server_args)
            ids = payload.get("input_ids") if isinstance(payload, Mapping) else None
            image_id = getattr(
                getattr(owner, "hf_config", None), "image_token_id", None
            )
            if ids is not None and image_id is not None:
                flat = (
                    ids.detach().cpu().flatten().tolist()
                    if isinstance(ids, torch.Tensor)
                    else list(ids)
                )
                spans, start = [], None
                for index, token in enumerate(flat + [None]):
                    if token == image_id and start is None:
                        start = index
                    elif token != image_id and start is not None:
                        spans.append([start, index - 1])
                        start = None
                record["image_token_id"] = image_id
                record["image_token_count"] = flat.count(image_id)
                record["image_token_spans"] = spans
        if tensors:
            name = f"{stamp}.pt"
            torch.save(tensors, root / name)
            record["tensor_file"] = name
        (root / f"{stamp}.json").write_text(
            json.dumps(record, default=str, ensure_ascii=False)
        )
        brief = {k: v for k, v in record.items() if k != "server_args"}
        print(
            "NPU_VLM_DIAG " + json.dumps(brief, default=str, ensure_ascii=False),
            flush=True,
        )
    except Exception as error:
        print(
            f"NPU_VLM_DIAG_ERROR {stage}: {type(error).__name__}: {error}", flush=True
        )


def _is_qwen(owner):
    config = getattr(owner, "hf_config", None)
    return (
        getattr(owner, "model_type", getattr(config, "model_type", None))
        == "qwen2_5_vl"
    )


def _wrap_method(cls, name):
    original = getattr(cls, name, None)
    if original is None or getattr(original, "_vlm_diagnostic", False):
        return
    signature = inspect.signature(original)

    def inputs(self, args, kwargs):
        values = dict(signature.bind(self, *args, **kwargs).arguments)
        values.pop("self", None)
        return values

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            if not _is_qwen(self):
                return await original(self, *args, **kwargs)
            values = inputs(self, args, kwargs)
            request = values.get("request_obj")
            token = _RID.set(getattr(request, "rid", None))
            try:
                emit(name + ".input", values, self)
                result = await original(self, *args, **kwargs)
                emit(
                    name + ".output",
                    {"result": result, "input_ids_may_be_padded": True},
                    self,
                )
                return result
            except Exception as error:
                emit(name + ".error", {"error": repr(error)}, self)
                raise
            finally:
                _RID.reset(token)
    else:

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            if not _is_qwen(self):
                return original(self, *args, **kwargs)
            emit(name + ".input", inputs(self, args, kwargs), self)
            result = original(self, *args, **kwargs)
            emit(name + ".output", result, self)
            return result

    wrapped._vlm_diagnostic = True
    setattr(cls, name, wrapped)


def _instrument(module):
    if module.__name__.endswith("base_processor"):
        cls = module.BaseMultimodalProcessor
        for name in ("process_mm_data", "_fast_image_processor_device"):
            _wrap_method(cls, name)
    elif module.__name__.endswith("processors.qwen_vl"):
        _wrap_method(module.QwenVLImageProcessor, "process_mm_data_async")
    else:
        original_factory = module.npu_wrapper_preprocess

        @functools.wraps(original_factory)
        def factory(function):
            original = original_factory(function)

            @functools.wraps(original)
            def preprocess(self, *args, **kwargs):
                emit("npu_preprocess.input", {"args": args, "kwargs": kwargs})
                result = original(self, *args, **kwargs)
                emit("npu_preprocess.output", result)
                return result

            return preprocess

        module.npu_wrapper_preprocess = factory
    print(f"NPU_VLM_DIAG_HOOK {module.__name__} pid={os.getpid()}", flush=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        _instrument(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec


def install_import_hook():
    import sys

    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
