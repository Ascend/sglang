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
from array import array
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

_RID = contextvars.ContextVar("vlm_diagnostic_rid", default=None)
_SEQUENCE = itertools.count()
_ITEM_REQUESTS = {}
_WEIGHT_OWNERS = set()
_LAYER_NAMES = {}
_MODEL_COUNTS = {}
_MODEL_OBSERVERS = set()
_FRAME = contextvars.ContextVar("vlm_diagnostic_frame", default=None)
_MODULES = {
    "sglang.srt.multimodal.processors.base_processor",
    "sglang.srt.multimodal.processors.qwen_vl",
    "sglang.srt.hardware_backend.npu.modules.qwen_vl_processor",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.managers.scheduler_components.request_receiver",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.mm_utils",
    "sglang.srt.models.qwen2_5_vl",
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
    "pad_value",
    "format",
    "rid",
    "mm_inputs",
    "multimodal_inputs",
    "origin_input_ids",
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
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "name": value.name}
    if depth >= 12:
        return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        return {
            str(k): _summarize(v, tensors, f"{key}.{k}", depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, array)):
        if isinstance(value, array):
            value = list(value)
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


def emit(stage, payload, owner=None, *, save_tensors=True):
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
            "rank": (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else None
            ),
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
        if tensors and save_tensors:
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
    config = getattr(owner, "hf_config", getattr(owner, "config", None))
    if config is None:
        config = getattr(getattr(owner, "model_config", None), "hf_config", None)
    return (
        getattr(owner, "model_type", getattr(config, "model_type", None))
        == "qwen2_5_vl"
    )


def _observe_requests(stage, requests):
    """Observe only materialized multimodal requests; never unwrap transports here."""
    try:
        for request in requests or []:
            batch = getattr(request, "batch", None)
            if batch is not None:
                _observe_requests(stage, batch)
                continue
            if getattr(request, "mm_inputs", None) is None:
                continue
            token = _RID.set(getattr(request, "rid", None))
            try:
                emit(stage, {"request": request})
            finally:
                _RID.reset(token)
    except Exception as error:
        print(f"NPU_VLM_DIAG_ERROR {stage}: {error!r}", flush=True)


def _remember_items(value):
    for item in getattr(value, "mm_items", []) or []:
        if getattr(item, "hash", None) is not None:
            _ITEM_REQUESTS[item.hash] = _RID.get()


def _items_rid(items):
    rids = {_ITEM_REQUESTS.get(getattr(item, "hash", None)) for item in items} - {None}
    return next(iter(rids)) if len(rids) == 1 else _RID.get()


def _vision_weights(owner):
    """Hash loaded per-rank weights once, without storing model weights in artifacts."""
    if id(owner) in _WEIGHT_OWNERS:
        return
    try:
        import torch

        hashes = {}
        for name, value in owner.named_parameters():
            cpu = value.detach().cpu().contiguous()
            hashes[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        for index, block in enumerate(owner.blocks):
            _LAYER_NAMES[id(block)] = f"blocks.{index}"
        emit("vision.loaded_weights", hashes, save_tensors=False)
        _WEIGHT_OWNERS.add(id(owner))
    except Exception as error:
        print(f"NPU_VLM_DIAG_ERROR vision.loaded_weights: {error!r}", flush=True)


def _decoder_observers(owner):
    if id(owner) in _MODEL_OBSERVERS:
        return
    import torch

    weights = {}
    for prefix, module in (("model", owner.model), ("lm_head", owner.lm_head)):
        for name, value in module.named_parameters():
            cpu = value.detach().cpu().contiguous()
            weights[f"{prefix}.{name}"] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
    emit("decoder.loaded_weights", weights, save_tensors=False)

    def selected(value):
        if isinstance(value, torch.Tensor) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [selected(item) for item in value]
        if isinstance(value, Mapping):
            return {key: selected(item) for key, item in value.items()}
        return {"type": type(value).__name__}

    def observe(module, label, stage):
        def before(_module, args, kwargs):
            frame = _FRAME.get()
            if frame is not None:
                emit(
                    stage + ".input",
                    {
                        "module": label,
                        "frame": frame,
                        "args": selected(args),
                        "kwargs": selected(kwargs),
                    },
                    save_tensors=False,
                )

        def after(_module, _args, _kwargs, result):
            frame = _FRAME.get()
            if frame is not None:
                emit(
                    stage + ".output",
                    {"module": label, "frame": frame, "result": selected(result)},
                    save_tensors=False,
                )

        module.register_forward_pre_hook(before, with_kwargs=True)
        module.register_forward_hook(after, with_kwargs=True)

    for index, layer in enumerate(owner.model.layers):
        observe(layer, f"layers.{index}", "decoder_layer")
        if index == 0:
            for name in (
                "input_layernorm",
                "self_attn",
                "self_attn.qkv_proj",
                "self_attn.rotary_emb",
                "self_attn.attn",
                "self_attn.o_proj",
                "post_attention_layernorm",
                "mlp",
                "mlp.gate_up_proj",
                "mlp.act_fn",
                "mlp.down_proj",
            ):
                module = layer
                for part in name.split("."):
                    module = getattr(module, part, None)
                if module is not None and hasattr(module, "register_forward_hook"):
                    observe(module, "layers.0." + name, "decoder_component")
    _MODEL_OBSERVERS.add(id(owner))


def _wrap_model_forward(cls):
    original = cls.forward
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        try:
            values = dict(signature.bind(self, *args, **kwargs).arguments)
            batch = values["forward_batch"]
            items = [
                item
                for mm in getattr(batch, "mm_inputs", []) or []
                if mm is not None
                for item in mm.mm_items
            ]
            rid = _items_rid(items) or _ITEM_REQUESTS.get(("model", id(self)))
        except Exception as error:
            print(f"NPU_VLM_DIAG_ERROR model_forward.bind: {error!r}", flush=True)
            return original(self, *args, **kwargs)
        if rid is None:
            return original(self, *args, **kwargs)
        count = _MODEL_COUNTS.get((id(self), rid), 0)
        _MODEL_COUNTS[(id(self), rid)] = count + 1
        # Prefill plus the first eight decode steps includes the first divergent output token.
        if count > 8:
            return original(self, *args, **kwargs)
        rid_token = frame_token = None
        try:
            rid_token = _RID.set(rid)
            # Decode batches release mm_inputs; retain the single-request RID established at prefill.
            _ITEM_REQUESTS[("model", id(self))] = rid
            frame = {"index": count, "mode": str(batch.forward_mode)}
            frame_token = _FRAME.set(frame)
            _decoder_observers(self)
            payload = {
                "frame": frame,
                "input_ids": values.get("input_ids"),
                "positions": values.get("positions"),
            }
            for key in (
                "mrope_positions",
                "seq_lens",
                "extend_seq_lens",
                "extend_prefix_lens",
                "extend_start_loc",
                "extend_seq_lens_cpu",
                "extend_prefix_lens_cpu",
            ):
                payload[key] = getattr(batch, key, None)
            emit("model_forward.input", payload)
        except Exception as error:
            print(f"NPU_VLM_DIAG_ERROR model_forward.input: {error!r}", flush=True)
        try:
            result = original(self, *args, **kwargs)
            try:
                logits = getattr(result, "next_token_logits", None)
                emit(
                    "model_forward.output",
                    {"frame": _FRAME.get(), "next_token_logits": logits},
                )
            except Exception as error:
                print(f"NPU_VLM_DIAG_ERROR model_forward.output: {error!r}", flush=True)
            return result
        finally:
            if frame_token is not None:
                _FRAME.reset(frame_token)
            if rid_token is not None:
                _RID.reset(rid_token)

    cls.forward = wrapped


def _wrap_runtime_method(cls, name, stage, *, check_owner=True, save_tensors=True):
    original = getattr(cls, name)
    signature = inspect.signature(original)

    def prepare(self, args, kwargs):
        values = dict(signature.bind(self, *args, **kwargs).arguments)
        values.pop("self", None)
        request = values.get("tokenized_obj", values.get("recv_req"))
        rid = getattr(request, "rid", _RID.get())
        if stage == "image_feature":
            rid = _items_rid(values.get("items", []))
        if stage == "vision":
            _vision_weights(self)
        values["layer"] = _LAYER_NAMES.get(id(self))
        return values, rid

    def before(self, args, kwargs):
        # All bookkeeping is diagnostic: failures cannot change inference behavior.
        try:
            values, rid = prepare(self, args, kwargs)
            token = _RID.set(rid)
            if stage != "scheduler_receive":
                emit(stage + ".input", values, save_tensors=save_tensors)
            return token, values
        except Exception as error:
            print(f"NPU_VLM_DIAG_ERROR {stage}.input: {error!r}", flush=True)
            return None, {}

    def after(values, result):
        try:
            if stage == "scheduler_receive":
                _observe_requests(stage + ".output", values.get("recv_reqs"))
            else:
                if stage == "scheduler_prepare":
                    _remember_items(result)
                emit(
                    stage + ".output",
                    {"result": result, "layer": values.get("layer")},
                    save_tensors=save_tensors,
                )
        except Exception as error:
            print(f"NPU_VLM_DIAG_ERROR {stage}.output: {error!r}", flush=True)

    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def wrapped(self, *args, **kwargs):
            if check_owner and not _is_qwen(self):
                return await original(self, *args, **kwargs)
            token, values = before(self, args, kwargs)
            try:
                result = await original(self, *args, **kwargs)
                after(values, result)
                return result
            finally:
                if token is not None:
                    _RID.reset(token)
    else:

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            if check_owner and not _is_qwen(self):
                return original(self, *args, **kwargs)
            # Receiver polls with no work most of the time; avoid empty snapshots.
            if stage == "scheduler_receive" and not (args or kwargs.get("recv_reqs")):
                return original(self, *args, **kwargs)
            token, values = before(self, args, kwargs)
            try:
                result = original(self, *args, **kwargs)
                after(values, result)
                return result
            finally:
                if token is not None:
                    _RID.reset(token)

    setattr(cls, name, wrapped)


def _wrap_embedding(module):
    original = module.embed_mm_inputs
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        values = dict(signature.bind(*args, **kwargs).arguments)
        owner = values.get("multimodal_model")
        if not _is_qwen(owner):
            return original(*args, **kwargs)
        selected = {
            key: values.get(key)
            for key in (
                "mm_inputs_list",
                "extend_prefix_lens",
                "extend_seq_lens",
                "input_ids",
            )
        }
        token = None
        try:
            items = [
                item
                for mm in values.get("mm_inputs_list", []) or []
                if mm is not None
                for item in mm.mm_items
            ]
            token = _RID.set(_items_rid(items))
            emit("embedding_scatter.input", selected)
        except Exception as error:
            print(f"NPU_VLM_DIAG_ERROR embedding_scatter.input: {error!r}", flush=True)
        try:
            result = original(*args, **kwargs)
            emit("embedding_scatter.output", {"result": result})
            return result
        finally:
            if token is not None:
                _RID.reset(token)

    module.embed_mm_inputs = wrapped


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
    elif module.__name__.endswith("tokenizer_manager"):
        _wrap_runtime_method(
            module.TokenizerManager, "_send_one_request", "tokenizer_send"
        )
    elif module.__name__.endswith("request_receiver"):
        _wrap_runtime_method(
            module.SchedulerRequestReceiver,
            "_finalize_shm_features",
            "scheduler_receive",
        )
    elif module.__name__.endswith("scheduler"):
        _wrap_runtime_method(
            module.Scheduler, "handle_generate_request", "scheduler_request"
        )
        _wrap_runtime_method(
            module.Scheduler, "_get_multimodal_inputs", "scheduler_prepare"
        )
    elif module.__name__.endswith("mm_utils"):
        _wrap_embedding(module)
    elif module.__name__.endswith("models.qwen2_5_vl"):
        _wrap_model_forward(module.Qwen2_5_VLForConditionalGeneration)
        _wrap_runtime_method(
            module.Qwen2_5_VLForConditionalGeneration,
            "get_image_feature",
            "image_feature",
        )
        _wrap_runtime_method(
            module.Qwen2_5_VisionTransformer, "forward", "vision", check_owner=False
        )
        _wrap_runtime_method(
            module.Qwen2_5_VisionBlock,
            "forward",
            "vision_block",
            check_owner=False,
            save_tensors=False,
        )
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
        try:
            _instrument(module)
        except Exception as error:
            print(f"NPU_VLM_DIAG_ERROR hook {module.__name__}: {error!r}", flush=True)

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
