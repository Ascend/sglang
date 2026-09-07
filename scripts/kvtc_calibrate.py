#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
import os
import math
import re
import torch
import pprint
import logging
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

sys.path.append("../python")
from sglang.srt.mem_cache.allocator import token
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.utils.hf_transformers.common import get_rope_config
from sglang.srt.server_args import (
       ServerArgs,
       get_global_server_args,
       set_global_server_args_for_scheduler
    )
from transformers import AutoConfig
from collections import Counter, defaultdict
from enum import Enum, IntEnum

logger = logging.getLogger()
WORKER_DIR_PATTERN = re.compile(r"^tp_(\d+)_pp_(\d+)$")
KVTC_FILE_VERSION="v1-noquant"
TOKEN_SELECTION_FILE_VERSION = 2
SVD_ARTIFACT_FILE_VERSION = 1
SINK_TOKENS = 128
SVD_WORKERS = 4


def stable_digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TokenSelectionError(ValueError):
    pass


class SVDArtifactError(ValueError):
    pass


class TokenSelectionStore:
    """Finalize, persist, and serve one global selection per dump request."""

    def __init__(
        self,
        *,
        mode,
        path,
        input_dir,
        model_dir,
        sample_tokens,
        sampling_policy,
        dump_directories,
        workers,
    ):
        if mode not in (None, "load", "save"):
            raise TokenSelectionError(f"Invalid token selection mode: {mode}")
        self.mode = mode
        self.path = path
        self.input_dir = input_dir.absolute()
        self._selections = {}
        self._loaded_selections = None
        self._finalized = False

        self.metadata = {
            "kvtc_version": KVTC_FILE_VERSION,
            "model_dir": str(model_dir.resolve()),
            "sample_tokens": sample_tokens,
            "sampling_policy": sampling_policy.name.lower(),
            "dump_directories": [
                self._relative_path(path) for path in dump_directories
            ],
            "workers": workers,
            "sink_tokens_per_side": SINK_TOKENS,
        }
        if self.mode == "load":
            self._load()

    def _relative_path(self, path):
        try:
            return path.absolute().relative_to(self.input_dir).as_posix()
        except ValueError as error:
            raise TokenSelectionError(
                f"Selection source {path} is outside input directory {self.input_dir}"
            ) from error

    def _selection_key(self, dataset_path, request_id):
        return self._relative_path(dataset_path), request_id

    def _load(self):
        try:
            with self.path.open(encoding="utf-8") as file:
                document = json.load(file)
        except FileNotFoundError as error:
            raise TokenSelectionError(
                f"Token selection file does not exist: {self.path}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise TokenSelectionError(
                f"Cannot load token selection file {self.path}: {error}"
            ) from error

        if not isinstance(document, dict):
            raise TokenSelectionError(
                f"Token selection file {self.path} must contain a JSON object"
            )
        if document.get("version") != TOKEN_SELECTION_FILE_VERSION:
            raise TokenSelectionError(
                f"Unsupported token selection file version in {self.path}: "
                f"{document.get('version')!r}; expected {TOKEN_SELECTION_FILE_VERSION}"
            )

        saved_metadata = document.get("metadata")
        if not isinstance(saved_metadata, dict):
            raise TokenSelectionError(
                f"Token selection file {self.path} has no valid metadata object"
            )
        if saved_metadata != self.metadata:
            differing_fields = sorted(
                key
                for key in set(saved_metadata) | set(self.metadata)
                if saved_metadata.get(key) != self.metadata.get(key)
            )
            raise TokenSelectionError(
                f"Token selection file {self.path} is incompatible with this run; "
                f"different metadata fields: {', '.join(differing_fields)}"
            )

        selections = document.get("selections")
        if not isinstance(selections, list):
            raise TokenSelectionError(
                f"Token selection file {self.path} has no valid selections list"
            )

        loaded_selections = {}
        for selection in selections:
            try:
                key = (
                    selection["dump_directory"],
                    selection["request_id"],
                )
            except (KeyError, TypeError) as error:
                raise TokenSelectionError(
                    f"Malformed selection entry in {self.path}: {selection!r}"
                ) from error
            if key in loaded_selections:
                raise TokenSelectionError(
                    f"Duplicate selection entry in {self.path}: {key}"
                )
            loaded_selections[key] = selection

        self._loaded_selections = loaded_selections
        logger.info(
            "Loaded %d token selections from %s",
            len(self._loaded_selections),
            self.path,
        )

    @staticmethod
    def _validate_indices(key, indices, token_count, sampling_budget):
        if (
            not isinstance(indices, list)
            or len(indices) != sampling_budget
            or any(type(index) is not int for index in indices)
            or indices != sorted(set(indices))
            or any(
                index < SINK_TOKENS or index >= token_count - SINK_TOKENS
                for index in indices
            )
        ):
            raise TokenSelectionError(
                f"Saved selection for {key} contains invalid token indices"
            )

    def finalize(self, requests):
        """Finalize every selection and save it, when requested, before SVD."""
        if self._finalized:
            raise TokenSelectionError("Token selections have already been finalized")

        expected_keys = set()
        for request in requests:
            key = self._selection_key(
                request["dataset_path"], request["request_id"]
            )
            if key in expected_keys:
                raise TokenSelectionError(f"Duplicate global dump request: {key}")
            expected_keys.add(key)

            expected = {
                "dump_directory": key[0],
                "request_id": key[1],
                "token_count": request["token_count"],
                "sampling_budget": request["sampling_budget"],
            }
            if self.mode == "load":
                selection = self._loaded_selections.get(key)
                if selection is None:
                    raise TokenSelectionError(
                        f"No saved global token selection for {key}"
                    )
                differing_fields = [
                    field
                    for field, value in expected.items()
                    if selection.get(field) != value
                ]
                if differing_fields:
                    raise TokenSelectionError(
                        f"Saved selection for {key} does not match the current dump; "
                        f"different fields: {', '.join(differing_fields)}"
                    )
                indices = selection.get("selected_token_indices")
                self._validate_indices(
                    key,
                    indices,
                    request["token_count"],
                    request["sampling_budget"],
                )
            else:
                indices = sorted(
                    random.sample(
                        range(
                            SINK_TOKENS,
                            request["token_count"] - SINK_TOKENS,
                        ),
                        request["sampling_budget"],
                    )
                )

            self._selections[key] = {
                **expected,
                "selected_token_indices": indices,
            }

        if self.mode == "load":
            unused = set(self._loaded_selections) - expected_keys
            if unused:
                raise TokenSelectionError(
                    f"Token selection file {self.path} contains {len(unused)} "
                    f"unused selections, including: {sorted(unused)[:3]}"
                )

        self._finalized = True
        if self.mode == "save":
            self._save()
        elif self.mode == "load":
            logger.info("Reused all global token selections from %s", self.path)
        else:
            logger.info("Finalized %d global token selections", len(self._selections))

    def _save(self):
        document = {
            "version": TOKEN_SELECTION_FILE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "selections": sorted(
                self._selections.values(),
                key=lambda selection: (
                    selection["dump_directory"],
                    selection["request_id"],
                ),
            ),
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(document, file, indent=2)
                file.write("\n")
            os.replace(temporary_path, self.path)
        except OSError as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise TokenSelectionError(
                f"Cannot save token selections to {self.path}: {error}"
            ) from error
        logger.info(
            "Saved %d global token selections to %s",
            len(self._selections),
            self.path,
        )

    def get(self, dataset_path, request_id, token_count):
        if not self._finalized:
            raise TokenSelectionError("Token selections were not finalized before SVD")
        key = self._selection_key(dataset_path, request_id)
        selection = self._selections.get(key)
        if selection is None:
            raise TokenSelectionError(f"No global token selection for {key}")
        if selection["token_count"] != token_count:
            raise TokenSelectionError(
                f"Token count changed after discovery for {key}: expected "
                f"{selection['token_count']}, got {token_count}"
            )
        return selection["selected_token_indices"]

    def identity(self):
        if not self._finalized:
            raise TokenSelectionError(
                "Token selections were not finalized before computing their identity"
            )
        return {
            "metadata": self.metadata,
            "selections": sorted(
                self._selections.values(),
                key=lambda selection: (
                    selection["dump_directory"],
                    selection["request_id"],
                ),
            ),
        }

    def digest(self):
        return stable_digest(self.identity())


class Rope(object):
    rotary_emb = None
    rotary_dim = None
    is_neox_style = None

    @classmethod
    def ensure_server_args(cls):
        try:
            get_global_server_args()
        except ValueError:
            set_global_server_args_for_scheduler(ServerArgs(model_path="DUMMY"))

    @classmethod
    def load_model_config(cls, model_path):
        logger.info(f"Loading config from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        rope_theta, rope_scaling = get_rope_config(cfg)
        rope_theta = float(rope_theta)

        if rope_scaling and "mrope_section" in rope_scaling:
            raise ValueError(
                "Calibration does not support mRoPE models because KV dumps do not "
                "include the required multimodal position IDs."
            )

        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = cfg.hidden_size // cfg.num_attention_heads

        cls.ensure_server_args()
        cls.rotary_emb = get_rope(
            head_size=int(head_dim),
            rotary_dim=int(head_dim),
            max_position=int(getattr(cfg, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=float(getattr(cfg, "partial_rotary_factor", 1.0)),
            is_neox_style=getattr(cfg, "rope_is_neox_style", True),
            dtype=torch.float32,
        )
        if not hasattr(cls.rotary_emb, "cos_sin_cache"):
            raise ValueError(
                "Calibration only supports Qwen3-compatible 1-D RoPE variants "
                "with a direct cos/sin cache."
            )

        cls.rotary_dim = cls.rotary_emb.rotary_dim
        cls.is_neox_style = cls.rotary_emb.is_neox_style
        logger.info(
            "model.rope theta=%s rotary_dim=%s neox_style=%s scaling=%s",
            rope_theta,
            cls.rotary_dim,
            cls.is_neox_style,
            rope_scaling,
        )

    @classmethod
    def invert_rope(cls, tensor):
        tokens, _, _, _ = tensor.shape
        rope_cache_len = 64 * 1024

        assert tokens <= rope_cache_len, f"{tokens=}, {rope_cache_len=}"
        assert cls.rotary_emb is not None, "RoPE model config was not loaded"

        cos_sin_cache = cls.rotary_emb.cos_sin_cache
        if tokens > cos_sin_cache.shape[0]:
            raise ValueError(
                f"Dump has {tokens} tokens, but the model RoPE cache has only "
                f"{cos_sin_cache.shape[0]} positions."
            )

        cos, sin = cos_sin_cache[:tokens].chunk(2, dim=-1)
        cos = cos[:, None, None, :].to(tensor.dtype)
        sin = sin[:, None, None, :].to(tensor.dtype)
        denominator = (cos.square() + sin.square()).clamp_min(
            torch.finfo(tensor.dtype).eps
        )

        rotated = tensor[..., : cls.rotary_dim]
        passthrough = tensor[..., cls.rotary_dim :]
        if cls.is_neox_style:
            first, second = torch.chunk(rotated, 2, dim=-1)
            inverted = torch.cat(
                (
                    (first * cos + second * sin) / denominator,
                    (second * cos - first * sin) / denominator,
                ),
                dim=-1,
            )
        else:
            first, second = rotated[..., ::2], rotated[..., 1::2]
            inverted = torch.stack(
                (
                    (first * cos + second * sin) / denominator,
                    (second * cos - first * sin) / denominator,
                ),
                dim=-1,
            ).flatten(-2)

        return torch.cat((inverted, passthrough), dim=-1)


def get_tensors_paths(tensor_dir, prefix):
    tensor_files = [x for x in tensor_dir.iterdir() if x.is_file()]
    tensor_files = [x for x in tensor_files if x.name.startswith(prefix)]
    file_groups = list(set([str(x).split(".part")[0] for x in tensor_files]))
    file_groups = [
        sorted([f for f in tensor_files if g in str(f)]) for g in file_groups
    ]

    logger.info(f"Detected tensor files:\n{file_groups}")

    return file_groups


def tensor_sorting_fn(f):
    name, chunk_id, layer_id = re.search(
        r"(.*)chunk_(\d+)-layer_(\d+)\.bin$", str(f)
    ).groups()
    return (name, int(layer_id), int(chunk_id))


def count_tokens(datasets_list):
    ret = {}

    for kv_cache_paths in datasets_list:
        ret[kv_cache_paths] = {"ignored": 0, "short": 0, "long": 0}
        for paths in datasets_list[kv_cache_paths]:
            token_count = torch.concat([torch.load(p, map_location="cpu") for p in paths]).shape[0]
            if token_count < 1000:
                ret[kv_cache_paths]["ignored"] += 1
                continue
            if token_count < 8000:
                ret[kv_cache_paths]["short"] += 1
            else:
                ret[kv_cache_paths]["long"] += 1

    return ret


def load_tensor(paths):
    """Load one request only when every chunk has the same complete layer set."""
    if not paths:
        logger.warning("Discarding empty dump request")
        return None, None

    request_path = paths[0].parent
    request_name = paths[0].name.split("-K-", 1)[0].split("-V-", 1)[0]
    chunks = defaultdict(dict)
    for path in paths:
        match = re.search(r"chunk_(\d+)-layer_(\d+)\.bin$", path.name)
        if match is None:
            logger.warning("Discarding %s: unrecognized dump filename %s", request_name, path)
            return None, None

        chunk_id, layer_id = map(int, match.groups())
        if layer_id in chunks[chunk_id]:
            logger.warning(
                "Discarding %s: duplicate layer %s in chunk %s at %s",
                request_name,
                layer_id,
                chunk_id,
                request_path,
            )
            return None, None
        chunks[chunk_id][layer_id] = path

    chunk_ids = sorted(chunks)
    if chunk_ids != list(range(len(chunk_ids))):
        logger.warning(
            "Discarding %s: non-contiguous chunk IDs %s at %s",
            request_name,
            chunk_ids,
            request_path,
        )
        return None, None

    expected_layer_ids = None
    chunk_tensors = []
    for chunk_id in chunk_ids:
        layer_ids = sorted(chunks[chunk_id])
        if expected_layer_ids is None:
            expected_layer_ids = layer_ids
        elif layer_ids != expected_layer_ids:
            logger.warning(
                "Discarding %s: chunk %s has layers %s, expected %s at %s",
                request_name,
                chunk_id,
                layer_ids,
                expected_layer_ids,
                request_path,
            )
            return None, None

        try:
            layer_tensors = [
                torch.load(chunks[chunk_id][layer_id], map_location="cpu")
                for layer_id in layer_ids
            ]
            chunk = torch.stack(layer_tensors)
        except Exception as error:
            logger.warning(
                "Discarding %s: cannot load chunk %s at %s: %s",
                request_name,
                chunk_id,
                request_path,
                error,
            )
            return None, None

        if chunk.ndim != 4:
            logger.warning(
                "Discarding %s: chunk %s has shape %s, expected [layer, token, head, head_dim]",
                request_name,
                chunk_id,
                tuple(chunk.shape),
            )
            return None, None
        if chunk_tensors and (
            chunk.shape[0] != chunk_tensors[0].shape[0]
            or chunk.shape[2:] != chunk_tensors[0].shape[2:]
            or chunk.dtype != chunk_tensors[0].dtype
        ):
            logger.warning(
                "Discarding %s: chunk %s shape/dtype %s/%s differs from %s/%s",
                request_name,
                chunk_id,
                tuple(chunk.shape),
                chunk.dtype,
                tuple(chunk_tensors[0].shape),
                chunk_tensors[0].dtype,
            )
            return None, None
        chunk_tensors.append(chunk)

    try:
        ret = torch.concat(chunk_tensors, dim=1).transpose(0, 1)
    except RuntimeError as error:
        logger.warning(
            "Discarding %s: cannot combine validated chunks at %s: %s",
            request_name,
            request_path,
            error,
        )
        return None, None
    if ret.dtype != torch.bfloat16:
        logger.warning(
            "Discarding %s: dtype %s, expected torch.bfloat16", request_name, ret.dtype
        )
        return None, None

    logger.info("Loaded %s from %s", tuple(ret.shape), request_name)

    torch.cpu.synchronize()

    return ret, ret.shape[0]


def sample_tokens(tensor, token_indices):
    ids = torch.tensor(token_indices, device="cpu", dtype=torch.long)
    logger.debug(f"Sample ids\n{ids}")

    ret = tensor[ids, :].to(dtype=torch.float32, copy=True)
    logger.info(f"Original tokens: {tensor.shape}. Sampled tokens: {ret.shape}")

    return ret


def transform_tensors(tensors):
    if not tensors:
        raise ValueError("No valid calibration tensors were loaded")

    feature_shapes = Counter(tensor.shape[1:] for tensor in tensors)
    feature_shape, _ = feature_shapes.most_common(1)[0]
    valid_tensors = [tensor for tensor in tensors if tensor.shape[1:] == feature_shape]
    discarded = len(tensors) - len(valid_tensors)
    if discarded:
        logger.warning(
            "Discarding %s sampled tensors with non-canonical feature shapes; "
            "using %s from %s tensors",
            discarded,
            feature_shape,
            len(valid_tensors),
        )
        logger.warning("Observed sampled feature shapes: %s", dict(feature_shapes))

    ret = torch.concat(valid_tensors, dim=0).flatten(start_dim=1)

    torch.cpu.synchronize()

    return ret


class TensorFileManager(object):
    class SamplingPolicy(Enum):
        STRICT = 1
        RELAXED = 2
        OPEN = 3

    class KV(str, Enum):
        K = "-K-"
        V = "-V-"

    class Sequence(IntEnum):
        IGNORE = 0
        SHORT = 1000
        LONG = 8000

        @classmethod
        def bucket(cls, length):
            if length < cls.SHORT:
                return cls.IGNORE
            elif length < cls.LONG:
                return cls.SHORT
            else:
                return cls.LONG

    def __init__(self, input_dir_list, tp_pp_worker, sampling_policy):
        self.datasets = {}
        self.context_groups = {}
        self.token_counts = {}
        self.total_tokens = {kv: 0 for kv in TensorFileManager.KV}
        self.sampling_policy = sampling_policy

        for dataset_path in input_dir_list:
            self.datasets[dataset_path] = {}
            self.context_groups[dataset_path] = {}
            self.token_counts[dataset_path] = {}
            for kv in TensorFileManager.KV:
                files, counter, token_counts = self._get_tensors_paths(
                    dataset_path, tp_pp_worker, kv
                )
                self.datasets[dataset_path][kv] = files
                self.context_groups[dataset_path][kv] = counter
                self.token_counts[dataset_path][kv] = token_counts

        self.datasets_list = list(self.datasets.keys())

        self._log_detected_files()
        self._log_detected_files()

    def _get_tensors_paths(self, tensor_dir, tp_pp_worker, kv):
        """
        An example of tensor file name: fa712099aaa24207a1667853e4401503-K-chunk_0-layer_0.bin
        File format explained:
        fa712099aaa24207a1667853e4401503 - request id. All files from one sequence have the same request id
        -K- - file contains dumped keys. For values, it would be '-V-'
        chunk_0 - A sequence might be split into multiple chunks
        layer_0 - Each layer of kv cache is saved as a separate file. The number of layer is model specific

        To put together one full chunk, we need to take all its layers
        To put together one sequence, we to take all reconstructed chunks

        """
        tensor_files = [x for x in (tensor_dir / tp_pp_worker).iterdir() if x.is_file()]
        tensor_files = [x for x in tensor_files if kv in x.name]

        sequences = defaultdict(list)
        for tf in tensor_files:
            seq_prefix = tf.name.split(kv.value, 1)[0]
            sequences[seq_prefix].append(tf)

        for seq in sequences:
            sequences[seq].sort(key=tensor_sorting_fn)

        file_groups = list(sequences.values())

        ret = []
        token_counts = {}
        counter = {b: 0 for b in TensorFileManager.Sequence}
        logger.info(f"Looking for {kv} tensors at {tensor_dir / tp_pp_worker}")
        for fg in file_groups:
            chunks = [
                path
                for path in fg
                if re.search(r"chunk_\d+-layer_0\.bin$", path.name)
            ]
            if not chunks:
                logger.warning(
                    "Skipping request %s during discovery: no layer-0 dump files",
                    fg[0] if fg else "<empty>",
                )
                continue
            try:
                token_count = torch.concat(
                    [torch.load(path, map_location="cpu") for path in chunks]
                ).shape[0]
            except Exception as error:
                logger.warning(
                    "Skipping request %s during discovery: cannot load layer-0 dumps: %s",
                    fg[0] if fg else "<empty>",
                    error,
                )
                continue
            request_id = fg[0].name.split(kv.value, 1)[0]
            token_counts[request_id] = token_count
            bucket = TensorFileManager.Sequence.bucket(token_count)
            match bucket:
                case TensorFileManager.Sequence.IGNORE:
                    counter[bucket] += 1
                    logger.info(
                        f"Ignoring too short sequence {kv}. min_len=1000, ignored len={token_count}"
                    )
                case TensorFileManager.Sequence.SHORT:
                    counter[bucket] += 1
                    self.total_tokens[kv] += token_count
                    ret.append(fg)
                case TensorFileManager.Sequence.LONG:
                    counter[bucket] += 1
                    self.total_tokens[kv] += token_count
                    ret.append(fg)

        return ret, counter, token_counts

    def _log_detected_files(self):
        logger.debug(f"Tensor files:")
        for ds in self.datasets_list:
            for kv in TensorFileManager.KV:
                for path_list in self.datasets[ds][kv]:
                    logger.debug(f"{path_list}")
            logger.debug(f"{self.context_groups[ds][kv]}")

    def get_sequence_inventory(self, kv):
        return {
            (dataset_path, request_id): token_count
            for dataset_path in self.datasets_list
            for request_id, token_count in self.token_counts[dataset_path][kv].items()
        }

    def get_token_budget(self, kv: KV, dataset_name, N, token_cnt):
        sink_tokens = 2 * SINK_TOKENS

        if self.sampling_policy == TensorFileManager.SamplingPolicy.STRICT:
            assert token_cnt >= 1000
            # Divide the total sample count across every nonempty dataset/bucket.
            bucket = TensorFileManager.Sequence.bucket(token_cnt)

            assert bucket != TensorFileManager.Sequence.IGNORE, (
                f"Invalid sequence length {token_cnt}"
            )

            nonempty_groups = sum(
                self.context_groups[dataset][kv][bucket] > 0
                for dataset in self.datasets_list
                for bucket in (
                    TensorFileManager.Sequence.SHORT,
                    TensorFileManager.Sequence.LONG,
                )
            )
            token_budget = math.ceil(
                N
                / nonempty_groups
                / self.context_groups[dataset_name][kv][bucket]
            )

            if token_budget > token_cnt - sink_tokens:
                raise ValueError(
                    f"Too short sequence. Min length {token_budget + sink_tokens}, got {token_cnt}. "
                    "Generate longer KV cache dumps, set lower sample (-N) value, or "
                    "consider switching sampling policy to 'relaxed' (bear in mind it may hurt LLM accuracy)"
                )

            return token_budget

        elif self.sampling_policy == TensorFileManager.SamplingPolicy.RELAXED:
            assert token_cnt >= 1000
            # Equal amount of tokens will be sampled from each dataset, but ignore short vs long contexts

            dataset_token_budget = math.ceil(N / len(self.datasets_list))

            sequences_cnt = (
                self.context_groups[dataset_name][kv][TensorFileManager.Sequence.SHORT]
                + self.context_groups[dataset_name][kv][TensorFileManager.Sequence.LONG]
            )

            token_budget = math.ceil(dataset_token_budget / sequences_cnt)

            if token_budget > token_cnt - sink_tokens:
                raise ValueError(
                    f"Too short sequence. Min length {token_budget + sink_tokens}, got {token_cnt}. "
                    "Generate longer KV cache dumps, set lower sample (-N) value, or "
                    "consider switching sampling policy to 'open' (bear in mind it may hurt LLM accuracy)"
                )

            return token_budget
        elif self.sampling_policy == TensorFileManager.SamplingPolicy.OPEN:
            # Equal amount of tokens will be sampled from each sequence

            sequences_cnt = 0

            for dataset in self.context_groups:
                sequences_cnt += self.context_groups[dataset][kv][
                    TensorFileManager.Sequence.SHORT
                ]
                sequences_cnt += self.context_groups[dataset][kv][
                    TensorFileManager.Sequence.LONG
                ]

            token_budget = math.ceil(N / sequences_cnt)

            if token_budget > token_cnt - sink_tokens:
                raise ValueError(
                    f"Too short sequence. Min length {token_budget + sink_tokens}, got {token_cnt}. "
                    "Generate longer KV cache dumps, set lower sample (-N) value, or "
                    "consider switching sampling policy to 'open' (bear in mind it may hurt LLM accuracy)"
                )

            return token_budget

        else:
            assert False, f"Invalid sampling policy {self.sampling_policy}"


def build_global_sampling_requests(tensor_managers, workers, sample_tokens):
    """Validate identical worker/KV inventories and compute shared budgets."""
    reference_worker = workers[0]
    reference_manager = tensor_managers[reference_worker]
    reference_kv = TensorFileManager.KV.K
    reference_inventory = reference_manager.get_sequence_inventory(reference_kv)
    if not reference_inventory:
        raise TokenSelectionError(
            f"No dump requests found for {reference_worker}/{reference_kv.name}"
        )

    reference_keys = set(reference_inventory)
    for worker in workers:
        manager = tensor_managers[worker]
        for kv in TensorFileManager.KV:
            inventory = manager.get_sequence_inventory(kv)
            inventory_keys = set(inventory)
            if inventory_keys != reference_keys:
                missing = sorted(reference_keys - inventory_keys, key=str)[:3]
                extra = sorted(inventory_keys - reference_keys, key=str)[:3]
                raise TokenSelectionError(
                    "Dump request sets differ across workers or K/V tensors: "
                    f"{worker}/{kv.name} is missing {missing} and has extra {extra} "
                    f"relative to {reference_worker}/{reference_kv.name}"
                )

            mismatched = [
                (key, reference_inventory[key], inventory[key])
                for key in sorted(reference_keys, key=str)
                if inventory[key] != reference_inventory[key]
            ]
            if mismatched:
                key, expected, actual = mismatched[0]
                raise TokenSelectionError(
                    "Dump token counts differ across workers or K/V tensors for "
                    f"{key}: {reference_worker}/{reference_kv.name} has {expected}, "
                    f"but {worker}/{kv.name} has {actual}"
                )

    usable_inventory = {
        key: token_count
        for key, token_count in reference_inventory.items()
        if TensorFileManager.Sequence.bucket(token_count)
        != TensorFileManager.Sequence.IGNORE
    }
    if not usable_inventory:
        raise TokenSelectionError("No usable dump requests remain after length filtering")

    requests = []
    for (dataset_path, request_id), token_count in sorted(
        usable_inventory.items(), key=lambda item: (str(item[0][0]), item[0][1])
    ):
        sampling_budget = reference_manager.get_token_budget(
            reference_kv, dataset_path, sample_tokens, token_count
        )
        if sampling_budget >= token_count - 2 * SINK_TOKENS:
            raise TokenSelectionError(
                f"Sampling budget {sampling_budget} leaves no non-sink tokens in "
                f"{dataset_path}/{request_id} with {token_count} tokens"
            )
        requests.append(
            {
                "dataset_path": dataset_path,
                "request_id": request_id,
                "token_count": token_count,
                "sampling_budget": sampling_budget,
            }
        )

    logger.info(
        "Validated %d dump requests across %d workers and both K/V tensors; "
        "%d requests are eligible for global sampling",
        len(reference_inventory),
        len(workers),
        len(requests),
    )
    return requests


def build_svd_cache_metadata(
    *,
    input_dir,
    model_dir,
    dump_directories,
    workers,
    sampling_requests,
    sample_tokens,
    sampling_policy,
    svd_dim,
    svd_iter,
):
    input_dir = input_dir.absolute()
    return {
        "artifact_version": SVD_ARTIFACT_FILE_VERSION,
        "kvtc_version": KVTC_FILE_VERSION,
        "input_dir": str(input_dir),
        "model_dir": str(model_dir.resolve()),
        "dump_directories": [
            path.absolute().relative_to(input_dir).as_posix()
            for path in dump_directories
        ],
        "workers": list(workers),
        "sample_tokens": sample_tokens,
        "sampling_policy": sampling_policy.name.lower(),
        "sink_tokens_per_side": SINK_TOKENS,
        "sampling_requests_digest": stable_digest(
            [
                {
                    "dump_directory": request["dataset_path"]
                    .absolute()
                    .relative_to(input_dir)
                    .as_posix(),
                    "request_id": request["request_id"],
                    "token_count": request["token_count"],
                    "sampling_budget": request["sampling_budget"],
                }
                for request in sampling_requests
            ]
        ),
        "svd_dim": svd_dim,
        "svd_iter": svd_iter,
    }


def build_svd_cache_directory(output_path, cache_metadata):
    configuration_digest = stable_digest(cache_metadata)
    run_name = (
        f"kvtc-{KVTC_FILE_VERSION}"
        f"_N-{cache_metadata['sample_tokens']}"
        f"_policy-{cache_metadata['sampling_policy']}"
        f"_q-{cache_metadata['svd_dim']}"
        f"_niter-{cache_metadata['svd_iter']}"
        f"_{configuration_digest}"
    )
    return output_path.with_name(f"{output_path.name}.svd-artifacts") / run_name


def build_svd_job_metadata(cache_metadata, selection_digest, worker, kv):
    match = WORKER_DIR_PATTERN.fullmatch(worker)
    if match is None:
        raise SVDArtifactError(f"Invalid worker name for SVD artifact: {worker}")
    tp_rank, pp_rank = map(int, match.groups())
    return {
        **cache_metadata,
        "token_selection_digest": selection_digest,
        "worker": worker,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "kv": kv.name,
        "undo_rope": kv == TensorFileManager.KV.K,
    }


def save_svd_artifact(path, temporary_path, metadata, mu, basis):
    document = {
        "version": SVD_ARTIFACT_FILE_VERSION,
        "metadata": metadata,
        "mu": mu,
        "basis": basis,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(document, temporary_path)
        os.replace(temporary_path, path)
    except Exception as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise SVDArtifactError(f"Cannot save SVD artifact {path}: {error}") from error


def load_svd_artifact(path, expected_metadata):
    try:
        document = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SVDArtifactError(f"Cannot load SVD artifact {path}: {error}") from error

    if not isinstance(document, dict):
        raise SVDArtifactError(f"SVD artifact {path} must contain a dictionary")
    if document.get("version") != SVD_ARTIFACT_FILE_VERSION:
        raise SVDArtifactError(
            f"Unsupported SVD artifact version in {path}: "
            f"{document.get('version')!r}; expected {SVD_ARTIFACT_FILE_VERSION}"
        )

    metadata = document.get("metadata")
    if metadata != expected_metadata:
        if not isinstance(metadata, dict):
            differing_fields = ["metadata"]
        else:
            differing_fields = sorted(
                key
                for key in set(metadata) | set(expected_metadata)
                if metadata.get(key) != expected_metadata.get(key)
            )
        raise SVDArtifactError(
            f"SVD artifact {path} is incompatible with its expected final index; "
            f"different metadata fields: {', '.join(differing_fields)}"
        )

    mu = document.get("mu")
    basis = document.get("basis")
    if not isinstance(mu, torch.Tensor) or not isinstance(basis, torch.Tensor):
        raise SVDArtifactError(f"SVD artifact {path} has no valid mu/basis tensors")
    if mu.device.type != "cpu" or basis.device.type != "cpu":
        raise SVDArtifactError(f"SVD artifact {path} did not load on CPU")
    if mu.dtype != torch.float32 or basis.dtype != torch.float32:
        raise SVDArtifactError(
            f"SVD artifact {path} has dtype mu={mu.dtype}, basis={basis.dtype}; "
            "expected torch.float32"
        )
    expected_basis_shape = (mu.numel(), expected_metadata["svd_dim"])
    if mu.ndim != 1 or tuple(basis.shape) != expected_basis_shape:
        raise SVDArtifactError(
            f"SVD artifact {path} has shapes mu={tuple(mu.shape)}, "
            f"basis={tuple(basis.shape)}; expected one-dimensional mu and "
            f"basis={expected_basis_shape}"
        )
    return mu, basis


def assemble_svd_output(workers, jobs):
    expected_pairs = {
        (worker, kv) for worker in workers for kv in TensorFileManager.KV
    }
    actual_pairs = set(jobs)
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs, key=lambda pair: str(pair))
        extra = sorted(actual_pairs - expected_pairs, key=lambda pair: str(pair))
        raise SVDArtifactError(
            f"SVD artifact index is incomplete: missing={missing}, extra={extra}"
        )

    output_dict = {
        "version": KVTC_FILE_VERSION,
        "keys": {worker: {"mu": None, "basis": None} for worker in workers},
        "values": {worker: {"mu": None, "basis": None} for worker in workers},
    }
    for worker in workers:
        for kv in TensorFileManager.KV:
            job = jobs[(worker, kv)]
            mu, basis = load_svd_artifact(job["path"], job["metadata"])
            section = "keys" if kv == TensorFileManager.KV.K else "values"
            output_dict[section][worker]["mu"] = mu
            output_dict[section][worker]["basis"] = basis
            logger.info(
                "Assigned SVD artifact %s to final index %s/%s",
                job["path"],
                section,
                worker,
            )

    for section in ("keys", "values"):
        for worker in workers:
            if any(
                output_dict[section][worker][name] is None
                for name in ("mu", "basis")
            ):
                raise SVDArtifactError(
                    f"Final SVD output index {section}/{worker} was not populated"
                )
    return output_dict


def select_svd_jobs(workers, jobs, cache_policy):
    if cache_policy not in ("reuse", "overwrite"):
        raise SVDArtifactError(f"Invalid SVD cache policy: {cache_policy}")

    jobs_to_run = []
    reused_jobs = 0
    for worker in workers:
        for kv in TensorFileManager.KV:
            job = jobs[(worker, kv)]
            artifact_path = job["path"]
            if artifact_path.exists() and not artifact_path.is_file():
                raise SVDArtifactError(
                    f"SVD artifact path exists but is not a file: {artifact_path}"
                )
            if cache_policy == "reuse" and artifact_path.is_file():
                logger.info(
                    "Skipping SVD for worker=%s kv=%s: artifact already exists at %s",
                    worker,
                    kv.name,
                    artifact_path,
                )
                reused_jobs += 1
                continue
            if artifact_path.is_file():
                logger.info(
                    "Overwriting SVD artifact for worker=%s kv=%s at %s",
                    worker,
                    kv.name,
                    artifact_path,
                )
            jobs_to_run.append((worker, kv, job))

    logger.info(
        "SVD jobs: %d reused, %d scheduled, %d total",
        reused_jobs,
        len(jobs_to_run),
        len(jobs),
    )
    return jobs_to_run


def SVD(
    tensor_manager: TensorFileManager,
    selection_store: TokenSelectionStore,
    svd_dim: int,
    svd_iter: int,
    kv: TensorFileManager.KV,
    undo_rope: bool,
):
    logger.info(
        f"START SVD for data at {tensor_manager.datasets_list}/{kv}* q={svd_dim} iter={svd_iter}"
    )

    if undo_rope:
        logger.info(f"Load tensors, undo rope and sample")
    else:
        logger.info(f"Load tensors and sample")

    sampled_data = []

    for kv_cache_paths in tensor_manager.datasets_list:
        for paths in tensor_manager.datasets[kv_cache_paths][kv]:
            tensor, token_count = load_tensor(paths)
            if tensor is None or token_count is None:
                continue

            if token_count < TensorFileManager.Sequence.SHORT:
                logger.warning(
                    "Skipping %s: only %s tokens after reconstruction",
                    paths[0],
                    token_count,
                )
                continue

            try:
                if undo_rope:
                    tensor = Rope.invert_rope(tensor)
                request_id = paths[0].name.split(kv.value, 1)[0]
                token_indices = selection_store.get(
                    kv_cache_paths, request_id, token_count
                )
                sampled_data.append(sample_tokens(tensor, token_indices))
            except TokenSelectionError:
                raise
            except (AssertionError, RuntimeError, ValueError) as error:
                logger.warning("Skipping %s: %s", paths[0], error)

    pprint.pp([d.shape for d in sampled_data])

    if not sampled_data:
        raise RuntimeError(f"No valid {kv} dump requests remain for SVD")

    input_tensor = transform_tensors(sampled_data)

    n, p = input_tensor.shape
    dtype = input_tensor.dtype
    logger.info(f"SVD input tensor shape (n={n})x(p={p}). Dtype {dtype}")

    per_feature_mean = input_tensor.mean(dim=0)

    logger.info(f"Calculate lowrank SVD for q={svd_dim} iter={svd_iter}")
    U, S, Vh = torch.svd_lowrank(
        input_tensor, q=svd_dim, niter=svd_iter, M=per_feature_mean
    )

    logger.info(f"DONE SVD for data at")

    return per_feature_mean, U, S, Vh


def run_svd_job(
    tensor_manager: TensorFileManager,
    selection_store: TokenSelectionStore,
    worker: str,
    kv: TensorFileManager.KV,
    svd_dim: int,
    svd_iter: int,
    artifact_path: Path,
    artifact_temporary_path: Path,
    artifact_metadata: dict,
):
    mu, U, S, V = SVD(
        tensor_manager,
        selection_store,
        svd_dim,
        svd_iter,
        kv,
        undo_rope=kv == TensorFileManager.KV.K,
    )
    logger.info(
        "%s/%s: mu=%s U=%s S=%s V=%s",
        worker,
        kv,
        mu.shape,
        U.shape,
        S.shape,
        V.shape,
    )

    # U and S are not part of the calibration output and can be large. Release
    # them before persisting the tensors that are needed by the final object.
    del U, S
    save_svd_artifact(
        artifact_path, artifact_temporary_path, artifact_metadata, mu, V
    )
    logger.info(
        "Saved SVD artifact for worker=%s kv=%s at %s",
        worker,
        kv.name,
        artifact_path,
    )

    # Futures must never retain output tensors while other SVD jobs are running.
    del mu, V
    return worker, kv, artifact_path


def init_logger(log_dir, filename, log_level):
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    if log_level not in levels:
        raise ValueError(f"Invalid log level: {log_level}")

    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path = log_dir / filename

    logging.basicConfig(
        level=levels[log_level],
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)],
    )

    logger.info(f"Logging to {log_file_path}")


def discover_dump_directories(input_dir: Path) -> tuple[list[Path], list[str]]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    logger.info("Scanning %s for calibration dump directories", input_dir)
    dump_dirs = []
    worker_sets = {}
    for candidate in sorted(input_dir.iterdir()):
        if not candidate.is_dir():
            continue

        workers = sorted(
            (
                entry.name
                for entry in candidate.iterdir()
                if entry.is_dir() and WORKER_DIR_PATTERN.fullmatch(entry.name)
            ),
            key=lambda worker: tuple(map(int, WORKER_DIR_PATTERN.fullmatch(worker).groups())),
        )
        if not workers:
            logger.debug("Skipping %s: no tp_<X>_pp_<Y> worker directories", candidate)
            continue

        dump_dirs.append(candidate)
        worker_sets[candidate] = workers
        logger.info("Discovered dump directory %s with workers: %s", candidate, workers)

    if not dump_dirs:
        raise ValueError(
            f"No dump directories with tp_<X>_pp_<Y> workers found in {input_dir}"
        )

    workers = worker_sets[dump_dirs[0]]
    for dump_dir in dump_dirs[1:]:
        if worker_sets[dump_dir] != workers:
            raise ValueError(
                "Dump directories must have the same worker set. "
                f"Expected {workers} from {dump_dirs[0]}, but found "
                f"{worker_sets[dump_dir]} in {dump_dir}."
            )

    logger.info(
        "Using %d dump directories and %d workers: %s",
        len(dump_dirs),
        len(workers),
        workers,
    )
    return dump_dirs, workers


def create_argument_parser():
    parser = argparse.ArgumentParser(
        usage=(
            f"\n{os.path.basename(__file__)}"
            " -N <token_sample_count> --niter <the number of subspace iterations for svd_lowrank>"
            " -q <a slightly overestimated rank of svd matrix>"
            " -i <dump-parent-directory>"
            " -o <output_path> -m <model_path>"
            " --svd-cache-policy <reuse|overwrite>"
            " [--save-selected-tokens <path> | --load-selected-tokens <path> |"
            " --selected-tokens-cache <path>]\n"
        )
    )
    parser.add_argument(
        "--kvtc-version",
        action="version",
        version=KVTC_FILE_VERSION
    )
    parser.add_argument(
        "-N",
        "--sample-tokens",
        type=int,
        required=True,
        help="The total number of tokens to sample from the calibration dataset (default=200,000)",
    )
    parser.add_argument(
        "--niter",
        required=True,
        help="Please refer to 'niter' in torch.svd_lowrank in pytorch documentation",
    )
    parser.add_argument(
        "-q",
        "--svd_dim",
        required=True,
        help="Please refer to 'q' in torch.svd_lowrank in pytorch documentation",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Parent directory containing calibration dump directories. Tokens are sampled across all discovered dump directories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Calibration file output",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Calibration file output",
    )
    parser.add_argument(
        "-m", "--model-dir", required=True, help="Path to the target model directory"
    )
    parser.add_argument("-log", "--log-level", required=False, default="info")
    parser.add_argument(
        "-s",
        "--sampling-policy",
        default="strict",
        choices=list(str(p.name).lower() for p in TensorFileManager.SamplingPolicy),
    )
    parser.add_argument(
        "--svd-cache-policy",
        required=True,
        choices=("reuse", "overwrite"),
        help=(
            "Reuse existing compatible per-worker SVD artifacts and calculate only "
            "missing jobs, or overwrite all artifacts for this run. Artifacts are "
            "stored under <output>.svd-artifacts"
        ),
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--save-selected-tokens",
        metavar="PATH",
        help="Save the global token positions selected during this run as JSON",
    )
    selection_group.add_argument(
        "--load-selected-tokens",
        metavar="PATH",
        help="Load and reuse global token positions from a previous run",
    )
    selection_group.add_argument(
        "--selected-tokens-cache",
        metavar="PATH",
        help=(
            "Load global token positions when PATH exists; otherwise select tokens "
            "and save them to PATH"
        ),
    )
    return parser


def run():
    parser = create_argument_parser()

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    N = args.sample_tokens
    svd_iter = int(args.niter)
    svd_dim = int(args.svd_dim)
    log_level = args.log_level
    sampling_policy = TensorFileManager.SamplingPolicy[args.sampling_policy.upper()]

    init_logger(
        log_dir,
        f"svd-q{svd_dim}_iter{svd_iter}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log",
        log_level,
    )

    input_dir_list, workers = discover_dump_directories(input_dir)
    svd_workers = min(SVD_WORKERS, len(workers) * len(TensorFileManager.KV))

    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    torch.set_num_threads(available_cpus)

    tensor_managers = {
        worker: TensorFileManager(input_dir_list, worker, sampling_policy)
        for worker in workers
    }
    global_sampling_requests = build_global_sampling_requests(
        tensor_managers, workers, N
    )

    cache_metadata = build_svd_cache_metadata(
        input_dir=input_dir,
        model_dir=Path(args.model_dir),
        dump_directories=input_dir_list,
        workers=workers,
        sampling_requests=global_sampling_requests,
        sample_tokens=N,
        sampling_policy=sampling_policy,
        svd_dim=svd_dim,
        svd_iter=svd_iter,
    )
    svd_cache_directory = build_svd_cache_directory(output_path, cache_metadata)

    selection_mode = None
    selection_path = None
    if args.save_selected_tokens:
        selection_mode = "save"
        selection_path = Path(args.save_selected_tokens)
    elif args.load_selected_tokens:
        selection_mode = "load"
        selection_path = Path(args.load_selected_tokens)
    elif args.selected_tokens_cache:
        selection_path = Path(args.selected_tokens_cache)
        selection_mode = "load" if selection_path.exists() else "save"
    else:
        selection_path = svd_cache_directory / "selected-tokens.json"
        selection_mode = "load" if selection_path.exists() else "save"

    logger.info(
        "Token selection cache %s; %s selections at %s",
        "exists" if selection_path.exists() else "does not exist",
        "loading" if selection_mode == "load" else "saving new",
        selection_path,
    )
    if selection_path.absolute() == output_path.absolute():
        parser.error("The token selection path must differ from --output")

    selection_store = TokenSelectionStore(
        mode=selection_mode,
        path=selection_path,
        input_dir=input_dir,
        model_dir=Path(args.model_dir),
        sample_tokens=N,
        sampling_policy=sampling_policy,
        dump_directories=input_dir_list,
        workers=workers,
    )
    # Keep selection finalization and persistence ahead of all SVD worker creation.
    selection_store.finalize(global_sampling_requests)
    selection_digest = selection_store.digest()
    artifact_directory = svd_cache_directory / f"selection-{selection_digest}"
    logger.info(
        "SVD artifact directory for current sampling/SVD parameters: %s",
        artifact_directory,
    )

    Rope.load_model_config(args.model_dir)

    # This is the authoritative mapping between an SVD artifact and its final
    # output index. Never infer this mapping from completion or directory order.
    jobs = {}
    for worker in workers:
        for kv in TensorFileManager.KV:
            pair = (worker, kv)
            metadata = build_svd_job_metadata(
                cache_metadata, selection_digest, worker, kv
            )
            artifact_path = artifact_directory / f"{worker}-{kv.name}.pt"
            temporary_path = artifact_path.with_name(
                f".{artifact_path.name}.tmp-{os.getpid()}"
            )
            if pair in jobs:
                raise SVDArtifactError(f"Duplicate SVD job index: {pair}")
            jobs[pair] = {
                "path": artifact_path,
                "temporary_path": temporary_path,
                "metadata": metadata,
            }

    logger.info(
        f"-------------------- model={args.model_dir} N={N} q={svd_dim} iter={svd_iter} --------------------"
    )
    logger.info(
        "Running up to %d SVD jobs concurrently with up to %d PyTorch intra-op threads",
        svd_workers,
        available_cpus,
    )

    jobs_to_run = select_svd_jobs(workers, jobs, args.svd_cache_policy)

    if jobs_to_run:
        with ThreadPoolExecutor(
            max_workers=svd_workers, thread_name_prefix="svd"
        ) as executor:
            future_jobs = {
                executor.submit(
                    run_svd_job,
                    tensor_managers[worker],
                    selection_store,
                    worker,
                    kv,
                    svd_dim,
                    svd_iter,
                    job["path"],
                    job["temporary_path"],
                    job["metadata"],
                ): (worker, kv, job["path"])
                for worker, kv, job in jobs_to_run
            }

            completed_jobs = 0
            try:
                for future in as_completed(future_jobs):
                    expected_worker, expected_kv, expected_path = future_jobs[future]
                    worker, kv, artifact_path = future.result()
                    if (
                        worker != expected_worker
                        or kv != expected_kv
                        or artifact_path != expected_path
                    ):
                        raise SVDArtifactError(
                            "SVD future returned a result for the wrong final index: "
                            f"expected {(expected_worker, expected_kv, expected_path)}, "
                            f"got {(worker, kv, artifact_path)}"
                        )
                    completed_jobs += 1
            except Exception:
                for future in future_jobs:
                    future.cancel()
                logger.exception(
                    "SVD calibration failed after confirming %d of %d scheduled "
                    "artifacts; rerun with --svd-cache-policy reuse to resume",
                    completed_jobs,
                    len(jobs_to_run),
                )
                raise

    missing_artifacts = [
        job["path"] for job in jobs.values() if not job["path"].is_file()
    ]
    if missing_artifacts:
        raise SVDArtifactError(
            f"Cannot assemble final output; missing SVD artifacts: {missing_artifacts}"
        )

    # Load output matrices only after every SVD artifact has been published and
    # all SVD scratch tensors have been released.
    output_dict = assemble_svd_output(workers, jobs)
    torch.save(output_dict, output_path)


if __name__ == "__main__":
    run()
