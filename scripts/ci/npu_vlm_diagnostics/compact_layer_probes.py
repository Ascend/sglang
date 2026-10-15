"""Export tails and first-layer full attention inputs from immutable CI probes."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--full-attention-layers", default="0")
args = parser.parse_args()
full_layers = set(map(int, args.full_attention_layers.split(",")))
args.output.mkdir(parents=True, exist_ok=True)
destination = args.output / "layer-probes"
destination.mkdir(exist_ok=True)
slices = {}
for path in args.source.iterdir():
    target = destination / path.name
    if (
        path.suffix == ".pt"
        and "-f0-" in path.name
        and ".self_attn.attn-input.pt" in path.name
        and not any(f"-layers.{layer}." in path.name for layer in full_layers)
    ):
        tensors = torch.load(path, weights_only=True, map_location="cpu")
        original = {key: list(value.shape) for key, value in tensors.items()}
        tensors = {
            key: value[-1:].clone() if value.ndim and value.shape[0] > 1 else value
            for key, value in tensors.items()
        }
        torch.save(tensors, target)
        slices[path.name] = original
    else:
        shutil.copyfile(path, target)
(destination / "compaction.json").write_text(json.dumps(slices, indent=2))
with tarfile.open(args.output / "layer-probes-compact.tar.gz", "w:gz") as archive:
    archive.add(destination, arcname="layer-probes")
print(
    "compact bytes",
    (args.output / "layer-probes-compact.tar.gz").stat().st_size,
    flush=True,
)
shutil.rmtree(destination)
