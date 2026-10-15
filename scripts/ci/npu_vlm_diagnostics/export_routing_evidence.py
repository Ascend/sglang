"""Export small model/hidden-state evidence before the full diagnostic artifact."""

import csv
import json
import re
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = root.with_name(root.name + "-routing")
out.mkdir(parents=True, exist_ok=True)
for phase in ("baseline", "qwen", "replay", "route-replay", "route-raw"):
    source, target = root / phase, out / phase
    target.mkdir(exist_ok=True)
    for name in ("console.log", "result.json"):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    snapshots = target / "snapshots"
    snapshots.mkdir(exist_ok=True)
    for path in (source / "snapshots").glob("*.json"):
        record = json.loads(path.read_text())
        if record["stage"] not in {
            "lm_head_projection.output",
            "decoder.loaded_weights",
            "runtime.fingerprint",
        }:
            continue
        shutil.copy2(path, snapshots / path.name)
        if record.get("tensor_file"):
            shutil.copy2(
                path.parent / record["tensor_file"], snapshots / record["tensor_file"]
            )
    layers = target / "layer-boundaries"
    layers.mkdir(exist_ok=True)
    for path in (source / "layer-probes").glob("*.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        selected = [
            r for r in records if re.fullmatch(r"layers\.\d+|final_norm", r["module"])
        ]
        (layers / path.name).write_text("".join(json.dumps(r) + "\n" for r in selected))
        for record in selected:
            shutil.copy2(path.parent / record["file"], layers / record["file"])
for source in (root / "routing-operators").glob("*"):
    if not source.is_dir():
        continue
    target = out / "operators" / source.name
    target.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "runtime.json", "console.log"):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    runtime = source / "runtime.json"
    if runtime.exists():
        pid = str(json.loads(runtime.read_text())["pid"])
        for path in Path("/root/ascend/log").glob("*/plog/*"):
            if path.is_file() and re.search(rf"(?:^|[-_]){pid}(?:[-_.]|$)", path.name):
                shutil.copy2(path, target / (path.parent.parent.name + "-" + path.name))
    for path in source.rglob("kernel_details.csv"):
        for row in csv.DictReader(path.open()):
            print("VLM_PROFILE_KERNEL", json.dumps({"mode": source.name, "row": row}))
missing = [
    phase
    for phase in ("qwen", "replay", "route-replay", "route-raw")
    if not (out / phase / "result.json").exists()
]
(out / "export.json").write_text(json.dumps({"missing_results": missing}, indent=2))
if missing:
    raise RuntimeError(f"Missing experiment results: {missing}")
