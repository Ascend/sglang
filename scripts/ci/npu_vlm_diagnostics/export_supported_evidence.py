"""Keep responses, provenance, kernel identity and small hidden-state evidence."""
import json
from pathlib import Path
import re
import shutil
import sys
root = Path(sys.argv[1])
out = root.with_name(root.name + "-supported")
out.mkdir(parents=True, exist_ok=True)
for path in root.rglob("*"):
    if not path.is_file():
        continue
    relative = path.relative_to(root)
    if "snapshots" in relative.parts:
        if path.suffix != ".json":
            continue
        record = json.loads(path.read_text())
        if record.get("stage") not in {"lm_head_projection.output", "runtime.fingerprint", "decoder.loaded_weights", "gate_up.operator.input", "gate_up.operator.output"}:
            continue
        if record.get("stage") == "lm_head_projection.output" and record.get("tensor_file"):
            tensor = path.parent / record["tensor_file"]
            target = out / tensor.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tensor, target)
    elif not (path.suffix in {".json", ".txt", ".log"} or path.name == "kernel_details.csv"):
        continue
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
for runtime in (root / "supported-operators").glob("*/runtime.json"):
    pid = str(json.loads(runtime.read_text())["pid"])
    target = out / runtime.parent.relative_to(root)
    for path in Path("/root/ascend/log").glob("*/plog/*"):
        if path.is_file() and re.search(rf"(?:^|[-_]){pid}(?:[-_.]|$)", path.name):
            shutil.copy2(path, target / (path.parent.parent.name + "-" + path.name))
