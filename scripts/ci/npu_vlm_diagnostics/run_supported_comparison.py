"""Fresh-process public API comparisons, followed by all nine original tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("artifacts/npu-vlm").resolve()
HERE = Path(__file__).resolve().parent
CASE = "test/registered/npu/vlm_models/test_npu_vlm_input_format.py"
MODES = ("default", "gmm", "nz", "addmm", "deterministic", "high_precision")
report = {"operators": {}, "raw_cases": {}, "selected": None}
base_env = dict(os.environ)
for key in ("VLM_CANN_ROUTE_BUDGET", "VLM_DEVICE_LIMIT_CUBE", "VLM_DEVICE_LIMIT_VECTOR"):
    if base_env.get(key):
        raise RuntimeError(f"Prohibited experiment environment: {key}")
base_env["VLM_REQUIRE_24_CORES"] = "1"
base_env["PYTHONPATH"] = str(HERE / "hooks") + os.pathsep + base_env.get("PYTHONPATH", "")


def run(name, args, mode, timeout, diagnostics=False):
    dest = ROOT / name
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(base_env, VLM_SUPPORTED_MATMUL=mode, ASCEND_GLOBAL_LOG_LEVEL="0")
    if diagnostics:
        env["VLM_DIAGNOSTICS_DIR"] = str(dest / "snapshots")
    else:
        env.pop("VLM_DIAGNOSTICS_DIR", None)
    with (dest / "console.log").open("w") as log:
        code = subprocess.call(["timeout", "--kill-after=30s", f"{timeout}s", sys.executable, "-u", *args], env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {"returncode": code}
    result_file = dest / "result.json"
    if result_file.exists():
        data = json.loads(result_file.read_text())
        result.update({k: data[k] for k in ("successful", "tests_run", "expected", "skipped") if k in data})
    print("NPU_VLM_SUPPORTED_PHASE", name, json.dumps(result), flush=True)
    return result


def save():
    (ROOT / "supported-summary.json").write_text(json.dumps(report, indent=2))


from modelscope import snapshot_download
model = Path("/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct")
if not (model / "model.safetensors.index.json").exists():
    model = Path(snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct"))
for mode in MODES:
    name = "supported-operators/" + mode
    args = [str(HERE / "replay_supported_matmul.py"), "--source", str(ROOT / "qwen/snapshots"), "--model", str(model), "--output", str(ROOT / name), "--candidate", mode, "--profile", "--save-output"]
    if mode == "default":
        args.append("--require-match")
    report["operators"][mode] = run(name, args, mode, 300)
    save()
report["k_order"] = run("k-order", [str(HERE / "replay_k_order.py"), "--source", str(ROOT / "qwen/snapshots"), "--model", str(model), "--output", str(ROOT / "k-order")], "default", 600)
save()
for mode in MODES[1:]:
    if report["operators"][mode]["returncode"] != 0:
        continue
    name = "supported-raw/" + mode
    report["raw_cases"][mode] = run(name, [str(HERE / "run_case.py"), CASE, "--mode", "qwen", "--output-dir", str(ROOT / name)], mode, 600, True)
    save()
    if report["raw_cases"][mode]["returncode"] == 0:
        report["selected"] = mode
        break
# Even if all raw candidates fail, run the entire file with the first valid public API.
selected = report["selected"] or next((m for m in MODES[1:] if report["operators"][m]["returncode"] == 0), "default")
report["full_mode"] = selected
report["full"] = run("supported-full", [str(HERE / "run_case.py"), CASE, "--mode", "full", "--output-dir", str(ROOT / "supported-full")], selected, 2100, True)
save()
full = report["full"]
if not (full["returncode"] == 0 and full.get("successful") and full.get("tests_run") == 9 and full.get("expected") == 9 and not full.get("skipped") and report["operators"]["default"]["returncode"] == 0 and report["k_order"]["returncode"] == 0):
    raise RuntimeError("Public API candidate did not satisfy all nine original assertions and evidence calibration")
