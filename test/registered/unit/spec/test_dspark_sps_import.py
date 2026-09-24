"""Client-only SPS import regression; no model, network, CUDA or NPU required."""

import ast
import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
REL = Path("python/sglang/benchmark/dspark_sps_profiler.py")
FIXED = REPO / REL


def run_python(code, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "python")
    env["HF_HUB_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_profiler_reaches_server_query_without_dataset_imports(tmp_path):
    code = r"""
        import importlib.abc
        import runpy
        import sys

        class BlockServerBenchmark(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "sglang.benchmark.one_batch_server":
                    raise ImportError("cannot import name 'MMMUDataset': simulated circular import")
                if fullname.startswith("sglang.benchmark.datasets"):
                    raise AssertionError("SPS must not import the dataset registry")

        sys.meta_path.insert(0, BlockServerBenchmark())
        ns = runpy.run_path(sys.argv[1])
        assert callable(ns["get_tokenizer"])
        assert ns["DEFAULT_TIMEOUT"] == 600
        class ReachedServer(Exception):
            pass
        def fetch(**kwargs):
            assert kwargs["base_url"] == "http://127.0.0.1:8030"
            assert kwargs["allowed_modes"] == ("compact", "cap-accept")
            raise ReachedServer()
        ns["run_profile"].__globals__["fetch_server_context"] = fetch
        try:
            ns["run_profile"](
                base_url="http://127.0.0.1:8030", batch_sizes=[1, 2, 4],
                settings=None, out=sys.argv[3], repeats=3,
                local_tokenizer_path=None, fracs=[0.25, 1.0],
            )
        except ReachedServer:
            pass
        else:
            raise AssertionError("did not enter the collection path")
        assert "sglang.benchmark.one_batch_server" not in sys.modules
        assert "sglang.benchmark.datasets" not in sys.modules
    """
    fixed = run_python(code, FIXED, "fixed", tmp_path / "fixed.json")
    assert "using a torch-free fallback" not in fixed.stderr


def test_capacity_checks_preserve_original_boundaries():
    source = REPO / "python/sglang/benchmark/one_batch_server.py"
    names = {
        "should_skip_due_to_token_capacity",
        "should_skip_due_to_max_running_requests",
    }
    nodes = [
        n
        for n in ast.parse(source.read_text()).body
        if getattr(n, "name", None) in names
    ]
    original = {}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), original
    )
    spec = importlib.util.spec_from_file_location("fixed_sps_profiler", FIXED)
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    with contextlib.redirect_stdout(io.StringIO()):
        for threshold in (0, 1, 4, float("inf")):
            for bs in (0, 1, 4, 5):
                fn = "should_skip_due_to_max_running_requests"
                assert getattr(fixed, fn)(bs, threshold) == original[fn](bs, threshold)
        for threshold in (0, 4, 8, 9, float("inf")):
            for bs, inp, out in ((1, 4, 4), (2, 2, 2), (3, 1, 2)):
                fn = "should_skip_due_to_token_capacity"
                assert getattr(fixed, fn)(bs, inp, out, threshold) == original[fn](
                    bs, inp, out, threshold
                )


def test_fit_only_fallback_survives_missing_runtime(tmp_path):
    import shutil

    copy = tmp_path / REL
    copy.parent.mkdir(parents=True)
    shutil.copyfile(FIXED, copy)
    table_rel = Path("python/sglang/srt/speculative/dspark_components/dspark_sps.py")
    table = tmp_path / table_rel
    table.parent.mkdir(parents=True)
    shutil.copyfile(REPO / table_rel, table)
    result = run_python(
        r"""
        import importlib.abc
        import runpy
        import sys
        class NoRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "sglang":
                    raise ImportError("runtime intentionally unavailable")
        sys.meta_path.insert(0, NoRuntime())
        ns = runpy.run_path(sys.argv[1])
        assert ns["get_tokenizer"] is None
        import json
        from pathlib import Path
        out = Path(sys.argv[1]).parent / "synthetic_fit.json"
        rows = [{"batch_tokens": 4, "steps_per_sec": 25.0},
                {"batch_tokens": 8, "steps_per_sec": 12.5}]
        out.with_name("synthetic_fit.rounds.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        ns["fit_profile"](out=str(out), max_batch_tokens=None, self_check=True, plot=False)
        table = ns["load_sps_table_from_path"](str(out))
        assert table.lookup(4) == 25.0
        assert table.lookup(8) == 12.5
        """,
        copy,
    )
    assert "runtime intentionally unavailable" in result.stderr
