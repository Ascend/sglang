"""Run the original VLM cases, retaining assertions and recording provenance."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import subprocess
import sys
import time
import unittest
from pathlib import Path


def file_info(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "missing": True}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def provenance(case, utils, root):
    import sglang
    from PIL import Image

    package = Path(sglang.__file__).resolve().parent
    files = [case, Path(utils.__file__)]
    files += [
        package / name
        for name in (
            "srt/multimodal/processors/base_processor.py",
            "srt/multimodal/processors/qwen_vl.py",
            "srt/hardware_backend/npu/modules/qwen_vl_processor.py",
            "srt/models/qwen2_5_vl.py",
            "srt/managers/tokenizer_manager.py",
            "srt/layers/rotary_embedding.py",
            "srt/layers/rotary_embedding/mrope.py",
            "srt/layers/rotary_embedding/__init__.py",
            "srt/layers/rotary_embedding/base.py",
            "srt/utils/common.py",
        )
    ]
    models = {}
    for name in (
        "GEMMA_3_4B_IT_WEIGHTS_PATH",
        "KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH",
        "QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH",
    ):
        model = Path(getattr(utils, name))
        config_names = (
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tiktoken.model",
            "generation_config.json",
            "model.safetensors.index.json",
            "configuration_kimi_vl.py",
            "image_processing_kimi_vl.py",
            "processing_kimi_vl.py",
            "modeling_kimi_vl.py",
        )
        models[name] = {
            "path": str(model),
            "configs": [file_info(model / n) for n in config_names],
            "shards": [
                {"name": p.name, "bytes": p.stat().st_size}
                for p in sorted(model.glob("*.safetensors"))
            ],
        }
    images = []
    for name in ("IMAGE_MAN_IRONING_PATH", "IMAGE_SGL_LOGO_PATH"):
        path = getattr(utils, name)
        item = file_info(path)
        if not item.get("missing"):
            with Image.open(path) as image:
                item.update(
                    size=list(image.size),
                    mode=image.mode,
                    pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                )
            # Preserve the actual cached fixtures; never silently replace them.
            import shutil

            (root / "images").mkdir(exist_ok=True)
            shutil.copyfile(path, root / "images" / Path(path).name)
        images.append(item)
    versions = {}
    for name in (
        "sglang",
        "torch",
        "torch_npu",
        "transformers",
        "torchvision",
        "Pillow",
        "numpy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    repo = package.parent.parent
    git = {}
    for name, arguments in (
        ("head", ["rev-parse", "HEAD"]),
        ("dirty", ["status", "--porcelain", "--", "python/sglang/srt"]),
    ):
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments], capture_output=True, text=True
        )
        git[name] = (
            result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        )
    selected_env = {
        name: os.environ.get(name)
        for name in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "SGLANG_USE_MODELSCOPE",
            "SGLANG_IS_IN_CI",
            "SGLANG_SET_CPU_AFFINITY",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK",
            "PYTORCH_NPU_ALLOC_CONF",
            "STREAMS_PER_DEVICE",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "HF_ENDPOINT",
            "VLM_DIAGNOSTICS_DIR",
        )
    }
    manifest = {
        "sglang_file": str(package / "__init__.py"),
        "git": git,
        "versions": versions,
        "files": [file_info(p) for p in files],
        "models": models,
        "images": images,
        "env": selected_env,
    }
    (root / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print("NPU_VLM_PROVENANCE " + json.dumps(manifest, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--mode", choices=("full", "qwen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for model in ("gemma", "kimi", "qwen"):
        parser.add_argument(f"--{model}-model")
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    import sglang.test.ascend.test_ascend_utils as utils

    for model, constant in (
        ("gemma", "GEMMA_3_4B_IT_WEIGHTS_PATH"),
        ("kimi", "KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH"),
        ("qwen", "QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH"),
    ):
        if getattr(args, f"{model}_model"):
            setattr(utils, constant, getattr(args, f"{model}_model"))
    if args.image_dir:
        utils.IMAGE_MAN_IRONING_PATH = str(
            args.image_dir / "man_ironing_on_back_of_suv.png"
        )
        utils.IMAGE_SGL_LOGO_PATH = str(args.image_dir / "sgl_logo.png")
    provenance(args.case, utils, root)
    namespace = runpy.run_path(str(args.case), run_name="vlm_diagnostic_case")
    base = namespace["VLMInputTestBase"]
    original_verify = base.verify_response
    original_processor = base.get_processor_output
    responses = []
    if os.environ.get("VLM_DIAGNOSTICS_DIR"):
        from npu_vlm_diagnostic_hooks import emit
    else:

        def emit(*_args, **_kwargs):
            pass

    def processor_with_record(self, *positional, **keyword):
        result = original_processor(self, *positional, **keyword)
        if type(self).__name__ == "TestQwenVLUnderstandsImage":
            emit(
                "hf_reference",
                {"test": self.id(), "inputs": result[0], "prompt": result[1]},
            )
        return result

    def verify_with_record(self, output):
        record = {"test": self.id(), "output": output}
        responses.append(record)
        (root / "responses.json").write_text(
            json.dumps(responses, ensure_ascii=False, indent=2)
        )
        print("NPU_VLM_RESPONSE " + json.dumps(record, ensure_ascii=False), flush=True)
        try:
            return original_verify(self, output)
        except AssertionError:
            if (
                os.environ.get("VLM_DIAGNOSTICS_DIR")
                and type(self).__name__ == "TestQwenVLUnderstandsImage"
                and self._testMethodName == "test_accepts_image"
            ):
                try:
                    self.get_processor_output()
                except Exception as error:
                    print(f"NPU_VLM_REFERENCE_ERROR {error!r}", flush=True)
            raise

    base.get_processor_output = processor_with_record
    base.verify_response = verify_with_record
    if args.mode == "qwen":
        suite = unittest.TestSuite(
            [namespace["TestQwenVLUnderstandsImage"]("test_accepts_image")]
        )
        expected = 1
    else:
        names = sorted(
            name
            for name, cls in namespace.items()
            if isinstance(cls, type)
            and issubclass(cls, unittest.TestCase)
            and cls.__module__ == "vlm_diagnostic_case"
        )
        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromTestCase(namespace[name])
            for name in names
        )
        expected = 9
    if suite.countTestCases() != expected:
        raise RuntimeError(f"Expected {expected} cases, got {suite.countTestCases()}")
    start = time.time()
    result = unittest.TextTestRunner(verbosity=2, failfast=False).run(suite)
    summary = {
        "mode": args.mode,
        "elapsed": time.time() - start,
        "successful": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "expected": expected,
        "failures": [(str(t), trace) for t, trace in result.failures],
        "errors": [(str(t), trace) for t, trace in result.errors],
        "skipped": [(str(t), reason) for t, reason in result.skipped],
    }
    (root / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(
        0
        if result.wasSuccessful() and result.testsRun == expected and not result.skipped
        else 1
    )


if __name__ == "__main__":
    main()
