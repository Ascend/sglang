"""Replay captured vision output through the original precomputed-input helper."""

import argparse
import hashlib
import json
import os
import runpy
import sys
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-model")
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    import sglang.test.ascend.test_ascend_utils as utils
    import torch
    from run_case import provenance

    if args.qwen_model:
        utils.QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH = args.qwen_model
    if args.image_dir:
        utils.IMAGE_MAN_IRONING_PATH = str(
            args.image_dir / "man_ironing_on_back_of_suv.png"
        )
        utils.IMAGE_SGL_LOGO_PATH = str(args.image_dir / "sgl_logo.png")
    provenance(args.case, utils, root)
    records = [
        json.loads(p.read_text())
        for p in (args.source_dir / "snapshots").glob("*.json")
    ]
    captured = next(
        d for d in records if d["stage"] == "vision.output" and d["rank"] == 0
    )
    tensor_path = args.source_dir / "snapshots" / captured["tensor_file"]
    embedding = torch.load(tensor_path, weights_only=True, map_location="cpu")[
        "value.result"
    ]
    digest = hashlib.sha256(
        embedding.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    if digest != captured["payload"]["result"]["sha256"]:
        raise RuntimeError(
            "Captured vision tensor hash differs from its diagnostic record"
        )
    namespace = runpy.run_path(str(args.case), run_name="embedding_replay_source")
    original = namespace["TestQwenVLUnderstandsImage"]
    responses = []

    class CapturedEmbeddingReplay(original):
        async def test_captured_embedding(self):
            inputs, _ = self.get_processor_output()
            output = await self.engine.async_generate(
                input_ids=inputs["input_ids"][0].tolist(),
                image_data=[
                    self._precomputed_image_data(inputs, embedding.to(self.device))
                ],
                sampling_params=dict(temperature=0.0, max_new_tokens=512),
            )
            responses.append(
                {
                    "test": self.id(),
                    "source_rid": captured["rid"],
                    "embedding_sha256": digest,
                    "output": output,
                }
            )
            (root / "responses.json").write_text(
                json.dumps(responses, ensure_ascii=False, indent=2)
            )
            print(
                "NPU_VLM_RESPONSE " + json.dumps(responses[-1], ensure_ascii=False),
                flush=True,
            )
            self.verify_response(output)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite([CapturedEmbeddingReplay("test_captured_embedding")])
    )
    summary = {
        "successful": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": [(str(t), trace) for t, trace in result.failures],
        "errors": [(str(t), trace) for t, trace in result.errors],
        "skipped": [(str(t), reason) for t, reason in result.skipped],
    }
    if os.environ.get("VLM_DIAGNOSTICS_DIR"):
        records = [
            json.loads(p.read_text())
            for p in Path(os.environ["VLM_DIAGNOSTICS_DIR"]).glob("*.json")
        ]
        required = {
            "embedding_scatter.output",
            "decoder.loaded_weights",
            "model_forward.input",
            "model_forward.output",
            "decoder_layer.output",
            "decoder_component.output",
        }
        missing = {
            str(rank): sorted(
                required - {d["stage"] for d in records if d["rank"] == rank}
            )
            for rank in (0, 1)
        }
        summary["diagnostics"] = {
            "complete": not any(missing.values()),
            "per_rank_missing": missing,
        }
        print("NPU_VLM_COVERAGE " + json.dumps(summary["diagnostics"]), flush=True)
    (root / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(
        0
        if result.wasSuccessful()
        and result.testsRun == 1
        and summary.get("diagnostics", {}).get("complete", True)
        else 1
    )


if __name__ == "__main__":
    main()
