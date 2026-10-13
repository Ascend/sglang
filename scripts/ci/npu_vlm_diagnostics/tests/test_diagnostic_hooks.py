import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import npu_vlm_diagnostic_hooks as hooks


class TestDiagnosticHooks(CustomTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, VLM_DIAGNOSTICS_DIR=self.temp.name)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_tensor_snapshots_preserve_scalar_bfloat16_and_noncontiguous_values(self):
        values = {
            "scalar": torch.tensor(3, device="cpu"),
            "bf16": torch.arange(12, dtype=torch.bfloat16, device="cpu")
            .reshape(3, 4)
            .t(),
        }
        before = {key: value.clone() for key, value in values.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            hooks.emit("tensor", values)
        root = Path(self.temp.name)
        record = json.loads(next(root.glob("*.json")).read_text())
        saved = torch.load(root / record["tensor_file"], weights_only=True)
        for key, value in values.items():
            torch.testing.assert_close(value, before[key])
            torch.testing.assert_close(saved[f"value.{key}"], value)
        self.assertEqual(record["payload"]["scalar"]["values"], 3)

    def test_sync_wrapper_preserves_return_identity_and_original_exception(self):
        class Processor:
            model_type = "qwen2_5_vl"

            def run(self, value):
                if isinstance(value, Exception):
                    raise value
                return value

        hooks._wrap_method(Processor, "run")
        marker = {"pixel_values": torch.ones(2, device="cpu")}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(Processor().run(marker), marker)
            failure = RuntimeError("original failure")
            with self.assertRaises(RuntimeError) as caught:
                Processor().run(failure)
        self.assertIs(caught.exception, failure)

    def test_unwritable_artifact_directory_never_changes_runtime_result(self):
        class Processor:
            model_type = "qwen2_5_vl"

            def run(self, value):
                return value

        hooks._wrap_runtime_method(Processor, "run", "test_boundary")
        marker = object()
        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            with contextlib.redirect_stdout(io.StringIO()) as log:
                self.assertIs(Processor().run(marker), marker)
        self.assertIn("NPU_VLM_DIAG_ERROR", log.getvalue())

    def test_async_wrapper_correlates_request_and_restores_context(self):
        class Request:
            rid = "request-123"

        class Processor:
            model_type = "qwen2_5_vl"

            async def process_mm_data_async(self, request_obj):
                await asyncio.sleep(0)
                return {"mrope_positions": torch.ones(3, 2, device="cpu")}

        hooks._wrap_method(Processor, "process_mm_data_async")
        with contextlib.redirect_stdout(io.StringIO()):
            result = asyncio.run(Processor().process_mm_data_async(Request()))
        self.assertEqual(list(result["mrope_positions"].shape), [3, 2])
        records = [
            json.loads(p.read_text()) for p in Path(self.temp.name).glob("*.json")
        ]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["rid"] == "request-123" for record in records))
        self.assertIsNone(hooks._RID.get())

    def test_nested_multimodal_features_and_offsets_are_not_truncated(self):
        """The first CI artifact silently lost feature/offsets inside mm_items."""
        feature = torch.arange(6, device="cpu").reshape(2, 3)
        item = SimpleNamespace(feature=feature, offsets=[(15, 294)], hash=42)
        output = SimpleNamespace(mm_items=[item])
        with contextlib.redirect_stdout(io.StringIO()):
            hooks.emit("processor.output", {"result": output})
        root = Path(self.temp.name)
        record = json.loads(next(root.glob("*.json")).read_text())
        saved = torch.load(root / record["tensor_file"], weights_only=True)
        torch.testing.assert_close(saved["value.result.mm_items.0.feature"], feature)
        self.assertEqual(
            record["payload"]["result"]["mm_items"][0]["offsets"], [[15, 294]]
        )
        self.assertEqual(item.offsets, [(15, 294)])
        torch.testing.assert_close(item.feature, feature)

    def test_decoder_observers_preserve_results_exceptions_and_outer_context(self):
        """Hooks must remain observational even when artifact writes fail."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
                self.lm_head = torch.nn.Linear(1, 1)
                self.result = SimpleNamespace(next_token_logits=None)

            def forward(self, input_ids, positions, forward_batch):
                if isinstance(input_ids, Exception):
                    raise input_ids
                self.result.next_token_logits = self.model.layers[0](input_ids)
                return self.result

        hooks._wrap_model_forward(Model)
        model = Model()
        batch = SimpleNamespace(mm_inputs=[], forward_mode="EXTEND")
        outer = hooks._FRAME.set("outer")
        outer_rid = hooks._RID.set("decoder-test-request")
        try:
            with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    self.assertIs(model(torch.ones(1, 1), None, batch), model.result)
                    failure = RuntimeError("original decoder failure")
                    with self.assertRaises(RuntimeError) as caught:
                        model(failure, None, batch)
                self.assertIs(caught.exception, failure)
                self.assertIn("NPU_VLM_DIAG_ERROR", log.getvalue())
            self.assertEqual(hooks._FRAME.get(), "outer")
            self.assertEqual(hooks._RID.get(), "decoder-test-request")
        finally:
            hooks._FRAME.reset(outer)
            hooks._RID.reset(outer_rid)


if __name__ == "__main__":
    unittest.main()
