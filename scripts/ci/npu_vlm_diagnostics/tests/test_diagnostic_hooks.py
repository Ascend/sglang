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
import npu_vlm_layer_probe as layer_probe


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

    def test_layer_probe_retains_full_attention_inputs_and_prefill_hidden_tail(self):
        class Attention(torch.nn.Module):
            def forward(self, q, k, v):
                return q

        layer = torch.nn.Module()
        layer.input_layernorm = torch.nn.Identity()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.attn = Attention()
        owner = SimpleNamespace(
            model=SimpleNamespace(layers=[layer], norm=torch.nn.Identity())
        )
        with patch.object(torch.distributed, "get_rank", return_value=0):
            with patch.dict(
                os.environ,
                VLM_DIAGNOSTICS_DIR=str(Path(self.temp.name) / "snapshots"),
            ):
                layer_probe.install(owner, hooks._FRAME)
        frame_token = hooks._FRAME.set({"index": 0, "mode": "prefill"})
        self.addCleanup(hooks._FRAME.reset, frame_token)
        q = torch.arange(30, dtype=torch.bfloat16).reshape(5, 6)
        self.assertIs(layer.input_layernorm(q), q)
        self.assertIs(layer.self_attn.attn(q, q, q), q)
        root = Path(self.temp.name) / "layer-probes"
        norm = torch.load(
            root / "r0-f0-layers.0.input_layernorm-input.pt", weights_only=True
        )
        attn = torch.load(
            root / "r0-f0-layers.0.self_attn.attn-input.pt", weights_only=True
        )
        output = torch.load(
            root / "r0-f0-layers.0.self_attn.attn-output.pt", weights_only=True
        )
        torch.testing.assert_close(norm["args.0"], q[-1:])
        torch.testing.assert_close(attn["args.0"], q)
        torch.testing.assert_close(output["result"], q[-1:])

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

    def test_layer_probe_handles_rotary_module_shared_between_layers(self):
        rotary = torch.nn.Identity()

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Module()
                self.self_attn.rotary_emb = rotary

            def forward(self, x):
                return self.self_attn.rotary_emb(x)

        layers = [Layer(), Layer()]
        owner = SimpleNamespace(
            model=SimpleNamespace(layers=layers, norm=torch.nn.Identity())
        )
        with patch.object(torch.distributed, "get_rank", return_value=0):
            with patch.dict(
                os.environ,
                VLM_DIAGNOSTICS_DIR=str(Path(self.temp.name) / "snapshots"),
            ):
                layer_probe.install(owner, hooks._FRAME)
        token = hooks._FRAME.set({"index": 0, "mode": "prefill"})
        self.addCleanup(hooks._FRAME.reset, token)
        x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        layers[0](x)
        layers[1](x + 1)
        root = Path(self.temp.name) / "layer-probes"
        for index in (0, 1):
            saved = torch.load(
                root / f"r0-f0-layers.{index}.self_attn.rotary_emb-output.pt",
                weights_only=True,
            )
            torch.testing.assert_close(saved["result"], (x + index)[-1:])
        records = [
            json.loads(line)
            for line in (root / "rank-0.jsonl").read_text().splitlines()
        ]
        self.assertEqual(sum(r["module"].endswith("rotary_emb") for r in records), 4)

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

    def test_qkv_capture_excludes_checkpoint_parameters_and_preserves_output(self):
        class TupleLinear(torch.nn.Linear):
            def forward(self, x):
                return super().forward(x), None

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                layer.self_attn.qkv_proj = TupleLinear(2, 3)
                self.model.layers = torch.nn.ModuleList([layer])
                self.lm_head = torch.nn.Linear(2, 3)

        model = Model()
        x = torch.ones(1, 2)
        qkv = model.model.layers[0].self_attn.qkv_proj
        expected = qkv(x)[0].clone()
        frame = hooks._FRAME.set({"index": 0, "mode": "EXTEND"})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hooks._decoder_observers(model)
                actual = qkv(x)
            torch.testing.assert_close(actual[0], expected)
            records = [
                json.loads(p.read_text()) for p in Path(self.temp.name).glob("*.json")
            ]
            for stage, key in (
                ("qkv.operator.input", "value.x"),
                ("qkv.operator.output", "value.y"),
            ):
                record = next(d for d in records if d["stage"] == stage)
                saved = torch.load(
                    Path(self.temp.name) / record["tensor_file"], weights_only=True
                )
                self.assertEqual(set(saved), {key})
                torch.testing.assert_close(
                    saved[key], x if stage.endswith("input") else actual[0]
                )
        finally:
            hooks._FRAME.reset(frame)

    def test_focused_snapshots_keep_vision_replay_and_qkv_tensors(self):
        with patch.dict(os.environ, VLM_DIAGNOSTICS_TENSOR_SCOPE="qkv"):
            with contextlib.redirect_stdout(io.StringIO()):
                for stage in (
                    "tokenizer_send.input",
                    "vision.output",
                    "qkv.operator.input",
                ):
                    hooks.emit(stage, {"result": torch.ones(2, device="cpu")})
        records = {
            json.loads(p.read_text())["stage"]: json.loads(p.read_text())
            for p in Path(self.temp.name).glob("*.json")
        }
        self.assertNotIn("tensor_file", records["tokenizer_send.input"])
        for stage in ("vision.output", "qkv.operator.input"):
            self.assertTrue(
                (Path(self.temp.name) / records[stage]["tensor_file"]).is_file()
            )

    def test_gate_up_capture_is_opt_in_prefill_only_and_retains_all_rows(self):
        class TupleLinear(torch.nn.Linear):
            def forward(self, x):
                return super().forward(x), None

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                layer = torch.nn.Module()
                layer.mlp = torch.nn.Module()
                layer.mlp.gate_up_proj = TupleLinear(2, 6, bias=False)
                self.model.layers = torch.nn.ModuleList([layer])
                self.lm_head = torch.nn.Linear(2, 3)

        model = Model()
        projection = model.model.layers[0].mlp.gate_up_proj
        x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        expected = projection(x)[0].clone()
        with contextlib.redirect_stdout(io.StringIO()):
            hooks._decoder_observers(model)
            with patch.dict(os.environ, VLM_DIAGNOSTICS_TENSOR_SCOPE="qkv"):
                for enabled, index in (("0", 0), ("1", 1), ("1", 0)):
                    with patch.dict(os.environ, VLM_GATE_UP_PROBE=enabled):
                        token = hooks._FRAME.set({"index": index, "mode": "EXTEND"})
                        try:
                            actual = projection(x)
                            torch.testing.assert_close(actual[0], expected)
                            self.assertIsNone(actual[1])
                        finally:
                            hooks._FRAME.reset(token)
        records = [
            json.loads(path.read_text()) for path in Path(self.temp.name).glob("*.json")
        ]
        captures = [
            record
            for record in records
            if record["stage"].startswith("gate_up.operator.")
        ]
        self.assertEqual(len(captures), 2)
        for record in captures:
            self.assertEqual(record["payload"]["frame"]["index"], 0)
            saved = torch.load(
                Path(self.temp.name) / record["tensor_file"], weights_only=True
            )
            key, reference = (
                ("value.x", x)
                if record["stage"].endswith("input")
                else ("value.y", expected)
            )
            self.assertEqual(set(saved), {key})
            self.assertEqual(saved[key].shape[0], 5)
            torch.testing.assert_close(saved[key], reference)

    def test_lm_head_logging_preserves_fp32_result_and_context(self):
        class Processor:
            use_fp32_lm_head = True

            def _compute_lm_head(self, hidden_states, lm_head):
                self.output = hidden_states.float() @ lm_head.weight.float().t()
                return self.output

        hooks._wrap_lm_head_projection(Processor)
        processor = Processor()
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        head = SimpleNamespace(weight=torch.ones(3, 2, dtype=torch.bfloat16))
        token = hooks._FRAME.set({"index": 3, "mode": "DECODE"})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = processor._compute_lm_head(x, head)
            self.assertIs(result, processor.output)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(hooks._FRAME.get()["index"], 3)
            record = json.loads(next(Path(self.temp.name).glob("*.json")).read_text())
            self.assertTrue(record["payload"]["enable_fp32_lm_head"])
            self.assertEqual(record["payload"]["result"]["dtype"], "torch.float32")
            self.assertTrue((Path(self.temp.name) / record["tensor_file"]).is_file())
        finally:
            hooks._FRAME.reset(token)

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
