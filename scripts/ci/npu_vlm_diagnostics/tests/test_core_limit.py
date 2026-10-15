"""CPU-only guards for the optional causal core-limit intervention."""

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hooks"))
import npu_vlm_core_limit as core_limit


class FakeNPU:
    def __init__(self):
        self.device = 0
        self.limits = {}
        self.calls = []
        self.failure = None
        self.ignore_set = False

    def current_device(self):
        return self.device

    def get_device_limit(self, device):
        return dict(
            self.limits.get(device, {"cube_core_num": 24, "vector_core_num": 48})
        )

    def set_device_limit(self, device, cube_num, vector_num):
        if self.failure is not None:
            raise self.failure
        self.calls.append((device, cube_num, vector_num))
        if not self.ignore_set:
            limits = self.get_device_limit(device)
            for field, value in (
                ("cube_core_num", cube_num),
                ("vector_core_num", vector_num),
            ):
                if value != -1:
                    limits[field] = value
            self.limits[device] = limits


class TestCoreLimit(CustomTestCase):
    def setUp(self):
        core_limit._APPLIED.clear()
        self.addCleanup(core_limit._APPLIED.clear)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.npu = FakeNPU()
        self.torch = SimpleNamespace(
            npu=self.npu, get_default_dtype=lambda: "torch.float32"
        )
        modules = patch.dict(
            sys.modules,
            {"torch": self.torch, "torch_npu": SimpleNamespace(npu=self.npu)},
        )
        modules.start()
        self.addCleanup(modules.stop)

    def model_class(self):
        npu = self.npu

        class Model:
            def __init__(self):
                self.weights = [
                    SimpleNamespace(dtype="torch.bfloat16") for _ in range(2)
                ]
                self.observed = []

            def parameters(self):
                return iter(self.weights)

            def forward(self, value, forward_batch=None):
                self.observed.append(
                    (
                        npu.current_device(),
                        npu.get_device_limit(npu.current_device()),
                        forward_batch,
                    )
                )
                if isinstance(value, Exception):
                    raise value
                return value

        return Model

    def enable(self, cls):
        os.environ.update(VLM_DEVICE_LIMIT_CUBE="20", VLM_DEVICE_LIMIT_VECTOR="40")
        core_limit.install(cls)

    def test_absent_environment_leaves_forward_identity_and_needs_no_torch(self):
        cls = self.model_class()
        original = cls.forward
        marker = object()
        with patch.dict(sys.modules, {"torch": None, "torch_npu": None}):
            core_limit.install(cls)
            self.assertIs(cls.forward, original)
            self.assertIs(cls().forward(marker), marker)
        self.assertEqual(self.npu.calls, [])

    def test_limit_precedes_unassociated_warmup_and_preserves_dtype_and_output(self):
        cls = self.model_class()
        self.enable(cls)
        installed = cls.forward
        core_limit.install(cls)
        self.assertIs(cls.forward, installed)
        model, marker = cls(), object()
        before = [(id(value), value.dtype) for value in model.weights]
        warmup = SimpleNamespace(mm_inputs=[], forward_mode="IDLE")
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertIs(model.forward(marker, forward_batch=warmup), marker)
            self.assertIs(model.forward(marker), marker)
        self.assertEqual(
            model.observed[0][1], {"cube_core_num": 20, "vector_core_num": 40}
        )
        self.assertIs(model.observed[0][2], warmup)
        self.assertEqual([(id(value), value.dtype) for value in model.weights], before)
        self.assertEqual(self.npu.calls, [(0, 20, 40)])
        records = [
            json.loads(line.split(" ", 1)[1]) for line in log.getvalue().splitlines()
        ]
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["dtype_unchanged"])
        self.assertEqual(records[0]["actual"], model.observed[0][1])
        self.assertEqual(records[0]["parameter_dtypes_after"], {"torch.bfloat16": 2})

    def test_cache_is_scoped_to_pid_and_current_device(self):
        cls = self.model_class()
        self.enable(cls)
        model = cls()
        with contextlib.redirect_stdout(io.StringIO()):
            with patch.object(core_limit.os, "getpid", return_value=100):
                model.forward(None)
                self.npu.device = 1
                model.forward(None)
                model.forward(None)
            with patch.object(core_limit.os, "getpid", return_value=101):
                model.forward(None)
        self.assertEqual(self.npu.calls, [(0, 20, 40), (1, 20, 40), (1, 20, 40)])
        self.assertTrue(
            all(
                item[1] == {"cube_core_num": 20, "vector_core_num": 40}
                for item in model.observed
            )
        )

    def test_api_failure_propagates_without_running_or_caching_the_model(self):
        cls = self.model_class()
        self.enable(cls)
        model = cls()
        self.npu.failure = RuntimeError("device limit unsupported")
        with self.assertRaises(RuntimeError) as caught:
            model.forward(object())
        self.assertIs(caught.exception, self.npu.failure)
        self.assertEqual(model.observed, [])
        self.assertEqual(core_limit._APPLIED, {})
        self.npu.failure = None
        with contextlib.redirect_stdout(io.StringIO()):
            marker = object()
            self.assertIs(model.forward(marker), marker)
        self.assertEqual(
            model.observed[0][1], {"cube_core_num": 20, "vector_core_num": 40}
        )

    def test_wrong_observed_limits_fail_before_the_model_executes(self):
        cls = self.model_class()
        self.enable(cls)
        model = cls()
        self.npu.ignore_set = True
        with self.assertRaisesRegex(RuntimeError, "expected 20, observed 24"):
            model.forward(object())
        self.assertEqual(model.observed, [])
        self.assertEqual(core_limit._APPLIED, {})

    def test_partial_limit_preserves_unspecified_vector_limit(self):
        cls = self.model_class()
        os.environ["VLM_DEVICE_LIMIT_CUBE"] = "20"
        core_limit.install(cls)
        model = cls()
        with contextlib.redirect_stdout(io.StringIO()):
            model.forward(None)
        self.assertEqual(
            model.observed[0][1], {"cube_core_num": 20, "vector_core_num": 48}
        )

    def test_invalid_limit_and_original_model_errors_are_not_swallowed(self):
        for value in ("", "0", "-1", "not-an-integer"):
            with self.subTest(value=value):
                cls = self.model_class()
                os.environ["VLM_DEVICE_LIMIT_CUBE"] = value
                core_limit.install(cls)
                model = cls()
                with self.assertRaises(ValueError):
                    model.forward(None)
                self.assertEqual(model.observed, [])
        cls = self.model_class()
        self.enable(cls)
        failure = RuntimeError("original model error")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                cls().forward(failure)
        self.assertIs(caught.exception, failure)


if __name__ == "__main__":
    unittest.main()
