"""CPU regression for transfer_to_npu dispatch; does not emulate NPU kernels."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import dspark_schedule
from sglang.srt.hardware_backend.npu.attention import dspark_compact
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class MigratedNpuTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="npu")

    @property
    def is_cuda(self):
        # torch_npu.contrib.transfer_to_npu aliases is_cuda to is_npu.
        return True


class CudaTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda")

    @property
    def is_cuda(self):
        return True


def config(width=4, minimum=1, eps=0.0):
    return SimpleNamespace(
        resolved_max_verify_len=lambda: width,
        min_verify_len=minimum,
        survival_eps=eps,
    )


@pytest.mark.parametrize("budget", [0, 1, 2, 8])
def test_npu_reaches_dedicated_backend_before_is_cuda(budget):
    tensor = torch.ones(2, 4).as_subclass(MigratedNpuTensor)
    cfg = config()
    expected = object()
    with (
        patch.object(
            dspark_schedule, "inputs_on_cuda", side_effect=AssertionError("aliased")
        ),
        patch.object(
            dspark_schedule.ScheduleVerifyLensTopk,
            "triton",
            side_effect=AssertionError("wrong backend"),
        ),
        patch.object(
            dspark_compact, "schedule_verify_lens_npu", return_value=expected
        ) as backend,
    ):
        assert (
            dspark_schedule.ScheduleVerifyLensTopk.execute(
                confidence=tensor, budget=budget, cfg=cfg
            )
            is expected
        )
    backend.assert_called_once_with(tensor, budget=budget, cfg=cfg)


def test_cuda_keeps_original_triton_route():
    tensor = torch.ones(2, 4).as_subclass(CudaTensor)
    cfg = config()
    with (
        patch.object(dspark_schedule.ScheduleVerifyLensTopk, "triton") as triton,
        patch.object(
            dspark_schedule.ScheduleVerifyLensTopk,
            "torch",
            side_effect=AssertionError("wrong backend"),
        ),
    ):
        dspark_schedule.ScheduleVerifyLensTopk.execute(
            confidence=tensor, budget=2, cfg=cfg
        )
    triton.assert_called_once_with(confidence=tensor, budget=2, cfg=cfg)


@pytest.mark.parametrize("budget", [0, 1, 3, 8, 99])
@pytest.mark.parametrize("eps", [0.0, 0.3, 1.1])
def test_cpu_reference_and_npu_algorithm_preserve_ranking(budget, eps):
    # Ties at different positions/requests, invalid tails, and a partial tile.
    tensor = torch.tensor([[0.5, 1.0, 0.2, 0.0], [1.0, 0.5, 0.2, 0.0]])
    cfg = config(eps=eps)
    expected = dspark_schedule.schedule_verify_lens_topk(
        confidence=tensor, budget=budget, cfg=cfg
    )
    actual = dspark_schedule.ScheduleVerifyLensTopk.execute(
        confidence=tensor, budget=budget, cfg=cfg
    )
    npu_algorithm_on_cpu = dspark_compact.schedule_verify_lens_npu(
        tensor, budget=budget, cfg=cfg
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(npu_algorithm_on_cpu, expected)
