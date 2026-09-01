import inspect
import unittest

from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
    C128SidecarComponent,
)
from sglang.srt.mem_cache.unified_cache.components import TreeComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# ``TreeComponent`` lifecycle hooks that received ``result: InsertResult`` in
# sgl-project/sglang PR #37091. Subclass overrides must stay compatible.
_CONTRACT_METHODS = (
    "update_component_on_insert_overlap",
    "recover_after_unevict",
    "commit_insert_component_data",
)


class TestC128SidecarComponentContract(CustomTestCase):
    """sgl-project/sglang PR #37091 added ``result: InsertResult`` to several
    ``TreeComponent`` hooks, and the insert walk now calls them with
    ``result=...`` as a keyword argument. A subclass that keeps the old
    signature raises ``TypeError`` on the evict -> unevict path. Guard that the
    NPU C128 sidecar stays signature-compatible with the base class.
    """

    def test_overrides_accept_every_base_class_parameter(self):
        for method_name in _CONTRACT_METHODS:
            base_params = inspect.signature(
                getattr(TreeComponent, method_name)
            ).parameters
            sub_params = inspect.signature(
                getattr(C128SidecarComponent, method_name)
            ).parameters
            missing = [name for name in base_params if name not in sub_params]
            with self.subTest(method=method_name):
                self.assertEqual(
                    missing,
                    [],
                    f"{C128SidecarComponent.__name__}.{method_name} is missing "
                    f"base-class parameter(s) {missing}; unified-cache contract "
                    f"broken.",
                )


if __name__ == "__main__":
    unittest.main()