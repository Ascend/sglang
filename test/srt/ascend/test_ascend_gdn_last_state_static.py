import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ASCEND_GDN = (
    ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py"
)
GDN_TRITON = ROOT / "python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py"


def test_ascend_gdn_extend_skips_missing_last_recurrent_state():
    source = ASCEND_GDN.read_text(encoding="utf-8")
    tree = ast.parse(source)

    guarded_blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "last_recurrent_state"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value is None
            for comparator in node.test.comparators
        )
        and any(isinstance(op, ast.IsNot) for op in node.test.ops)
    ]

    assert any(
        isinstance(child, ast.If)
        and "forward_batch.spec_algorithm.is_none" in ast.unparse(child.test)
        for block in guarded_blocks
        for child in ast.walk(block)
    )


def test_gdn_triton_returns_indexed_recurrent_state_when_kernel_omits_final_state():
    source = GDN_TRITON.read_text(encoding="utf-8")
    assert "last_recurrent_state is None" in source
    assert "last_recurrent_state = recurrent_state" in source
