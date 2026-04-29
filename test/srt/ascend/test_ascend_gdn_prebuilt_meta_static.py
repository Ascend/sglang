import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAMBA_META = ROOT / "python/sglang/srt/layers/attention/mamba/mamba2_metadata.py"
ASCEND_GDN = (
    ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py"
)
GDN_BACKEND = ROOT / "python/sglang/srt/layers/attention/linear/gdn_backend.py"
GDN_TRITON = ROOT / "python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py"
SERVER_ARGS = ROOT / "python/sglang/srt/server_args.py"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_forward_metadata_declares_non_spec_chunked_prefill_meta():
    tree = _module(MAMBA_META)
    forward_metadata = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ForwardMetadata"
    )
    fields = {
        node.target.id
        for node in forward_metadata.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "non_spec_chunked_prefill_meta" in fields


def test_ascend_gdn_backend_builds_meta_in_init_forward_metadata():
    source = ASCEND_GDN.read_text(encoding="utf-8")
    assert "GDNChunkedPrefillCache" in source
    assert "_maybe_build_non_spec_chunked_prefill_meta" in source
    assert "non_spec_chunked_prefill_meta" in source
    assert "disable_ascend_gdn_prebuilt_chunk_meta" in source


def test_gdn_extend_passes_prebuilt_meta_to_kernel():
    backend_source = GDN_BACKEND.read_text(encoding="utf-8")
    triton_source = GDN_TRITON.read_text(encoding="utf-8")
    ascend_source = ASCEND_GDN.read_text(encoding="utf-8")
    assert (
        'getattr(forward_metadata, "non_spec_chunked_prefill_meta", None)'
        in backend_source
    )
    assert "prebuilt_meta=prebuilt_meta" in backend_source
    assert "prebuilt_meta=None" in triton_source
    assert 'extra["prebuilt_meta"] = prebuilt_meta' in triton_source
    assert "cu_seqlens_cpu" in triton_source
    assert "prebuilt_meta=prebuilt_meta" in ascend_source
