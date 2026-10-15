"""Opt-in diagnostics for the container's SGLang, including spawned workers."""

import os

if os.environ.get("VLM_DIAGNOSTICS_DIR") or os.environ.get("VLM_SUPPORTED_ONLY") == "1":
    from npu_vlm_diagnostic_hooks import install_import_hook

    install_import_hook()
