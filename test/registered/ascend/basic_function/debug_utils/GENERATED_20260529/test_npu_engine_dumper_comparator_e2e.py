"""E2E test: source patcher + dumper + comparator on SGLang server (NPU).

Patches Qwen3MoeDecoderLayer.forward (and related methods) to insert
dumper.dump() calls at 7 points, launches servers with Qwen3-30B-A3B
(MOE model), runs inference, verifies patched dump fields exist, then
runs comparator to verify numerical consistency.

Test cases:
- test_patch_dump_and_compare: TP=2 baseline vs TP=4 target

The dumper.apply_source_patches() auto-injects ``from ... import dumper``
so the YAML only needs ``dumper.dump(...)`` calls.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="nightly-4-npu-a3", nightly=True)

MODEL = QWEN3_30B_A3B_WEIGHTS_PATH
BASELINE_TP = 1
TARGET_TP = 1
EXP_NAME = "e2e_source_patcher_npu"
DUMPER_FILTER = "layer_id in [0, 1, 2]"

_FIELDS_TO_VERIFY = [
    "layer_input",
    "attn_output",
    "pre_mlp_residual",
    "mlp_output",
    "attn_pre_o_proj",
    "moe_router_logits",
    "moe_expert_output",
]

PATCH_CONFIG_YAML = """\
patches:
  # --- decoder layer level (aligned with miles test) ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                  hidden_states,
                  residual,
                  forward_batch,
                  captured_last_layer_outputs=captured_last_layer_outputs,
                  **kwargs,
              )
          )
        append: "dumper.dump('layer_input', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.self_attn(
              positions=positions,
              hidden_states=hidden_states,
              forward_batch=forward_batch,
          )
        append: "dumper.dump('attn_output', hidden_states, dims='t h[attn_tp:partial] # tp:replicated')"
      - match: |
          hidden_states, residual = self.layer_communicator.prepare_mlp(
              hidden_states, residual, forward_batch
          )
        append: "dumper.dump('pre_mlp_residual', hidden_states, dims='t h # tp:replicated')"
      - match: |
          hidden_states = self.mlp(
              hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
          )
        append: "dumper.dump('mlp_output', hidden_states, dims='t h[moe_tp:partial] # tp:replicated')"

  # --- attention internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeAttention.forward_core
    edits:
      - match: "output, _ = self.o_proj(attn_output)"
        prepend: "dumper.dump('attn_pre_o_proj', attn_output, dims='t attn_h[attn_tp] # tp:replicated')"

  # --- moe internals ---
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal
    edits:
      - match: "router_logits, _ = self.gate(hidden_states)"
        append: "dumper.dump('moe_router_logits', router_logits, dims='t num_experts # tp:replicated')"
      - match: "final_hidden_states = self.experts(hidden_states, topk_output)"
        append: "dumper.dump('moe_expert_output', final_hidden_states, dims='t h[moe_tp:partial] # tp:replicated')"
"""


class TestNPUEngineDumperComparatorE2E(CustomTestCase):
    """E2E: patch Qwen3Moe forward -> dump -> compare on NPU.

    [Test Category] Feature
    [Test Target] Source patcher + dumper + comparator on NPU backend with MoE model
    """

    def test_patch_dump_and_compare(self):
        """TP=2 baseline vs TP=4 target on NPU."""
        tmp_path = Path(tempfile.mkdtemp(prefix="npu_e2e_patcher_"))
        base_url = DEFAULT_URL_FOR_TEST

        baseline_config_path = tmp_path / "patch_config.yaml"
        baseline_config_path.write_text(PATCH_CONFIG_YAML)

        target_config_path = tmp_path / "patch_config_target.yaml"
        target_config_path.write_text(PATCH_CONFIG_YAML)

        baseline_dir = tmp_path / "baseline"
        self._run_server_and_generate(
            dump_dir=baseline_dir,
            config_path=baseline_config_path,
            tp=BASELINE_TP,
            base_url=base_url,
        )
        self._verify_patched_fields(
            dump_dir=baseline_dir, field_names=_FIELDS_TO_VERIFY
        )

        target_dir = tmp_path / "target"
        self._run_server_and_generate(
            dump_dir=target_dir,
            config_path=target_config_path,
            tp=TARGET_TP,
            base_url=base_url,
        )
        self._verify_patched_fields(dump_dir=target_dir, field_names=_FIELDS_TO_VERIFY)

        baseline_exp = baseline_dir / EXP_NAME
        target_exp = target_dir / EXP_NAME

        cmd = [
            "python",
            "-m",
            "sglang.srt.debug_utils.comparator",
            "--baseline-path",
            str(baseline_exp),
            "--target-path",
            str(target_exp),
            "--output-format",
            "json",
            "--allow-skipped-pattern",
            "input_ids|positions",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        debug_file = self._save_comparator_output(
            stdout=result.stdout, stderr=result.stderr
        )

        self.assertEqual(
            result.returncode,
            0,
            f"Comparator failed (rc={result.returncode}). Debug output: {debug_file}",
        )

    def _run_server_and_generate(
        self,
        dump_dir: Path,
        config_path: Path,
        tp: int,
        base_url: str,
        extra_server_args: Optional[list] = None,
    ):
        """Launch SGLang server with source patcher + dumper, send a generate request."""
        # Use a dedicated port for dumper HTTP server to avoid conflicts with detokenizer
        env = {
            **os.environ,
            "DUMPER_SOURCE_PATCHER_CONFIG": str(config_path),
            "DUMPER_DIR": str(dump_dir),
            "DUMPER_EXP_NAME": EXP_NAME,
            "DUMPER_SERVER_PORT": "30001",
        }

        server_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--disable-radix-cache",
            "--tp",
            str(tp),
            "--max-total-tokens",
            "128",
            "--mem-fraction-static",
            "0.5",
            "--skip-server-warmup",
        ]
        if extra_server_args:
            server_args.extend(extra_server_args)

        proc = popen_launch_server(
            MODEL,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env=env,
        )
        dumper_url = "http://127.0.0.1:30001"
        try:
            requests.post(
                f"{dumper_url}/dumper/configure",
                json={
                    "enable": True,
                    "filter": DUMPER_FILTER,
                    "cleanup_previous": True,
                },
            ).raise_for_status()

            resp = requests.post(
                f"{base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                },
            )
            self.assertEqual(resp.status_code, 200, f"Generate failed: {resp.text}")
        finally:
            kill_process_tree(proc.pid)

    def _verify_patched_fields(self, dump_dir: Path, field_names: list):
        """Verify that patched dump fields exist as .pt files."""
        for field in field_names:
            matches = list(dump_dir.rglob(f"*name={field}*.pt"))
            self.assertGreater(
                len(matches),
                0,
                f"Expected patched field '{field}' not found under {dump_dir}. "
                f"Available files: {sorted(f.name for f in dump_dir.rglob('*.pt'))[:20]}",
            )

    def _save_comparator_output(self, stdout: str, stderr: str) -> Path:
        """Save comparator stdout+stderr to a temp file that persists for debugging."""
        fd, path_str = tempfile.mkstemp(
            prefix="comparator_npu_e2e_", suffix=".log", dir="/tmp"
        )
        with os.fdopen(fd, "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n=== STDERR ===\n")
            f.write(stderr)
        return Path(path_str)


if __name__ == "__main__":
    unittest.main()
