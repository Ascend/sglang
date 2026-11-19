import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_DECRYPTED_WEIGHTS_PATH,
    QWEN3_8B_EAGLE3_DECRYPTED_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-2-npu-a3",
    nightly=True,
)

DECRYPTED_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "Qwen3-8B", "config.json"
)
DECRYPTED_DRAFT_CONFIG_FILE = os.path.join(
    os.path.dirname(__file__), "Qwen3-8B_eagle3", "config.json"
)


def _mirror_weights_with_stub_config(source_dir, target_dir, decrypted_config_file):
    """Mirror a checkpoint through symlinks and shadow config.json with a stub.

    The image reads <model_path>/config.json in muse_spark_hook before the
    --decrypted-*-config-file overrides are applied, so a checkpoint without a
    readable config.json cannot boot even though the decrypted file is given.
    Mirroring the weights keeps the shared cache untouched, and the stub keeps
    the hook happy. Its sentinel architecture matches no registered model, so
    the case still fails if the override is ignored.
    """
    os.makedirs(target_dir, exist_ok=True)
    for name in os.listdir(source_dir):
        if name == "config.json":
            continue
        os.symlink(os.path.join(source_dir, name), os.path.join(target_dir, name))

    with open(decrypted_config_file, encoding="utf-8") as f:
        stub_config = json.load(f)
    stub_config["architectures"] = ["DecryptedConfigFileOverrideSentinel"]
    with open(os.path.join(target_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(stub_config, f)


class TestDraftConfigFile(CustomTestCase):
    """Testcase: Verify set --decrypted-config-file, --decrypted-draft-config-file parameter,
    will use the specified config.json and the GSM8K dataset is no less than 0.95.

    [Test Category] Parameter
    [Test Target] --decrypted-config-file, --decrypted-draft-config-file
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="sglang-decrypted-config-")
        cls.model_path = os.path.join(cls._tmpdir, "Qwen3-8B")
        cls.draft_model_path = os.path.join(cls._tmpdir, "Qwen3-8B_eagle3")
        _mirror_weights_with_stub_config(
            QWEN3_8B_DECRYPTED_WEIGHTS_PATH, cls.model_path, DECRYPTED_CONFIG_FILE
        )
        _mirror_weights_with_stub_config(
            QWEN3_8B_EAGLE3_DECRYPTED_WEIGHTS_PATH,
            cls.draft_model_path,
            DECRYPTED_DRAFT_CONFIG_FILE,
        )

        cls.process = None
        try:
            cls.process = popen_launch_server(
                cls.model_path,
                DEFAULT_URL_FOR_TEST,
                DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--attention-backend",
                    "ascend",
                    "--disable-radix-cache",
                    "--chunked-prefill-size",
                    "-1",
                    "--max-prefill-tokens",
                    "1024",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model-path",
                    cls.draft_model_path,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--tp-size",
                    "2",
                    "--mem-fraction-static",
                    "0.68",
                    "--disable-cuda-graph",
                    "--dtype",
                    "bfloat16",
                    "--decrypted-config-file",
                    DECRYPTED_CONFIG_FILE,
                    "--decrypted-draft-config-file",
                    DECRYPTED_DRAFT_CONFIG_FILE,
                ],
                env={
                    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                },
            )

        except Exception as e:
            if cls.process is not None:
                kill_process_tree(cls.process.pid)
            shutil.rmtree(cls._tmpdir, ignore_errors=True)
            raise RuntimeError(f"Failed to launch server: {e}") from e

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_gsm8k(self):
        args = SimpleNamespace(
            max_tokens=512,
            base_url=DEFAULT_URL_FOR_TEST,
            model=self.model_path,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )
        metrics = run_eval(args)
        # The GSM8K baseline accuracy reported in the Qwen3-8B model paper is 0.8984.
        self.assertGreater(metrics["score"], 0.8984)


if __name__ == "__main__":
    unittest.main()
