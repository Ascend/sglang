import os
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestVLMModels(CustomTestCase):
    model = ""
    mmmu_accuracy = 0.00
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.35,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        4,
    ]
    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    @classmethod
    def setUpClass(cls):
        # Removed argument parsing from here
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Set OpenAI API key and base URL environment variables. Needed for lmm-evals to work.
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.base_url}/v1"

    def _run_vlm_mmmu_test(self, test_name="", custom_env=None):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        process = None

        try:
            # Prepare environment variables
            process_env = os.environ.copy()
            if custom_env:
                process_env.update(custom_env)

            process = popen_launch_server(
                self.model,
                base_url=self.base_url,
                timeout=self.timeout_for_server_launch,
                api_key=self.api_key,
                other_args=self.other_args,
                env=process_env,
            )

            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmmu",
                num_examples=100,
                num_threads=64,
                max_tokens=30,
            )

            args.return_latency = True

            metrics, latency = run_eval(args)

            metrics["score"] = round(metrics["score"], 4)
            metrics["latency"] = round(latency, 4)
            print(
                f"{'=' * 42}\n{self.model} - metrics={metrics} score={metrics['score']}\n{'=' * 42}\n"
            )

            self.assertGreaterEqual(
                metrics["score"],
                self.mmmu_accuracy,
                f"Model {self.model} accuracy ({metrics['score']}) below expected threshold ({self.mmmu_accuracy:.4f}){test_name}",
            )

        except Exception as e:
            print(f"Error testing {self.model}{test_name}: {e}")
            self.fail(f"Test failed for {self.model}{test_name}: {e}")

        finally:
            # Ensure process cleanup happens regardless of success/failure
            if process is not None and process.poll() is None:
                print(f"Cleaning up process {process.pid}")
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")
