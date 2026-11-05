import subprocess
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import write_results_to_github_step_summary
from sglang.test.sgl_eval_utils import run_sgl_eval


class TestMMLU:
    mmlu_num_examples = 128

    def test_mmlu(self):
        accuracy_mmlu_threshold = getattr(self, "accuracy_mmlu", 0.00)

        # NB: getattr's default is evaluated eagerly, so referencing
        # self.other_args here would raise AttributeError on mixins whose
        # subclass never defines it. Resolve defensively instead.
        if hasattr(self, "server_cmd"):
            server_cmd = self.server_cmd
        else:
            server_cmd = subprocess.list2cmdline(
                map(str, getattr(self, "other_args", []))
            )
        model_metrics = {
            "server": server_cmd,
            "client": "sgl-eval",
            "accuracy_threshold": getattr(self, "accuracy_mmlu", "N/A"),
        }

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmlu",
                num_examples=self.mmlu_num_examples,
                num_threads=32,
            )
            print("Starting mmlu test...")
            metrics = run_sgl_eval(args)
            model_metrics["accuracy"] = metrics["score"]
            model_metrics["latency"] = metrics.get("latency", "-")
            model_metrics["output_throughput"] = metrics.get("output_throughput", "-")
            self.assertGreater(metrics["score"], accuracy_mmlu_threshold)
        except Exception as e:
            model_metrics["error"] = e
            self.fail(f"Test failed for {self.model}: {e}")
        finally:
            write_results_to_github_step_summary({self.model: model_metrics})
