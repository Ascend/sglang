import subprocess
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import write_results_to_github_step_summary
from sglang.test.sgl_eval_utils import run_sgl_eval


class TestMMLU:
    # Plain mixin, not a test case. Without this, pytest collects it from every
    # module that imports it (python_classes = Test*), instantiating it without
    # a server and failing with AttributeError.
    __test__ = False

    mmlu_num_examples = 128

    def test_mmlu(self):
        accuracy_mmlu_threshold = getattr(self, "accuracy_mmlu", 0.00)

        model_metrics = {
            "server": getattr(
                self, "server_cmd", subprocess.list2cmdline(map(str, self.other_args))
            ),
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
