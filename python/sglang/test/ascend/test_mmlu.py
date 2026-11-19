import subprocess
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import write_results_to_github_step_summary
from sglang.test.sgl_eval_utils import run_sgl_eval


class TestMMLU:
    # NB: pytest collects every module attribute matching Test* — including
    # this mixin imported into a test file's namespace. A bare instance has
    # no base_url/model/fail, so each importing file gains a spurious failing
    # test. Opt the mixin out explicitly; real host classes set __test__ = True.
    __test__ = False

    mmlu_num_examples = 128

    def test_mmlu(self):
        accuracy_mmlu_threshold = getattr(self, "accuracy_mmlu", 0.00)

        # getattr's default is evaluated eagerly; resolve the server command
        # defensively instead.
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
