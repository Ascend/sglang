import io
import logging
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Initialize logging configuration (replace print)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Register CI task for NPU environment
register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class BaseTestDetokenizerWatchdog:
    """Testcase: Ensure that soft-watchdog-timeout is set by default in the CI environment, and in non-CI environments it is not set by default and needs to be set manually.

    [Test Category] Parameter
    [Test Target] --soft-watchdog-timeout
    """

    ci_mode = None
    set_soft_watchdog = None
    soft_watchdog_value = 10
    stuck_seconds = 350
    expected_log = None
    expected_assert_error = (
        "stuck tester can be enabled only if soft watchdog is enabled"
    )

    @classmethod
    def setUpClass(cls):
        # Set CI-mode env for the whole class lifetime: the server reads it at
        # launch, and CustomTestCase reads is_in_ci() during test methods to
        # pick the retry count -- both must see the scenario value.
        if cls.ci_mode is not None:
            cls._ci_env_ctx = envs.SGLANG_IS_IN_CI.override(cls.ci_mode)
            cls._ci_env_ctx.__enter__()
            cls.addClassCleanup(cls._ci_env_ctx.__exit__, None, None, None)

        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = None
        cls.launch_success = False
        cls.error_found_in_log = False  # Mark if expected error is found in logs

        # Build launch arguments (whether to set soft-watchdog-timeout)
        other_args = ["--skip-server-warmup"]
        if cls.set_soft_watchdog:
            other_args.extend(["--soft-watchdog-timeout", str(cls.soft_watchdog_value)])

        # Scenario 4 timeout set to 60 seconds (ensure complete log printing)
        timeout = (
            60
            if (cls.ci_mode is False and cls.set_soft_watchdog is False)
            else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

        try:
            # Simulate detokenizer blocking
            with envs.SGLANG_TEST_STUCK_DETOKENIZER.override(cls.stuck_seconds):
                cls.process = popen_launch_server(
                    QWEN3_0_6B_WEIGHTS_PATH,
                    DEFAULT_URL_FOR_TEST,
                    timeout=timeout,
                    other_args=other_args,
                    return_stdout_stderr=(cls.stdout, cls.stderr),
                )
            cls.launch_success = True
        except Exception:
            # Scenario 4 expects the launch to fail: either the launch times
            # out, or the detokenizer subprocess asserts and the server exits
            # (code -9). Both surface as a generic Exception. Verify the
            # expected AssertionError is present in the logs below.
            cls.launch_success = False
            # Scenarios' server environments.
            envs.SGLANG_TEST_STUCK_DETOKENIZER.clear()
            # Read complete logs
            combined_log = cls.stdout.getvalue() + cls.stderr.getvalue()
            # Check if contains expected AssertionError string
            if cls.expected_assert_error in combined_log:
                cls.error_found_in_log = True
                logger.info(
                    f"\n[Scenario 4] Found expected error in logs: {cls.expected_assert_error}"
                )
                # Print complete logs for troubleshooting
                logger.info(f"\n[Scenario 4] Complete logs:\n{combined_log}")
            else:
                # Expected error not found, raise the original launch error
                raise

    @classmethod
    def tearDownClass(cls):
        # Final fallback cleanup
        if cls.process:
            kill_process_tree(cls.process.pid)
        if cls.stdout:
            cls.stdout.close()
        if cls.stderr:
            cls.stderr.close()

    def test_detokenizer_watchdog(self):
        combined_log = self.stdout.getvalue() + self.stderr.getvalue()

        # Scenario 1 (CI + no flag): verify the CI default (300s) is applied via
        # server args print and watchdog creation. Firing is covered by scenarios
        # 2/3 with a short timeout (same code path), so no need to wait 300s.
        if self.ci_mode is True and self.set_soft_watchdog is False:
            self.assertTrue(self.launch_success, "Server launch failed")
            self.assertIn(
                "soft_watchdog_timeout=300",
                combined_log,
                "Scenario 1: CI default soft watchdog (300s) was not applied",
            )
            self.assertIn(
                "Watchdog DetokenizerManager initialized",
                combined_log,
                "Scenario 1: real DetokenizerManager watchdog was not created",
            )
            logger.info(
                "[Scenario 1] Test passed: CI default soft watchdog (300s) is applied"
            )
            return

        # Scenario 4: Non-CI + no soft watchdog → verify AssertionError in logs
        if self.ci_mode is False and self.set_soft_watchdog is False:
            self.assertTrue(
                self.error_found_in_log,
                f"Scenario 4: Expected error not found in logs: {self.expected_assert_error}",
            )
            logger.info(
                "[Scenario 4] Test passed: Found expected AssertionError string in logs"
            )
            return

        # Scenarios 2-3: Launch success → call API and verify timeout logs
        self.assertTrue(self.launch_success, "Server launch failed")
        logger.info("Start call /generate API", extra={"flush": True})
        try:
            requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "Hello, please repeat this sentence for 1000 times.",
                    "sampling_params": {"max_new_tokens": 100, "temperature": 0},
                },
                timeout=40,
            )
        except requests.exceptions.ReadTimeout as e:
            # The detokenizer is deliberately stuck, so the request read-timeout
            # is expected; what matters is the watchdog timeout log below.
            logger.info(f"requests.post timed out (expected): {e}")
        logger.info("End call /generate API", extra={"flush": True})

        # Merge output and verify expected logs
        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(
            self.expected_log,
            combined_output,
            f"Expected log not found: {self.expected_log}",
        )
        logger.info(
            f"[Scenario {self.__class__.__name__}] Test passed: Found expected log {self.expected_log}"
        )


# ===================== Test Subclasses for Four Scenarios =====================
# Scenario 1: CI environment + no soft-watchdog (default 300s) → block 350s
class TestCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = False
    stuck_seconds = 0
    expected_log = None


# Scenario 2: CI environment + set soft-watchdog (20s) → block 30s
class TestCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = True
    soft_watchdog_value = 20
    stuck_seconds = 30
    expected_log = "DetokenizerManager watchdog timeout"


# Scenario 3: Non-CI environment + set soft-watchdog (20s) → block 30s
class TestNonCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = True
    soft_watchdog_value = 20
    stuck_seconds = 30
    expected_log = "DetokenizerManager watchdog timeout"


# Scenario 4: Non-CI environment + no soft-watchdog (verify AssertionError in logs)
class TestNonCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = False


# ===================== Watchdog Tests =====================
class BaseTestSoftWatchdog:
    """Testcase: Verify that soft-watchdog-timeout triggers correctly when Scheduler init is stuck.

    [Test Category] Parameter
    [Test Target] --soft-watchdog-timeout
    """

    env_override = None
    expected_message = None

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        with cls.env_override():
            cls.process = popen_launch_server(
                QWEN3_0_6B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--soft-watchdog-timeout",
                    "20",
                    "--skip-server-warmup",
                ],
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_scheduler_init_watchdog(self):
        logger.info("Start call /generate API")
        try:
            requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "Hello, please repeat this sentence for 100 times.",
                    "sampling_params": {"max_new_tokens": 100, "temperature": 0},
                },
                timeout=30,
            )
        except requests.exceptions.ReadTimeout as e:
            logger.info(f"requests.post timeout (but expected): {e}")

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(self.expected_message, combined_output)


class TestSoftWatchdogTokenizer(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_TOKENIZER.override(30)
    expected_message = "TokenizerManager watchdog timeout"


class TestSoftWatchdogSchedulerInit(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.override(30)
    expected_message = "Scheduler watchdog timeout"


# ===================== Main Function =====================
def load_tests(loader, standard_tests, pattern):
    """Pin the original scenario order (unittest sorts by class name by default)."""
    suite = unittest.TestSuite()
    for cls in (
        TestCIWithoutSoftWatchdog,
        TestCIWithSoftWatchdog,
        TestNonCIWithSoftWatchdog,
        TestNonCIWithoutSoftWatchdog,
        TestSoftWatchdogTokenizer,
        TestSoftWatchdogSchedulerInit,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


if __name__ == "__main__":
    # CI executes this file as `python3 file.py -f`; run every TestCase subclass
    # so that test failures propagate as a non-zero exit code.
    unittest.main()
