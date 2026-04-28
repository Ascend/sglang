import logging
import os
import re
import subprocess
import threading
import time
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    SERVICE_PORT,
    check_role,
    launch_pd_mix_node,
    launch_pd_separation_node,
    launch_router,
    wait_server_ready,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

AISBENCHMARK = "aisbench"
BENCHMARK_TOOL_DEFAULT = AISBENCHMARK

PYTHON_FOR_TEST_TOOL = "test_env_transformers_tool/bin/python"
if not os.path.exists(PYTHON_FOR_TEST_TOOL) or not os.access(
    PYTHON_FOR_TEST_TOOL, os.X_OK
):
    PYTHON_FOR_TEST_TOOL = "python3"
logger.info(f"PYTHON_FOR_TEST_TOOL: {PYTHON_FOR_TEST_TOOL}")

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600
MAX_SERVER_KEEP_ALIVE_TIME = 3600

# Timeouts and delays
SERVER_INITIALIZATION_DELAY = 120

if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
    )
else:
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_VISIBLE_DEVICES", "0")[0]) * 100
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_SERVER_PORT_FOR_TEST + 66}"


def run_aisbench(
    host,
    port,
    model_path,
    dataset_name,
    dataset_type,
    output_len,
    max_concurrency,
    num_prompts,
):

    metrics_path = os.getenv("METRICS_DATA_FILE")
    result_path = "./aisbench_result" if not metrics_path else metrics_path
    logger.info(f"The metrics result file: {result_path}")

    cmd = f"/bin/bash /root/sglang/python/sglang/test/ascend/e2e/run_aisbench.sh "
    cmd += f"--mode accuracy "
    cmd += f"--ip {host} "
    cmd += f"--port {str(port)} "
    cmd += f"--model {os.path.basename(model_path)} "
    cmd += f"--model-path {model_path} "
    cmd += f"--dataset-name {dataset_name} "
    cmd += f"--dataset-type {dataset_type} "
    cmd += f"--output-path {result_path} "
    cmd += f"--output-len {output_len} "
    cmd += f"--batch-size {max_concurrency} "
    cmd += f"--num-prompts {num_prompts}"

    
    logger.info(f"Command: {cmd}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
    )

    output_lines = []
    try:
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                print(line, end="")
            output_lines.append(line.strip())

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with return code: {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)

        logger.info("Command executed successfully")

        metrics = {}
        full_output = "\n".join(output_lines)

        matches = re.findall(r"accuracy\s+[a-zA-Z]+\s+([\d.]+)", full_output)

        if matches:
            final_accuracy = float(matches[-1])
            metrics["accuracy"] = final_accuracy
            logger.info(f"The Final Accuracy: {final_accuracy}")
        else:
            logger.info(f"Can Not Find The Accuracy")

        return metrics

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating process...")
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.info("Process terminated")
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing it...")
            process.kill()
            logger.info("Process killed")
        raise
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        process.terminate()
        process.wait(timeout=5)
        raise


def assert_metrics(self, metrics):
    """Assert benchmark metrics against expected values.

    Args:
        metrics (dict): Benchmark metrics dictionary.
    """
    if not metrics:
        raise Exception("No metrics obtained from benchmark")

    if self.accuracy is not None:
        self.assertGreaterEqual(
            float(metrics["accuracy"]),
            self.accuracy,
            f"Accuracy check failed. Expected >= {self.accuracy}, Got: {metrics['accuracy']}",
        )


class TestAscendAccuracyTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "gsm8k_gen_4_shot_cot_str"  # gsm8k
    dataset_type = "gsm8k"
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    accuracy = 0.1
    output_len = 512
    max_concurrency = 1
    num_prompts = 100000

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        for key, value in env.items():
            logger.info(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                logger.info(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

        other_args = list(cls.other_args)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    def run_accuracy(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=host,
                port=port,
                model_path=self.model,
                dataset_name=self.dataset_name,
                dataset_type=self.dataset_type,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
            )
            assert_metrics(self, metrics)


class TestAscendAccuracyMultiNodePdMixTestCaseBase(CustomTestCase):
    model_config = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "gsm8k_gen_4_shot_cot_str"  # gsm8k
    dataset_type = "gsm8k"
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    accuracy = 0.1
    output_len = 512
    max_concurrency = 1

    @classmethod
    def setUpClass(cls):
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = "master" if cls.hostname.endswith("sglang-node-0") else "worker"
        logger.info(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_mix_master_node()
        cls.start_pd_mix_worker_node()

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    @check_role(allowed_roles=["master"])
    def start_pd_mix_master_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        wait_server_ready(f"{cls.base_url}/health")

        logger.info(
            f"Wait {SERVER_INITIALIZATION_DELAY}s, starting run benchmark ......"
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["worker"])
    def start_pd_mix_worker_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        logger.info(
            f"{cls.role} node started, keeping test alive for {MAX_SERVER_KEEP_ALIVE_TIME} seconds"
        )
        time.sleep(MAX_SERVER_KEEP_ALIVE_TIME)

    @check_role(allowed_roles=["master", "worker"])
    def run_accuracy(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=self.host,
                port=self.port,
                model_path=self.model_config.get("model_path"),
                dataset_name=self.dataset_name,
                dataset_type=self.dataset_type,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
            )
            assert_metrics(self, metrics)

class TestAscendAccuracyMultiNodePdSepTestCaseBase(CustomTestCase):
    model_config = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "gsm8k_gen_4_shot_cot_str"  # gsm8k
    dataset_type = "gsm8k"
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    max_attempts = 2
    accuracy = 0.1
    output_len = 512
    max_concurrency = 1
    num_prompts = 100000

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = (
            "router"
            if "router" in cls.hostname
            else "prefill" if "prefill" in cls.hostname else "decode"
        )
        logger.info(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_server()
        cls.start_router_server()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    @classmethod
    @check_role(allowed_roles=["router"])
    def start_router_server(cls):
        logger.info(f"Starting router in thread...")
        sglang_thread = threading.Thread(target=launch_router, args=(cls.model_config,))
        sglang_thread.daemon = True
        sglang_thread.start()

        health_check_url = f"{cls.base_url}/health"
        logger.info(f"Waiting for router to be ready at {health_check_url}")
        wait_server_ready(health_check_url)

        logger.info(
            f"Waiting {SERVER_INITIALIZATION_DELAY} seconds for the server to fully initialize..."
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["prefill", "decode"])
    def start_pd_server(cls):
        logger.info(f"Starting pd separation node...")
        cls.process = launch_pd_separation_node(cls.model_config)
        logger.info(f"Pd separation node started with PID: {cls.process.pid}")

        # Loop to check if the process is still running
        while True:
            if cls.process.poll() is None:
                # Process is still running
                time.sleep(30)
            else:
                # Process has exited
                exit_code = cls.process.poll()
                raise Exception(
                    f"Sglang process exited on node {cls.host} {cls.hostname} with exit code: {exit_code}"
                )

    @check_role(allowed_roles=["router"])
    def run_accuracy(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=host,
                port=port,
                model_path=self.model_config.get("model_path"),
                dataset_name=self.dataset_name,
                dataset_type=self.dataset_type,
                output_len=self.output_len,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
            )
            assert_metrics(self, metrics)
