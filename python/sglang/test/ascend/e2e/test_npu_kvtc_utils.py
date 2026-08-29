import ast
import asyncio
import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import pandas as pd
import requests
import sys
import numpy as np
from urllib.parse import urlparse

from contextlib import contextmanager
from pathlib import Path
from openai import AsyncOpenAI

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    EVALSCOPE,
    TestAscendAccuracyTestCaseBase,
    run_evalscope,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    TestAscendPerformanceTestCaseBase,
    run_aisbench,
)
from sglang.test.kits.lm_eval_kit import LMEvalMixin

logger = logging.getLogger("kvtc_utils")

KVTC_EVALSCOPE = EVALSCOPE
KVTC_LM_EVAL = "lm_eval"

KVTC_REPO_PATH = Path(__file__).resolve().parents[5]
KVTC_CACHE_PATH = Path("/root/.cache/KVTC")
KVTC_DATASETS_PATH = KVTC_REPO_PATH / "python/sglang/test/ascend/e2e"
KVTC_CALIBRATION_PATH = KVTC_CACHE_PATH / "calibrations"
KVTC_CALIBRATION_LOCK_PATH = KVTC_CACHE_PATH / ".calibration.lock"
KVTC_DUMP_METADATA_FILENAME = "metadata.json"
KVTC_CALIBRATION_METADATA_FILENAME = "calibration.metadata.json"
KVTC_CALIBRATION_FILENAME = "kvtc.pt"
KVTC_CALIBRATION_SCRIPT_PATH = KVTC_REPO_PATH / "scripts" / "kvtc_calibrate.py"

OPENMATH_PARTS = 10
KVTC_DATASET_CONFIG = {
    "openmath": {
        "paths": KVTC_DATASETS_PATH / "openmath_selected_problems.parquet",
        "prompt_columns": ["problem", "solution"],
    },
    "fineweb": {
        "paths": KVTC_DATASETS_PATH / "fineweb_selected_problems.parquet",
        "prompt_columns": ["text"],
    },
}

KVTC_CALIBRATION_PARAMS = {
    "N": 200000,
    "q": 10000,
    "niter": 2,
}

class _AscendKvtcTestCaseBase:
    kvtc_dataset_config = KVTC_DATASET_CONFIG
    kvtc_client_concurrency = 4
    kvtc_limit_calibration = 0

    kvtc_calibration_params = KVTC_CALIBRATION_PARAMS

    kvtc_keys_compression_ratio = 8
    kvtc_values_compression_ratio = 8
    kvtc_sliding_window = 128
    kvtc_hicache_size = 15
    base_url = None

    @classmethod
    def _get_arg_value(cls, option: str, default: int) -> int:
        value = default
        other_args = list(cls.other_args or [])
        for index, argument in enumerate(other_args[:-1]):
            if argument == option:
                value = int(other_args[index + 1])
        return value

    @classmethod
    def _without_arg(cls, other_args: list, option: str) -> list:
        result = list(other_args)
        while option in result:
            option_index = result.index(option)
            del result[option_index : option_index + 2]
        return result

    @classmethod
    def _get_kvtc_config_metadata(cls) -> dict:
        if cls.model is None:
            raise ValueError("model must be set for a KVTC test")

        return {
            "model": str(cls.model),
            "tp_size": cls._get_arg_value("--tp-size", 1),
            "pp_size": cls._get_arg_value("--pp-size", 1),
            "datasets": cls.kvtc_dataset_config,
            "calibration_limit": cls.kvtc_limit_calibration,
        }

    @classmethod
    def _set_kvtc_artifact_paths(cls) -> None:
        metadata = cls._get_kvtc_config_metadata()
        metadata_json = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), default=str
        )
        config_id = hashlib.sha256(metadata_json.encode()).hexdigest()[:16]
        model_id = hashlib.sha256(str(cls.model).encode()).hexdigest()[:8]
        model_name = Path(str(cls.model)).name
        parallelism = f"tp-{metadata['tp_size']}-pp-{metadata['pp_size']}"

        cls.kvtc_artifact_path = (
            KVTC_CALIBRATION_PATH
            / f"{model_name}-{model_id}"
            / parallelism
            / f"config-{config_id}"
        )
        cls.kvtc_dump_path = cls.kvtc_artifact_path / "dump"
        cls.kvtc_calibration_path = (
            cls.kvtc_artifact_path
            / cls._get_kvtc_calibration_version()
            / KVTC_CALIBRATION_FILENAME
        )

    @classmethod
    def _check_metadata(cls, metadata_path: Path) -> bool:
        expected_metadata = json.loads(
            json.dumps(cls._get_kvtc_config_metadata(), default=str)
        )
        try:
            return json.loads(metadata_path.read_text()) == expected_metadata
        except (OSError, json.JSONDecodeError):
            return False

    @classmethod
    def _write_metadata(cls, metadata_path: Path) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(
                cls._get_kvtc_config_metadata(), indent=2, sort_keys=True, default=str
            )
            + "\n"
        )
        temporary_path.replace(metadata_path)

    @classmethod
    def _has_current_kvtc_dump(cls) -> bool:
        metadata_path = cls.kvtc_dump_path / KVTC_DUMP_METADATA_FILENAME
        return (
            cls.kvtc_dump_path.is_dir()
            and cls._check_metadata(metadata_path)
            and all(
                (cls.kvtc_dump_path / dataset_name).is_dir()
                for dataset_name in cls.kvtc_dataset_config
            )
        )

    @classmethod
    def _has_current_kvtc_calibration(cls) -> bool:
        metadata_path = cls.kvtc_artifact_path / KVTC_CALIBRATION_METADATA_FILENAME
        return cls.kvtc_calibration_path.is_file() and cls._check_metadata(
            metadata_path
        )

    @classmethod
    async def _send_kvtc_dump_requests(cls, dataset_name: str, prompts) -> int:
        client = AsyncOpenAI(base_url=f"{cls.base_url}/v1", api_key="None")
        semaphore = asyncio.Semaphore(cls.kvtc_client_concurrency)

        async def send_request(entry) -> bool:
            prompt = entry
            async with semaphore:
                logger.debug("KVTC dump %s request %s", dataset_name)
                try:
                    await client.chat.completions.create(
                        model=str(cls.model),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_completion_tokens=16*1024,
                    )
                except Exception:
                    logger.error(
                        "!!! KVTC DUMP REQUEST FAILED; SKIPPING DATASET=%s !!!",
                        dataset_name,
                        exc_info=True,
                    )
                    return False
                return True

        try:
            results = await asyncio.gather(
                *(send_request(prompt) for prompt in prompts)
            )
        finally:
            await client.close()

        collected_dumps = sum(results)
        expected_dumps = len(prompts)
        logger.info(
            "KVTC dump dataset %s: collected %s/%s dumps",
            dataset_name,
            collected_dumps,
            expected_dumps,
        )
        if collected_dumps != expected_dumps:
            logger.error(
                "!!! KVTC DUMP DATASET INCOMPLETE: %s collected %s/%s dumps !!!",
                dataset_name,
                collected_dumps,
                expected_dumps,
            )
        return collected_dumps

    @classmethod
    def _remove_incomplete_dump(cls) -> None:
        for dump_path in (
            cls.kvtc_dump_path,
            cls.kvtc_dump_path.with_name("dump.tmp"),
        ):
            if dump_path.is_dir():
                shutil.rmtree(dump_path)
            elif dump_path.exists():
                dump_path.unlink()

    @classmethod
    def _create_kvtc_dump(cls) -> None:
        logger.info("Creating KVTC dump: %s", cls.kvtc_dump_path)
        dataset_prompts = {}
        for dataset_name in cls.kvtc_dataset_config:
            prompts = cls.get_kvtc_prompts(dataset_name)
            dataset_prompts[dataset_name] = prompts

        cls._remove_incomplete_dump()
        staging_path = cls.kvtc_dump_path.with_name("dump.tmp")
        staging_path.mkdir(parents=True)

        original_args = cls.other_args
        try:
            for dataset_name, prompts in dataset_prompts.items():
                dataset_dump_path = staging_path / dataset_name
                dataset_dump_path.mkdir()
                cls.other_args = original_args + ["--dump-kv-path", dataset_dump_path]
                cls.process = None
                dump_server_started = False
                try:
                    super().setUpClass()
                    dump_server_started = True
                    asyncio.run(cls._send_kvtc_dump_requests(dataset_name, prompts))
                finally:
                    try:
                        if dump_server_started or cls.process is not None:
                            super().tearDownClass()
                    finally:
                        cls.process = None
                        cls.other_args = original_args
            cls._write_metadata(staging_path / KVTC_DUMP_METADATA_FILENAME)
            staging_path.replace(cls.kvtc_dump_path)
        except BaseException:
            if staging_path.exists():
                shutil.rmtree(staging_path)
            raise

    @classmethod
    def _get_kvtc_calibration_version(cls) -> str:
        if not KVTC_CALIBRATION_SCRIPT_PATH.is_file():
            raise FileNotFoundError(
                "KVTC calibration script is not present: "
                f"{KVTC_CALIBRATION_SCRIPT_PATH}"
            )

        p = subprocess.run(
            [
                sys.executable,
                str(KVTC_CALIBRATION_SCRIPT_PATH),
                "--kvtc-version",
            ],
            check=True,
            capture_output=True,
        )

        return p.stdout.decode().strip()

    @classmethod
    def _run_kvtc_calibration(cls) -> None:
        if not KVTC_CALIBRATION_SCRIPT_PATH.is_file():
            raise FileNotFoundError(
                "KVTC calibration script is not present: "
                f"{KVTC_CALIBRATION_SCRIPT_PATH}"
            )

        cls.kvtc_artifact_path.mkdir(parents=True, exist_ok=True)
        cls.kvtc_calibration_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cls.kvtc_calibration_path.with_suffix(
            cls.kvtc_calibration_path.suffix + ".tmp"
        )
        temporary_path.unlink(missing_ok=True)
        logger.info("Calibrating KVTC dump: %s", cls.kvtc_dump_path)
        try:
            p = subprocess.run(
                [
                    sys.executable,
                    str(KVTC_CALIBRATION_SCRIPT_PATH),
                    "--model-dir",
                    cls.model,
                    "--input-dir",
                    str(cls.kvtc_dump_path),
                    "--output",
                    str(temporary_path),
                    "--log-dir",
                    str(KVTC_CACHE_PATH),
                    "-q",
                    str(cls.kvtc_calibration_params["q"]),
                    "-N",
                    str(cls.kvtc_calibration_params["N"]),
                    "--niter",
                    str(cls.kvtc_calibration_params["niter"]),
                    "--sampling-policy",
                    "relaxed",
                    "--svd-cache-policy",
                    "reuse",
                ],
                check=True,
                capture_output=True,
            )

            logger.info("KVTC calibration script done")
            logger.info(p.stdout.decode())
            logger.info(p.stderr.decode())
            logger.info("------------------------------------------------")

            if not temporary_path.is_file():
                raise RuntimeError(
                    "KVTC calibration script did not create its output: "
                    f"{temporary_path}"
                )
            temporary_path.replace(cls.kvtc_calibration_path)
            cls._write_metadata(
                cls.kvtc_artifact_path / KVTC_CALIBRATION_METADATA_FILENAME
            )
        except BaseException as e:
            if isinstance(e, subprocess.CalledProcessError):
                logger.error(f"KVTC calibration script failed {e.output}")
                logger.exception("")
            temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def _ensure_kvtc_artifacts(cls) -> None:
        KVTC_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Waiting for KVTC calibration lock: %s", KVTC_CALIBRATION_LOCK_PATH)
        with KVTC_CALIBRATION_LOCK_PATH.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if cls._has_current_kvtc_calibration():
                    logger.info(
                        "Reusing KVTC calibration: %s", cls.kvtc_calibration_path
                    )
                    return
                if not cls._has_current_kvtc_dump():
                    cls._create_kvtc_dump()
                cls._run_kvtc_calibration()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @classmethod
    def setUpClass(cls):
        cls._set_kvtc_artifact_paths()
        cls._ensure_kvtc_artifacts()
        cls._kvtc_test_server_started = False

        cls._kvtc_original_other_args = cls.other_args
        cls.other_args = cls.other_args + [
            "--enable-hierarchical-cache",
            "--hicache-kvtc-params",
            cls.kvtc_calibration_path,
            "--hicache-size",
            cls.kvtc_hicache_size,
            "--hicache-kvtc-k-cr",
            cls.kvtc_keys_compression_ratio,
            "--hicache-kvtc-v-cr",
            cls.kvtc_values_compression_ratio,
            "--hicache-kvtc-sw",
            cls.kvtc_sliding_window,
        ]
        try:
            super().setUpClass()
            cls._kvtc_test_server_started = True
        except BaseException:
            try:
                if getattr(cls, "process", None) is not None:
                    super().tearDownClass()
            finally:
                cls.process = None
                cls.other_args = cls._kvtc_original_other_args
            raise

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_kvtc_test_server_started", False):
            try:
                super().tearDownClass()
            finally:
                cls.other_args = cls._kvtc_original_other_args

    @classmethod
    def load_kvtc_dataset(cls, dataset_name: str):
        logger.info(f"Loading KVTC calibration dataset: {dataset_name}...")

        return pd.read_parquet(
                cls.kvtc_dataset_config[dataset_name]["paths"],
                columns=cls.kvtc_dataset_config[dataset_name]["prompt_columns"],
                )

    @classmethod
    def get_kvtc_prompts(cls, dataset_name: str):
        dataset = cls.load_kvtc_dataset(dataset_name)

        prompts = dataset.astype(str).agg(" ".join, axis=1)

        logger.info(f"Found {len(prompts)} calibration prompts for {dataset_name}")

        if cls.kvtc_limit_calibration:
            logger.warning(
                f"Short KVTC calibration active, limiting to {cls.kvtc_limit_calibration} prompts"
            )
            prompts = prompts[: cls.kvtc_limit_calibration]

        return prompts

    def _validate_trim_eviction(self, data):
        pass

    def trim_cache(self):
        requests.post(url=self.base_url + "/trim_cache", timeout=30)
        resp = requests.get(url=self.base_url + "/radix_tree", timeout=30)
        data = ast.literal_eval(resp.text)
        self._validate_trim_eviction(data)


class TestAscendPerformanceKvtcTestCaseBase(
    _AscendKvtcTestCaseBase, TestAscendPerformanceTestCaseBase
):
    pass


class TestAscendAccuracyKvtcTestCaseBase(
    _AscendKvtcTestCaseBase, TestAscendAccuracyTestCaseBase
):
    pass


class TestAscendPerformanceKvtcTestCaseLME(
    _AscendKvtcTestCaseBase, TestAscendPerformanceTestCaseBase
):
    model = None
    batch_size = 16
    backend = "local-completions"
    metadata = {}
    num_fewshot = 0
    benchmark_size_limit = None
    apply_chat_template = False
    fewshot_as_multiturn = False
    gen_kwargs = None
    task_list = [{"name": None}]
    rtol = 1.0
    benchmark_tool = None
    max_retries = 1

    def launch_eval(self):
        model_args = {
            "model": self.model,
            #"base_url": f"{self.base_url}/v1/chat/completions",
            "base_url": f"{self.base_url}/v1/completions",
            "num_concurrent": self.kvtc_client_concurrency,
            "api_key": "None",
            "tokenizer_backend": "auto",
        }

        if self.benchmark_tool == KVTC_LM_EVAL:
            import lm_eval

            results = lm_eval.simple_evaluate(
                model=self.backend,
                model_args=model_args,
                tasks=[task["name"] for task in self.task_list],
                task_manager=lm_eval.tasks.TaskManager(metadata=self.metadata),
                num_fewshot=self.num_fewshot,
                limit=self.benchmark_size_limit,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                gen_kwargs=self.gen_kwargs,
                batch_size=self.batch_size,
            )

            return {task['name']: results["results"][task["name"]] for task in self.task_list}

        elif self.benchmark_tool == KVTC_EVALSCOPE:
            parsed_url = urlparse(self.base_url)
            host = parsed_url.hostname
            port = parsed_url.port

            # Thise should be global params
            datasets = ["gsm8k"]
            dataset_args = None
            generation_config=None
            dataset_dir = None
            stream = True
            timeout = 60000
            eval_type = "openai_api"

            model_name = os.path.basename(self.model)
            metrics = run_evalscope(
                host=host,
                port=port,
                model=model_name,
                datasets=datasets,
                dataset_args=dataset_args,
                eval_batch_size=self.batch_size,
                limit=self.benchmark_size_limit,
                generation_config=generation_config,
                dataset_dir=dataset_dir,
                stream=stream,
                timeout=timeout,
                eval_type=eval_type,
            )
            return metrics
        else:
            self.assertTrue(False, f"Invalid benchmarking tool {self.benchmark_tool}")


    def print_lm_eval_results(self, results):
        for task in results:
            for column in results[task]:
                logger.info(f"\t{column}: {results[task][column]}")
            logger.info("--------")
