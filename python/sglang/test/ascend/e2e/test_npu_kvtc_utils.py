import os
from concurrent.futures import ThreadPoolExecutor
_HF_BASE = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
from pathlib import Path

import pandas as pd
import requests
import logging

logger = logging.getLogger("kvtc_utils")

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    TestAscendPerformanceTestCaseBase,
)

KVTC_CACHE_PATH = Path("/root/.cache/KVTC")
KVTC_DATASET_PATH = KVTC_CACHE_PATH / "datasets"

OPENMATH_PARTS = 10
KVTC_DATASET_CONFIG = {
    "openmath": {
        "urls": [
            f"{_HF_BASE}/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00{i:03d}-of-00{OPENMATH_PARTS:03d}.parquet"
            for i in range(OPENMATH_PARTS)
        ],
        "prompt_column": "problem",
    },
    "fineweb": {
        "urls": [
            f"{_HF_BASE}/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2025-26/000_00000.parquet"
        ],
        "prompt_column": "text",
    },
}
KVTC_CALIBRATION_DATASET_INDICES = {
        "openmath": [58206, 30846, 86257, 62172, 89955, 58954, 48176, 71170, 75425, 30081, 17363, 78117, 17710, 42529, 24770, 28411, 63028, 91536, 59447, 47476, 68188, 15946, 11659, 17629, 12965, 64863, 63027, 49969, 43738, 47661, 9966, 43157, 86012, 22209, 34289, 50645, 75863, 43686, 11990, 27702, 39182, 64550, 6052, 50470, 27117, 52861, 55618, 73588, 19120, 59375, 50016, 67330, 57835, 77309, 44169, 80662, 55623, 89991, 59748, 77363, 59130, 86340, 69605, 14566, 73648, 90882, 88252, 8319, 85768, 16577, 74395, 92810, 90737, 9604, 16891, 26183, 74697, 13514, 17839, 39823, 30303, 26688, 78003, 22155, 21407, 49405, 31116, 36531, 89985, 27028, 4083, 89029, 40921, 41374, 93295, 50350, 5875, 13901, 42198, 58722, 4204, 71445, 6256, 23984, 63180, 78849, 6496, 73753, 18227, 25617, 66758, 88668, 8035, 75476, 26824, 30422, 67533, 6349, 10784, 88275, 42286, 56863, 2338, 22046, 34174, 89582, 69772, 63975, 2646, 22673, 44331, 1661, 75562, 77784, 14867, 29348, 49768, 5925],
        "fineweb": [394, 3052, 6299, 9609, 12725, 13821, 698, 4345, 6729, 11361, 12764, 14045, 1569, 4946, 7297, 12089, 12794, 14360, 1912, 5472, 7537, 12528, 12850, 16279, 2563, 5775, 9101, 12652, 13035, 16553, 52, 76, 80, 86, 99, 101, 106, 111, 115, 121, 123, 129, 149, 175, 186, 192, 201, 209, 214, 223, 241, 245, 249, 257, 266, 274, 293, 299, 321, 324, 345, 351, 374, 383, 395, 406, 413, 435, 451, 452, 461, 467, 470, 483, 500, 504, 509, 524, 555, 567, 578, 579, 588, 630, 638, 640, 644, 648, 649, 674, 706, 724, 741, 761, 769, 778, 784, 792, 795, 798, 801, 810, 812, 816, 857, 859, 871, 885, 900, 905, 921, 924, 925, 928, 953, 992, 1024, 1026, 1075, 1100, 1109, 1111, 1113, 1156, 1162, 1174, 1186, 1195, 1198, 1214, 1216, 1224, 1236, 1319, 1404, 1259, 1262, 1282, 1308, 1310, 1316, 1322, 1326, 1340, 1354, 1360, 1372, 1378, 1383, 1384]
}

PROXIES={
    "http": os.environ.get("http_proxy"),
    "https": os.environ.get("https_proxy"),
}

def _file_is_current(local_path: Path, remote_url: str) -> bool:
    if not local_path.exists():
        return False

    response = requests.head(
        remote_url,
        verify=False,
        proxies=PROXIES,
    )
    response.raise_for_status()
    remote_size = response.headers.get("X-Linked-Size")
    return remote_size is not None and local_path.stat().st_size == int(remote_size)


def _download_dataset(dataset_name: str, remote_urls: list[str]) -> None:
    dataset_path = KVTC_DATASET_PATH / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading KVTC calibration dataset {dataset_name}...")

    for index, remote_url in enumerate(remote_urls):
        local_path = dataset_path / f"{dataset_name}_{index}.parquet"
        if _file_is_current(local_path, remote_url):
            logger.debug(f"Skipping {local_path}, already present")
            continue

        logger.debug(f"Downloading {local_path}...")

        response = requests.get(remote_url, verify=False, proxies=PROXIES)
        response.raise_for_status()
        local_path.write_bytes(response.content)


class _AscendKvtcTestCaseBase:
    kvtc_dataset_config = KVTC_DATASET_CONFIG
    kvtc_calibration_dataset_indices = KVTC_CALIBRATION_DATASET_INDICES
    kvtc_dataset_name = None

    @classmethod
    def setUpClass(cls):
        dataset_name = cls.kvtc_dataset_name
        if dataset_name is None:
            raise ValueError("kvtc_dataset_name must be set for a KVTC test")

        remote_urls = cls.kvtc_dataset_config[dataset_name]["urls"]
        _download_dataset(dataset_name, remote_urls)

        cls.kvtc_dataset_path = KVTC_DATASET_PATH / dataset_name
        cls.dataset_path = cls.kvtc_dataset_path
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def load_kvtc_dataset(self, dataset_name=None):
        dataset_name = dataset_name or self.kvtc_dataset_name
        dataset_path = KVTC_DATASET_PATH / dataset_name
        logger.info(f"Loading KVTC calibration dataset: {dataset_name}...")
        dataframes = [
            pd.read_parquet(dataset_path / dataset_file)
            for dataset_file in os.listdir(dataset_path)
        ]
        return pd.concat(dataframes, ignore_index=True)

    def get_kvtc_indices(self, dataset_name=None):
        return self.kvtc_calibration_dataset_indices[dataset_name or self.kvtc_dataset_name]

    def get_kvtc_prompts(self, dataset_name=None):
        dataset_name = dataset_name or self.kvtc_dataset_name
        indices = self.get_kvtc_indices(dataset_name)
        dataset = self.load_kvtc_dataset(dataset_name)
        prompt_column = self.kvtc_dataset_config[dataset_name]["prompt_column"]
        if prompt_column not in dataset.columns:
            raise ValueError(
                f"Dataset {dataset_name} does not contain prompt column "
                f"{prompt_column!r}; available columns: {list(dataset.columns)}"
            )
        prompts = [
            (prompt_id, entry[prompt_column])
            for prompt_id, entry in dataset.iterrows()
            if prompt_id in indices
        ]
        if len(prompts) != len(indices):
            logger.error(f"Some ({len(indices) - len(prompts)}) prompts not found for {dataset_name}")

        logger.info(f"Found {len(prompts)} calibration prompts for {dataset_name}")

        return prompts


class TestAscendPerformanceKvtcTestCaseBase(
    _AscendKvtcTestCaseBase, TestAscendPerformanceTestCaseBase
):
    pass


class TestAscendAccuracyKvtcTestCaseBase(
    _AscendKvtcTestCaseBase, TestAscendAccuracyTestCaseBase
):
    pass
