import functools
from typing import Callable
import torch
from data_preprocess import analysis_result, apk
from tqdm import tqdm
from pathlib import Path
import polars as pl
import os

STAGE_WEIGHT_MAPPING = {
    0.0: 0.0,
    1.0: 0.0625,
    2.0: 0.125,
    3.0: 0.25,
    4.0: 0.5,
    5.0: 1.0,
}

DATASET_SCHEMA = {
    "sha256": pl.String,
    "is_malicious": pl.Int32,
}

CACHE_SCHEMA = {
    "index": pl.Int64,
    "rule": pl.String,
    "weights": pl.Float32
}

def has_passing_stage_5_on_any_rule(
    dataset: "ApkDataset", sha256: str
) -> bool:
    indexed_result = dataset.load_cache(sha256)
    if indexed_result is None:
        return True
    stage_distribution = indexed_result["weights"].value_counts()
    num_of_rules_passing_stage_5 = stage_distribution.filter(
        pl.col("weights").eq(1)
    )
    return (
        (not num_of_rules_passing_stage_5.is_empty())
        and num_of_rules_passing_stage_5.item(0, "count") > 0
    )


def all_analysis_result_are_ready(dataset: "ApkDataset", sha256: str) -> bool:
    indexed_result = dataset.load_cache(sha256)
    if indexed_result is None:
        return True
    return indexed_result["weights"].null_count() == 0


class ApkDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_index_files: list[Path],
        rules: set[str],
        apk_filter_funcs: Callable[["ApkDataset", str], bool] = [
            has_passing_stage_5_on_any_rule,
            all_analysis_result_are_ready,
        ],
    ):

        # sha256, is_malicious, rule1, rule2, ...
        self.apk_info = self.load_apk_info(data_index_files)

        # Index Rules
        self.rules = (
            pl.DataFrame(rules, schema={"rule": pl.String})
            .unique(["rule"])
            .with_row_index()
        )

        # Load analysis result for each binary and rule.
        for sha256 in tqdm(self.apk_info["sha256"], desc="Preparing Dataset"):
            indexed_result = self.__prepare_analysis_result(sha256)
            self.save_cache(sha256, indexed_result)

        # Filter out apks based on filter_funcs
        for filter_func in tqdm(apk_filter_funcs, desc="Filtering Dataset"):
            self.filter_apk(filter_func)

        assert not self.apk_info.is_empty(), "APK dataset is empty."
        assert not self.rules.is_empty(), "Rule dataset is empty."

        # balance dataset
        value_count = (
            self.apk_info["is_malicious"]
            .value_counts()
            .sort(by="is_malicious")
        )
        current_benign_count = value_count.item(0, "count")
        current_malicious_count = value_count.item(1, "count")
        print(f"Current benign count: {current_benign_count}")
        print(f"Current malicious count: {current_malicious_count}")

        expected_count = min(current_benign_count, current_malicious_count)

        self.old_apk_info = self.apk_info
        self.apk_info = pl.concat(
            [
                self.old_apk_info.filter(pl.col("is_malicious").eq(0)).sample(
                    n=expected_count, with_replacement=False
                ),
                self.old_apk_info.filter(pl.col("is_malicious").eq(1)).sample(
                    n=expected_count, with_replacement=False
                ),
            ]
        )

        rule_to_filter_out = []
        for rule in self.rules["rule"]:
            has_any_passing_stage_5 = any(
                [
                    any(
                        [
                            float(stage) == 1.0
                            for stage in self.load_cache(sha256)
                            .filter(pl.col("rule").eq(rule))["weights"]
                            .to_list()
                        ]
                    )
                    for sha256 in self.apk_info["sha256"]
                ]
            )

            if not has_any_passing_stage_5:
                print(f"Rule {rule} has no passing stage 5.")
                rule_to_filter_out.append(rule)

        self.rules = self.rules.filter(
            ~pl.col("rule").is_in(rule_to_filter_out)
        )
        print(f"Num of rules: {len(self.rules)}")
        print(f"Num of apks: {len(self.apk_info)}")

    def filter_apk(
        self, filter_func: Callable[["ApkDataset", str], bool]
    ) -> None:
        partial_filter_func = functools.partial(
            filter_func, self
        )
        apk_to_drop = self.apk_info.filter(
            pl.col("sha256").map_elements(
                partial_filter_func, return_dtype=pl.Boolean
            ).not_()
        )
        print(
            f"Dropping APKs {filter_func.__name__}:",
            apk_to_drop["sha256"].to_list(),
        )

        self.apk_info = self.apk_info.filter(
            pl.col("sha256").is_in(apk_to_drop["sha256"]).not_()
        )

    @staticmethod
    def load_apk_info(datasets: list[Path]):
        __dataset_df = [
            pl.read_csv(
                dataset_path,
                has_header=True,
                columns=["sha256", "is_malicious"],
                schema_overrides=DATASET_SCHEMA,
            )
            for dataset_path in datasets
        ]
        dataset = pl.concat(__dataset_df)

        # Filter out apk not exits
        dataset = dataset.filter(
            pl.col("sha256").map_elements(
                lambda x: apk.get(x).exists(), return_dtype=pl.Boolean
            )
        )

        # Filter out analysis result not exits
        dataset = dataset.filter(
            pl.col("sha256").map_elements(
                lambda x: analysis_result.get_file(x).exists(),
                return_dtype=pl.Boolean,
            )
        )

        return dataset

    def __prepare_analysis_result(self, sha256: str) -> pl.DataFrame:
        result = (
            analysis_result.load_as_dataframe(sha256)
            .rename(
                {
                    "rule_name": "rule",
                }
            )
            .with_columns(
                pl.col("passing_stage").map_elements(
                    lambda x: STAGE_WEIGHT_MAPPING.get(float(x), 0.0),
                    return_dtype=pl.Float32,
                ).alias("weights"),
                pl.col("rule").map_elements(
                    lambda x: x.split("/")[-1],
                    return_dtype=pl.String,
                ),
            )
        )

        # drop row if the column "rule" is duplicate
        result = result.unique(subset=["rule"])

        indexed_result = self.rules.join(
            result, on="rule", how="left", maintain_order="left"
        )

        self.save_cache(sha256, indexed_result)

        indexed_result_torch = (
            indexed_result.select("weights")
            .transpose()
            .to_torch(dtype=pl.Float32)
        )

        assert (
            indexed_result_torch.numel() != 0
        ), f"Indexed result for {sha256} is empty. {analysis_result.get_file(sha256)}"
        for i in range(indexed_result_torch.shape[1]):
            assert not (
                indexed_result.filter(pl.col("index").eq(i))
            ).is_empty(), f"{sha256} Missing index:{i}"

        assert indexed_result_torch.shape[1] == len(
            self.rules
        ), f"{sha256=}, {indexed_result_torch.shape=}, {len(self.rules)=}"

        return indexed_result

    @staticmethod
    def cache_folder() -> Path:
        folder = Path(os.getenv("DATASET_FOLDER")) / "cache"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @staticmethod
    def save_cache(sha256: str, indexed_result: pl.DataFrame) -> None:
        cache_file = ApkDataset.cache_folder() / f"{sha256}.csv"
        indexed_result.select(CACHE_SCHEMA.keys()).write_csv(cache_file, include_header=True)

    @staticmethod
    def load_cache(sha256: str) -> pl.DataFrame | None:
        cache_file = ApkDataset.cache_folder() / f"{sha256}.csv"
        if not cache_file.exists():
            return None
        return pl.read_csv(cache_file, has_header=True, schema=CACHE_SCHEMA)

    @functools.cache
    def __getitem__(self, index) -> torch.Tensor:
        sha256 = self.apk_info.item(row=index, column="sha256")

        indexed_result = self.load_cache(sha256)

        indexed_result_torch = (
            indexed_result.select("weights")
            .transpose()
            .to_torch(dtype=pl.Float32)
        )

        expected_score = torch.tensor(
            self.apk_info[index]["is_malicious"], dtype=torch.float32
        )

        return indexed_result_torch, expected_score

    def __len__(self) -> int:
        return len(self.apk_info)

    def verify(self) -> bool:
        [n for n in tqdm(self, desc="Checking Dataset")]
