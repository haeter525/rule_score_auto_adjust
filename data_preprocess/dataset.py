import functools
from typing import Callable, Iterable, override
import torch
from data_preprocess import analysis_result, apk
from tqdm import tqdm
from pathlib import Path
import polars as pl
import os

STAGE_WEIGHT_MAPPING = {
    0.0: (2**0) / (2**5),
    1.0: (2**1) / (2**5),
    2.0: (2**2) / (2**5),
    3.0: (2**3) / (2**5),
    4.0: (2**4) / (2**5),
    5.0: (2**5) / (2**5),
}

CACHE_SCHEMA = {"index": pl.Int64, "rule": pl.String, "weights": pl.Float32}


def apk_have_analysis_results(dataset: "ApkDataset", sha256: str) -> bool:
    return analysis_result.get_file(sha256).exists()


def apk_exists(dataset: "ApkDataset", sha256: str) -> bool:
    return apk._get_path(sha256).exists()


def apk_has_passing_stage_5_on_any_rule(
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
        not num_of_rules_passing_stage_5.is_empty()
    ) and num_of_rules_passing_stage_5.item(0, "count") > 0


def all_analysis_result_are_ready(dataset: "ApkDataset", sha256: str) -> bool:
    indexed_result = dataset.load_cache(sha256)
    if indexed_result is None:
        return True
    return indexed_result["weights"].null_count() == 0


def drop_benign_apks_whose_analysis_result_is_same_with_malware(
    dataset: "ApkDataset", sha256: str
) -> bool:
    APK_TO_DROP = {
        "0002AAE56E7F80D54F6AA3EB7DA8BA9A28E58A3ECE3B15171FCF5EBEE30B95FE",
        "0002196E3C3904632CD03D32AB649C63E02B320E19D13134AF9196BDD69FEA38",
        "000186F14A4B204F0F40627EA5480004C939AC80D8BAA203A510E55BD6FE5C78",
        "000478113CC2750DF15EE09CB81B7E17D1C27C85F26D075D487D3AD6CAD51BB8",
        "00046289C6B726BB7600933EE6A83BED17C35F18D37C3795A59A8AE526BFB6CE",
        "0000499EBC668A309BC345F0F9B7C32CEA341F8FCB35F29F66C88F87A425CE58",
    }

    return sha256 not in APK_TO_DROP


class ApkDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_index_files: list[Path],
        rules: Iterable[str],
        apk_filter_funcs: list[Callable[["ApkDataset", str], bool]] = [
            apk_exists,
            apk_have_analysis_results,
            apk_has_passing_stage_5_on_any_rule,
            # all_analysis_result_are_ready,
            drop_benign_apks_whose_analysis_result_is_same_with_malware,
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

        need_to_prepare_dataset = True
        if self.hash_storage().exists():
            with self.hash_storage().open("r") as file:
                need_to_prepare_dataset = (
                    file.readline().strip() == str(self.__hash__())
                )

        with self.hash_storage().open("w") as hash_file:
            hash_file.write(str(self.__hash__()))

        # Load analysis result for each binary and rule.
        if need_to_prepare_dataset:
            for sha256 in tqdm(
                self.apk_info["sha256"], desc="Preparing Dataset"
            ):
                indexed_result = self.__prepare_analysis_result(sha256)
                self.save_cache(sha256, indexed_result)

        # Filter out apks based on filter_funcs
        for filter_func in tqdm(
            iterable=apk_filter_funcs, desc="Filtering Dataset"
        ):
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
            only_pass_stage_5_on_malware = any(
                [
                    any(
                        [
                            stage and float(stage) == 1.0
                            for stage in self.load_cache(sha256)
                            .filter(pl.col("rule").eq(rule))["weights"]
                            .to_list()
                        ]
                    )
                    for sha256 in self.apk_info["sha256"]
                ]
            )

            if not only_pass_stage_5_on_malware:
                print(f"Rule {rule} has no passing stage 5.")
                rule_to_filter_out.append(rule)

        self.rules = self.rules.filter(
            ~pl.col("rule").is_in(rule_to_filter_out)
        )
        print(f"Num of rules: {len(self.rules)}")
        print(f"Num of apks: {len(self.apk_info)}")

    @override
    def __hash__(self) -> int:
        hash_value = sum(hash(val) for val in self.apk_info["sha256"])
        hash_value += sum(hash(val) for val in self.rules["rule"])
        
        return hash_value

    def hash_storage(self) -> Path:
        return self.working_folder() / "hash"

    def filter_apk(
        self, filter_func: Callable[["ApkDataset", str], bool]
    ) -> None:
        partial_filter_func = functools.partial(filter_func, self)
        apk_to_drop = self.apk_info.filter(
            pl.col("sha256")
            .map_elements(partial_filter_func, return_dtype=pl.Boolean)
            .not_()
        )

        log_file = self.log_folder() / f"filter_{filter_func.__name__}.csv"
        apk_to_drop.write_csv(log_file, include_header=True)

        print(f"Dropping {len(apk_to_drop)} APKs, see {log_file} for details")

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
                schema_overrides=apk.APK_SCHEMA,
            )
            for dataset_path in datasets
        ]
        dataset = pl.concat(__dataset_df)

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
                pl.col("passing_stage")
                .map_elements(
                    lambda x: STAGE_WEIGHT_MAPPING.get(float(x), 0.0),
                    return_dtype=pl.Float32,
                )
                .alias("weights"),
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

    def get_apk_info(self) -> pl.DataFrame:
        dataframe = (
            self.load_cache(sha256).rename({"weights": sha256})
            for sha256 in self.apk_info["sha256"]
        )

        return pl.concat(dataframe, how="horizontal")

    @staticmethod
    def __createIfNotExists(folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def working_folder(self) -> Path:
        return self.__createIfNotExists(
            Path(os.getenv("DATASET_FOLDER", "NOT_DEFINED")) / type(self).__name__
        )

    def cache_folder(self) -> Path:
        return self.__createIfNotExists(self.working_folder() / "cache")

    def log_folder(self) -> Path:
        return self.__createIfNotExists(self.working_folder() / "log")

    def save_cache(self, sha256: str, indexed_result: pl.DataFrame) -> None:
        cache_file = self.cache_folder() / f"{sha256}.csv"
        indexed_result.select(CACHE_SCHEMA.keys()).write_csv(
            cache_file, include_header=True
        )

    def load_cache(self, sha256: str) -> pl.DataFrame:
        cache_file = self.cache_folder() / f"{sha256}.csv"
        return pl.read_csv(cache_file, has_header=True, schema=CACHE_SCHEMA)

    @functools.cache
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        sha256 = self.apk_info.item(row=index, column="sha256")

        indexed_result = self.load_cache(sha256)

        indexed_result_torch = (
            indexed_result.select("weights")
            .fill_nan(0.0)
            .fill_null(0.0)
            .transpose()
            .to_torch(dtype=pl.Float32)
        ).view(-1)

        expected_score = torch.tensor(
            self.apk_info.item(index, "is_malicious"), dtype=torch.float32
        )

        return indexed_result_torch, expected_score

    def __len__(self) -> int:
        return len(self.apk_info)

    def verify(self) -> bool:
        return all([n for n in tqdm(iterable=self, desc="Checking Dataset")])
