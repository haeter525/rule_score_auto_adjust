# %%
from contextlib import suppress
import itertools
import json
from pathlib import Path
import os
from typing import Tuple
import dotenv
import polars as pl
dotenv.load_dotenv()

def load_analysis_result(sha256: str, analysis_result_folder: str = os.getenv("ANALYSIS_RESULT_FOLDER")) -> list[Tuple[str, int]]:
    def parse_item(item: list[str, int] | str) -> list[str, int]:
        return (item, -1) if isinstance(item, str) else item


    analysis_result = Path(analysis_result_folder) / f"{sha256}.apk_progress.json"

    try:
        with analysis_result.open("r") as file:
            content = json.load(file)
            return [parse_item(item) for item in content]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        print(f"Failed to parser JSON file: {analysis_result}")
        return []

def get_apk_trained(dataset_path: str) -> pl.DataFrame:
    sha256_apk_for_train = pl.read_csv("./top_20_trained_apk.csv", has_header=True)
    benign_list = pl.read_csv(dataset_path, has_header=True)
    benign_trained = sha256_apk_for_train.join(benign_list, on="sha256", how="inner")

    return benign_trained
    

def get_match_5_and_not_match_5_rule_set(dataset_path: str, rule_list_path: str = None):
    benign_trained = get_apk_trained(dataset_path)

    # 取得規則路徑集合
    if rule_list_path is not None:
        import polars as pl

        rule_score = pl.read_csv("optimized_rule_score_model_20250407_232924_40.csv", has_header=True).with_row_index()
        rule_index = pl.read_csv(rule_list_path, has_header=True)
        combine = rule_score.join(rule_index, on="index", how="left")

        del rule_score
        del rule_index
        rule_path_set = set(combine["rule_path"])

    # 計算 match benign 5 stage 的規則
    analysis_results = itertools.chain(*(
        load_analysis_result(sha256)
        for sha256 in benign_trained["sha256"]
    ))

    matching_stage_5_counter = set()
    matching_no_stage_5_counter = set()

    with suppress(FileNotFoundError), suppress(json.JSONDecodeError):
        for rule_path, stage in analysis_results:
            if stage == -1:
                continue

            if stage == 5:
                matching_stage_5_counter.add(rule_path)
            else:
                matching_no_stage_5_counter.add(rule_path)


    if rule_list_path is not None:
        matching_stage_5_counter.intersection_update(rule_path_set)
        matching_no_stage_5_counter.intersection_update(rule_path_set)

    matching_no_stage_5_counter.difference_update(matching_stage_5_counter)

    return [matching_stage_5_counter, matching_no_stage_5_counter]

malware_results = get_match_5_and_not_match_5_rule_set("./data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv")
benign_results = get_match_5_and_not_match_5_rule_set("./data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv")

print(f"Benign  5    : {len(benign_results[0]):4}")
print(f"Benign  Not 5: {len(benign_results[1]):4}")
print(f"Malware 5    : {len(malware_results[0]):4}")
print(f"Malware Not 5: {len(malware_results[1]):4}")
print()

print(f"Benign     5 & Malware     5: {len(malware_results[0].intersection(benign_results[0])):4}")
print(f"Benign     5 & Malware Not 5: {len(malware_results[1].intersection(benign_results[0])):4}")
print(f"Benign Not 5 & Malware     5: {len(malware_results[0].intersection(benign_results[1])):4}")
print(f"Benign Not 5 & Malware Not 5: {len(malware_results[1].intersection(benign_results[1])):4}")

# %%

rules = malware_results[0].intersection(benign_results[1])
rules
# %%
from pathlib import Path
rules_not_exist = [
    rule for rule in rules if not Path(rule).exists()
]
assert not rules_not_exist
# %%
import polars as pl
rule_names = [Path(rule).name for rule in rules]

table = pl.DataFrame(rule_names, schema={"rule":str}).with_row_index()
table.head()
# %%
malware_raw_table = get_apk_trained("./data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv")
benign_raw_table = get_apk_trained("./data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv")

def aggregate_analysis_result(sha256_list: list[str], output_csv:str):
    malware_table = table.clone()
    for sha256 in sha256_list:
        analysis_result = [
            (Path(rule_path).name, result)
            for rule_path, result in load_analysis_result(sha256)
            if isinstance(result, int)
        ]
        print(f"{sha256=}")
        analysis_result_table = pl.DataFrame(analysis_result, schema={"rule":str, "stage": pl.Int8})
        malware_table = malware_table.join(analysis_result_table, on="rule", how="left").rename({"stage": sha256}).with_columns(pl.col(sha256).fill_null(-1))
        
    malware_table.write_csv(output_csv, include_header=True)
    return malware_table

aggregate_analysis_result(malware_raw_table["sha256"], "/mnt/storage/data/rule_list/selected_rules_result_on_malware.csv")
aggregate_analysis_result(benign_raw_table["sha256"], "/mnt/storage/data/rule_list/selected_rules_result_on_benign.csv")

# %%
