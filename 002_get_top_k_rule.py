import json
from pathlib import Path
from typing import Counter
from tqdm import tqdm
import polars as pl

RULE_FOLDER = Path("/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/analysis_results")

results = list(RULE_FOLDER.glob("[!.]*.json"))

# error_json_list = []
# for result in tqdm(results, desc="Check if File are Json format"):
#     if not result.is_file():
#         print(f"Invalid file path: {result}")
#         continue
#     try:
#         with result.open("r") as content:
#             json.load(content)
#     except json.JSONDecodeError:
#         print(f"Invalid JSON file: {result}")
#         error_json_list.append(result)
#         continue

# if error_json_list:
#     print(error_json_list)
#     import sys
#     sys.exit(0)


ruleCounter = Counter()

for rule in tqdm(results):
    with rule.open("r") as content:
        try:
            analysis_result = json.load(content)

            assert isinstance(analysis_result, list)

            if len(analysis_result) == 0 \
                or (not isinstance(analysis_result[0], list)) \
                or len(analysis_result[0]) == 1:
                # no result or result in old format
                continue

            pass_stage_5_record = [
                rule_path
                for rule_path, stage in analysis_result
                if stage >= 5
            ]

            ruleCounter +=  Counter(pass_stage_5_record)
        except UnicodeDecodeError as e:
            print(f"Error on {rule}")


rule_count = (
    pl.DataFrame([i for i in ruleCounter.items()], strict=False)
    .transpose()
    .select([pl.col("column_0").alias("rule_path"), pl.col("column_1").str.to_integer().alias("number of passing stage 5")])
    .filter(pl.col("rule_path").str.contains("final_rules"))
    .sort(pl.col("number of passing stage 5"), descending=True)
    )
rule_count.write_csv("data/rule_count.csv", include_header=True)
rule_count.head(1000).write_csv("data/rule_top_1000.csv")


rule_count = pl.read_csv("data/rule_top_1000.csv", has_header=True)
local_rule_folder = Path("/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/final_rules")
link_folder = Path("data/rule_top_1000")
link_folder.mkdir(exist_ok=True)
for row in tqdm(rule_count.to_dicts(), desc="Linking Rules"):
    remote_rule_path = Path(row["rule_path"])

    local_rule_path = local_rule_folder / remote_rule_path.name
    assert local_rule_path.exists, f"Rule {str(local_rule_path)} is not exists. Abort."

    link_path = link_folder / remote_rule_path.name

    if link_path.exists(follow_symlinks=False):
        link_path.unlink()

    link_path.symlink_to(local_rule_path)

