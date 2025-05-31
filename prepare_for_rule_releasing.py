# %%
# 載入 APK 清單
import polars as pl
import data_preprocess.apk as apk_lib

DATASET_PATHS = ["data/lists/maliciousAPKs_top_0.4_vt_scan_date.csv"]

dataset = pl.concat((apk_lib.load_list(ds) for ds in DATASET_PATHS)).unique(
    "sha256", keep="any"
)
print(dataset.schema)
# 對於每個樣本，透過 VT 取得其 threat_label，並進一步拆解成 major, middle, minor
import data_preprocess.virust_total as vt
import tqdm
import re


with tqdm.tqdm(desc="Getting Threat Label", total=len(dataset)) as progress:
    ThreatLabels = pl.Struct(
        {
            "major_threat_label": pl.String(),
            "middle_threat_label": pl.String(),
            "minor_threat_label": pl.String(),
        }
    )

    def get_threat_label(sha256: str) -> dict[str, str]:
        try:
            report, status = vt.get_virus_total_report(sha256)

            threat_label = (
                report.get("data", {})
                .get("attributes", {})
                .get("popular_threat_classification", {})
                .get("suggested_threat_label", "./")
            )

            major, middle, minor = re.split("[./]", threat_label)
            return {
                "major_threat_label": major,
                "middle_threat_label": middle,
                "minor_threat_label": minor,
            }

        except BaseException as e:
            print(f"Error on {sha256}: {e}")
            return {
                "major_threat_label": "",
                "middle_threat_label": "",
                "minor_threat_label": "",
            }
        finally:
            progress.update()

    dataset = dataset.with_columns(
        pl.col("sha256")
        .map_elements(get_threat_label, return_dtype=ThreatLabels, strategy="threading")
        .alias("threat_labels")
    ).unnest("threat_labels")


dataset = dataset.with_columns(
    pl.col("middle_threat_label").str.replace("^kungfu$", "droidkungfu")
)

print(dataset.head(5))

print("major_threat_label:")
print(dataset["major_threat_label"].value_counts().sort(by="count", descending=True))

print("middle_threat_label:")
print(dataset["middle_threat_label"].value_counts().sort(by="count", descending=True))

# %%
# 挑選一個 middle threat label 作為要釋出規則的惡意程式家族
import click

target_middle_threat_label = click.prompt(
    "Enter target middle threat label",
    type=str,
    default="droidkungfu",
    show_default=True,
)

# 篩選出屬於此家族的樣本
target_dataset = dataset.filter(
    pl.col("middle_threat_label").eq(target_middle_threat_label)
)

print(target_dataset.head(5))
print(f"Num of apk: {len(target_dataset)}")

# %%
# 對於每個樣本，找出通過 5 階段分析的規則
import data_preprocess.rule as rule_lib
import data_preprocess.analysis_result as analysis_result_lib

PATH_TO_RULE_LIST = ["/mnt/storage/rule_score_auto_adjust/data/rule_top_1000.csv"]
rules = pl.concat(list(map(rule_lib.load_list, PATH_TO_RULE_LIST)))
rule_paths = [rule_lib.get(r) for r in rules["rule"].to_list()]

with tqdm.tqdm(desc="Getting stage 5 rules", total=len(target_dataset)) as progress:

    def get_stage_5_rules(sha256: str) -> list[str]:
        apk_path = apk_lib.download(sha256)
        analysis_result = analysis_result_lib.analyze_rules(
            sha256, apk_path, rule_paths, dry_run=True
        )
        progress.update()

        return [rule for rule, stage in analysis_result.items() if stage == 5]

    target_dataset = target_dataset.with_columns(
        pl.col("sha256")
        .map_elements(get_stage_5_rules, return_dtype=pl.List(pl.String()))
        .alias("stage_5_rules")
    )

stage_5_rules = (
    target_dataset.explode("stage_5_rules")
    .unique("stage_5_rules")
    .select("stage_5_rules")
    .rename({"stage_5_rules": "rule"})
)

print("A glance of current apk and rules mapping.")
print(target_dataset.head(5))

print("Num of apk: ", len(target_dataset))
print("Num of rules: ", len(rules))
print("Num of stage 5 rules: ", len(stage_5_rules))
# %%
# Load rules from quark-rules
from pathlib import Path
import polars as pl
import data_preprocess.rule as rule_lib

PATH_TO_QUARK_RULES = Path("/mnt/storage/quark-rules/rules")
rule_paths = [str(p.) for p in PATH_TO_QUARK_RULES.glob("*.json")]
default_rules = pl.DataFrame(rule_paths, schema={"rule_path": pl.String()})

default_rules = default_rules.with_columns(
    pl.col("rule_path")
    .map_elements(rule_lib.get_hash, return_dtype=pl.String())
    .alias("rule_hash")
)

# %%
# 篩除與 Quark 現有規則集重複的規則

stage_5_rules = stage_5_rules.with_columns(
    pl.col("rule")
    .map_elements(
        lambda r: rule_lib.get_hash(rule_lib.get(r)), return_dtype=pl.String()
    )
    .alias("rule_hash")
)

stage_5_rules_removing_default = stage_5_rules.join(
    default_rules, on="rule_hash", how="anti"
)

print("Num of stage 5 rules: ", len(stage_5_rules))
print(
    "Num of stage 5 rules after removing default rules: ",
    len(stage_5_rules_removing_default),
)

# %%
# 對於每個規則，找出其調整後的分數，再依照分數排序
# TODO - 把 model 訓練流程紀錄與結果（包含調整後的分數）記錄到 mlflow 中
import polars as pl

PATH_TO_APK_PREDICTION = "/mnt/storage/rule_score_auto_adjust/apk_prediction.csv"
prediction = (
    pl.read_csv(PATH_TO_APK_PREDICTION, has_header=True)
    .filter(pl.col("sha256").eq("rule_score"))
    .drop(["y_truth", "y_pred_row", "y_pred"])
    .transpose(include_header=True, header_name="rule", column_names="sha256")
)
prediction = prediction.sort(by="rule_score", descending=True)

# %%
stage_5_rules = (
    stage_5_rules.join(prediction, on="rule", how="left")
    .select(["rule", "rule_score"])
    .sort(by="rule_score", descending=True)
)
if stage_5_rules["rule_score"].is_null().any():
    raise ValueError("Some rules do not have scores.")
else:
    print("All rules have scores.")

# %%
# 抓出所有前 20% 分數最高的規則
subset = stage_5_rules.head(int(len(stage_5_rules) * 0.2))
print(subset.head(5))
print(f"Num of rules: {len(subset)}")
print(f"Max score: {subset['rule_score'].max()}")
print(f"Min score: {subset['rule_score'].min()}")
# %%
# 畫出規則分數分布圖
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(subset["rule"], subset["rule_score"])
plt.xlabel("Rule")
plt.ylabel("Rule Score")
plt.title("Rule Score Distribution")
plt.show()

print(f"Num of rules: {len(subset)}")
print(f"Max score: {subset['rule_score'].max()}")
print(f"Min score: {subset['rule_score'].min()}")

# %%
# 對於每個規則，生成一個規則描述
from generate_rule_description import BehaviorDescriptionAgent
import os
import json
import tqdm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
agent = BehaviorDescriptionAgent(OPENAI_API_KEY)

with tqdm.tqdm(desc="Getting rule description", total=len(subset)) as progress:

    def get_description(rule: str):
        progress.update()
        rule_path = rule_lib.get(rule)
        with rule_path.open("r") as content:
            api = json.loads(content.read())["api"]
        return agent.get_description(api)

    subset = subset.with_columns(
        pl.col("rule")
        .map_elements(
            get_description, return_dtype=pl.String(), strategy="thread_local"
        )
        .alias("description")
    )

subset.write_csv("./selected_rule_for_releasing.csv")

# %%
# 請 AI 針對規則進行分類

# %%
# 將 AI 的分類結果 join 進 subset
PATH_TO_AI_CLASSIFICATION = "./selected_rule_from_ai_0.2.csv"
ai_classification = pl.read_csv(
    PATH_TO_AI_CLASSIFICATION, has_header=True, columns=["rule", "category"]
)
ai_classification = ai_classification.join(subset, on="rule", how="left")
ai_classification.write_csv("./selected_rule_for_ai_0.2.csv")

# %%
# 讀取規則的兩個 API 至表格中
import polars as pl
import data_preprocess.rule as rule_lib
import json

ai_classification = pl.read_csv("./selected_rule_for_ai_0.2.csv", has_header=True)

if "api1" in ai_classification.columns:
    ai_classification = ai_classification.drop(["api1", "api2"])


def get_apis(rule: str) -> tuple[str, str]:
    rule_path = rule_lib.get(rule)
    with rule_path.open("r") as content:
        api1, api2 = json.loads(content.read())["api"]
    return {
        "api1": f"{api1['class']}{api1['method']}{api1['descriptor']}",
        "api2": f"{api2['class']}{api2['method']}{api2['descriptor']}",
    }


ai_classification = ai_classification.with_columns(
    pl.col("rule")
    .map_elements(
        get_apis,
        return_dtype=pl.Struct({"api1": pl.String(), "api2": pl.String()}),
        strategy="thread_local",
    )
    .alias("apis")
).unnest("apis")
print(ai_classification.head(5))
ai_classification.write_csv("./selected_rule_for_ai_0.2.csv")
# %%
# 將表格內容寫入規則檔案
from pathlib import Path
import polars as pl
import tqdm
import data_preprocess.rule as rule_lib
import json

rule_table = pl.read_csv(
    "./selected_rule_for_ai_0.2_manually_selected.csv", has_header=True
)

# %%
# 依照大類別與分數由高到低分類規則
rule_table = rule_table.sort(by=["category", "rule_score"], descending=[True, True])
rule_table.write_csv("./selected_rule_for_ai_0.2_manually_selected.csv")

# %%
# 加上編號
STARTING_NUMBER = 212
rule_table = rule_table.with_row_index(name="rule_id", offset=STARTING_NUMBER).with_columns(
    pl.col("rule_id").map_elements(lambda x: f"{x:05d}.json", return_dtype=pl.String())
)

# %%

output_rule_folder = Path("./selected_rule")
output_rule_folder.mkdir(exist_ok=True)

for rule_id, rule, _, _, description, api1, api2, label in tqdm.tqdm(rule_table.rows()):
    rule_path = rule_lib.get(rule)

    with rule_path.open("r") as content:
        rule_content = json.loads(content.read())

    rule_content["crime"] = description
    rule_content["label"] = label.split(",")

    with rule_path.open("w") as content:
        content.write(
            json.dumps(
                rule_content,
                indent=4,
            )
        )
        
    alter_path = output_rule_folder / rule_id
    with alter_path.open("w") as content:
        content.write(
            json.dumps(
                rule_content,
                indent=4,
            )
        )
        

    
# %%
# Load rules from quark-rules
from pathlib import Path
import polars as pl
import data_preprocess.rule as rule_lib
import tqdm
import json

PATH_TO_QUARK_RULES = Path("/mnt/storage/quark-rules/rules")
default_rules = pl.DataFrame([str(p) for p in PATH_TO_QUARK_RULES.glob("*.json")], schema={"rule_path": pl.String()})

with tqdm.tqdm(desc="Getting rule data", total=len(default_rules)) as progress:
    def get_rule_data(rule_path: str):
        with Path(rule_path).open("r") as content:
            rule_content = json.loads(content.read())
        
        progress.update()
        return {
            "api1": rule_content["api"][0]["class"] + rule_content["api"][0]["method"] + rule_content["api"][0]["descriptor"],
            "api2": rule_content["api"][1]["class"] + rule_content["api"][1]["method"] + rule_content["api"][1]["descriptor"],
            "description": rule_content["crime"],
            "label": ",".join(rule_content["label"]),
        }

    default_rules = default_rules.with_columns(
        pl.col("rule_path")
        .map_elements(get_rule_data, return_dtype=pl.Struct({"api1": pl.String(), "api2": pl.String(), "description": pl.String(), "label": pl.String()}), strategy="thread_local")
        .alias("rule_data")
    ).unnest("rule_data")
default_rules.head(5)
default_rules.write_csv("./default_rules.csv")
# %%
